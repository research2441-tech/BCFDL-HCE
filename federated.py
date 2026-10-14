from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import math
import random
import torch
import torch.nn.functional as F
from hce import HashContextEncoder, hce_loss, class_prototypes, aggregate_prototypes
from privacy import state_delta, clip_update, add_gaussian_noise, mask_update, secure_aggregate, weight_update
from security import robust_filter
from utils import tensor_state_nbytes

HCE_PREFIX = "hce."


@dataclass
class ClientResult:
    client_id: int
    update: dict
    samples: int
    train_loss: float
    upload_bytes: int
    prototypes: dict | None = None
    prototype_counts: dict | None = None
    prototype_bytes: int = 0


def clone_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def joint_state(model, hce=None) -> dict:
    """Global state exchanged in a round: classifier parameters plus, when enabled, the HCE head."""
    s = clone_state(model)
    if hce is not None:
        s.update({HCE_PREFIX + k: v for k, v in clone_state(hce).items()})
    return s


def apply_delta(model, delta, prefix: str = ""):
    s=model.state_dict()
    model.load_state_dict({k: s[k].detach().cpu()+delta[prefix+k] for k in s})


def weighted_average(updates: list[tuple[dict,int]]) -> dict:
    total=sum(n for _,n in updates); keys=updates[0][0]
    return {k:sum(u[k]*n for u,n in updates)/total for k in keys}


def prototype_payload_bytes(prototypes: dict | None, bits: int) -> int:
    """Packed B-bit HCE prototype per class plus an 8-byte sample count per class."""
    if not prototypes: return 0
    return int(len(prototypes) * ((bits + 7) // 8 + 8))


def train_client(model, loader, cfg, device, global_state, client_id: int, round_id: int,
                 method: str = "fedavg", hce: HashContextEncoder | None = None,
                 global_prototypes=None, previous_global=None, client_aux=None) -> ClientResult:
    model=deepcopy(model).to(device); model.train()
    hce=deepcopy(hce).to(device) if hce is not None else None
    if hce is not None: hce.train()
    params=list(model.parameters()) + (list(hce.parameters()) if hce is not None else [])
    try:  # fused kernel changes speed only, not the Adam update rule
        opt=torch.optim.Adam(params, lr=float(cfg["optimization"]["learning_rate"]), weight_decay=float(cfg["optimization"].get("weight_decay",0)), fused=True)
    except (RuntimeError, TypeError):
        opt=torch.optim.Adam(params, lr=float(cfg["optimization"]["learning_rate"]), weight_decay=float(cfg["optimization"].get("weight_decay",0)))
    ce=torch.nn.CrossEntropyLoss(); losses=[]; local_steps=0
    prox_mu=float(cfg.get("baselines",{}).get("fedprox_mu",.01))
    moon_mu=float(cfg.get("baselines",{}).get("moon_mu",.1))
    dyn_alpha=float(cfg.get("baselines",{}).get("feddyn_alpha",.01))
    pcounts={}; proto_acc=[]

    for _ in range(int(cfg["federated"]["local_epochs"])):
        for x,y in loader:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            logits,z=model(x,return_embedding=True); loss=ce(logits,y)
            if method=="fedprox":
                prox=0.0
                for (name,p) in model.named_parameters():
                    if name in global_state: prox=prox+((p-global_state[name].to(device))**2).sum()
                loss=loss+0.5*prox_mu*prox
            if method=="moon" and previous_global is not None:
                with torch.no_grad(): _,zg=previous_global.to(device)(x,return_embedding=True)
                loss=loss+moon_mu*(1-F.cosine_similarity(z,zg,dim=1).mean())
            if method=="feddyn" and client_aux:
                linear=0.0; quad=0.0
                for name,p in model.named_parameters():
                    if name in global_state:
                        linear += (p*client_aux.get(name,torch.zeros_like(p).cpu()).to(device)).sum()
                        quad += ((p-global_state[name].to(device))**2).sum()
                loss=loss+dyn_alpha*(linear+0.5*quad)
            if hce is not None:
                relaxed=hce(z)
                hl=hce_loss(relaxed,y,global_prototypes,
                            float(cfg["hce"]["similarity_weight"]),float(cfg["hce"]["quantization_weight"]),float(cfg["hce"]["prototype_weight"]))
                loss=loss+hl.total
                pp=class_prototypes(relaxed.detach(),y)
                cnt={int(c.item()):int((y==c).sum().item()) for c in y.unique()}; proto_acc.append((pp,cnt))
                for c,n in cnt.items(): pcounts[c]=pcounts.get(c,0)+n
            loss.backward(); opt.step(); losses.append(loss.detach()); local_steps+=1
    local=joint_state(model, hce)
    delta=state_delta(local,global_state)
    if method=="fednova":
        tau=max(1,local_steps); delta={k:v/tau for k,v in delta.items()}
    privacy=cfg.get("privacy",{})
    if privacy.get("clipping",False): delta,_=clip_update(delta,float(privacy.get("clip_norm",1.0)))
    if privacy.get("differential_privacy",False):
        std=float(privacy.get("noise_multiplier",0))*float(privacy.get("clip_norm",1.0))
        delta=add_gaussian_noise(delta,std,seed=int(cfg["experiment"]["seed"])+round_id*1000+client_id)
    protos=aggregate_prototypes(proto_acc) if proto_acc else None
    pbytes=prototype_payload_bytes(protos, int(cfg["hce"]["bits"])) if hce is not None else 0
    return ClientResult(client_id,delta,len(loader.dataset),float(torch.stack(losses).mean()) if losses else 0.0,
                        tensor_state_nbytes(delta)+pbytes,protos,pcounts or None,pbytes)


class FederatedTrainer:
    def __init__(self, model, client_loaders, cfg, device, hce: HashContextEncoder|None=None, method: str|None=None):
        self.model=model.to(device); self.loaders=client_loaders; self.cfg=cfg; self.device=device
        self.hce=hce.to(device) if hce is not None else None
        self.method=(method or cfg["federated"].get("aggregation","fedavg")).lower(); self.history=[]; self.global_prototypes=None
        self.previous_global=None; self.client_aux={i:{} for i in range(len(client_loaders))}

    def _participants(self, round_id):
        n=len(self.loaders); frac=float(self.cfg["federated"].get("client_fraction",1.0)); k=max(1,math.ceil(n*frac))
        ids=list(range(n))
        if self.cfg["federated"].get("cluster_training",False):
            clusters=max(1,int(self.cfg["federated"].get("clusters",2)))
            buckets=[ids[i::clusters] for i in range(clusters)]
            if self.cfg["federated"].get("odd_even_schedule",False): ids=buckets[round_id%clusters]
        rng=random.Random(int(self.cfg["experiment"]["seed"])+round_id); rng.shuffle(ids)
        return ids[:min(k,len(ids))]

    def round(self, round_id: int):
        global_state=joint_state(self.model, self.hce); ids=self._participants(round_id); results=[]
        down_model=tensor_state_nbytes(global_state)
        down_proto=prototype_payload_bytes(self.global_prototypes, int(self.cfg["hce"]["bits"])) if self.hce is not None else 0
        for cid in ids:
            r=train_client(self.model,self.loaders[cid],self.cfg,self.device,global_state,cid,round_id,self.method,self.hce,self.global_prototypes,self.previous_global,self.client_aux[cid])
            results.append(r)
        upload=sum(r.upload_bytes for r in results); proto_up=sum(r.prototype_bytes for r in results)
        updates=[r.update for r in results]
        if self.cfg.get("security",{}).get("robust_filter",False) and len(updates)>=3:
            keep=robust_filter(updates,float(self.cfg["security"].get("max_update_norm_zscore",3.5)),float(self.cfg["security"].get("cosine_floor",-.25)))
            results=[r for r,k in zip(results,keep) if k]
        privacy=self.cfg.get("privacy",{})
        if privacy.get("secure_aggregation",False) and len(results)>1:
            pids=[r.client_id for r in results]; masked=[]
            for r in results:
                masked.append((mask_update(weight_update(r.update,r.samples),r.client_id,pids,round_id,int(self.cfg["experiment"]["seed"])),r.samples))
            agg=secure_aggregate(masked)
        else: agg=weighted_average([(r.update,r.samples) for r in results])
        if self.method=="fednova":
            agg={k:v*max(1,int(self.cfg["federated"]["local_epochs"])) for k,v in agg.items()}
        self.previous_global=deepcopy(self.model).cpu() if self.method=="moon" else self.previous_global
        apply_delta(self.model,agg); self.model.to(self.device)
        if self.hce is not None:
            apply_delta(self.hce,agg,HCE_PREFIX); self.hce.to(self.device)
        if self.method=="scaffold":
            for r in results: self.client_aux[r.client_id]={k:(r.update[k]-agg[k]).detach().cpu() for k in agg}
        if self.method=="feddyn":
            for r in results:
                prev=self.client_aux[r.client_id]
                self.client_aux[r.client_id]={k:prev.get(k,torch.zeros_like(v))+r.update[k] for k,v in r.update.items()}
        proto_items=[(r.prototypes,r.prototype_counts) for r in results if r.prototypes]
        if proto_items: self.global_prototypes=aggregate_prototypes(proto_items)
        record={"round":round_id,"participants":[r.client_id for r in results],
                "scheduled":len(ids),
                "train_loss":sum(r.train_loss for r in results)/max(1,len(results)),
                "upload_bytes":int(upload),"download_bytes":int(len(ids)*(down_model+down_proto)),
                "prototype_upload_bytes":int(proto_up)}
        self.history.append(record); return record
