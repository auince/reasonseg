from __future__ import annotations
import json,torch,torch.nn.functional as F,sys
from pathlib import Path;from collections import Counter
from tqdm import tqdm
import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reasonseg.backends._bootstrap import ensure_root_model_package_loaded
from reasonseg.backends.beit3.modeling_utils import _get_large_config,BEiT3Wrapper
from reasonseg.modeling.evf_sam2 import _load_state_dict_or_raise,_BEIT_PRETRAIN_UNEXPECTED_KEYS
from transformers import AutoTokenizer
from model.BIOtagging.bio_schema import BIO_TAG_TO_ID,ID_TO_BIO_TAG,structure_to_bio_tags,tokens_to_bio_labels
from model.BIOtagging.query_parser_head import QueryParserHead

ROOT=Path(__file__).resolve().parents[1];device="cuda";N=14;B=256;M=18;OUT=ROOT/"model/BIOtagging/outputs/stage1_final"
OUT.mkdir(parents=True,exist_ok=True)

beit3=BEiT3Wrapper(_get_large_config()).to(device).eval()
st=torch.load(str(ROOT/"checkpoints/beit3_large_patch16_224.pth"),map_location="cpu")
_load_state_dict_or_raise(beit3,st["model"],context="BEiT3",allowed_unexpected_keys=_BEIT_PRETRAIN_UNEXPECTED_KEYS)
for p in beit3.parameters():p.requires_grad=False
tok=AutoTokenizer.from_pretrained(str(ROOT/"models/evf-sam2-multitask"),use_fast=True)

anns=json.loads((ROOT/"model/BIOtagging/data/llm_annotations_3k_reviewed.json").read_text())
qs=json.loads((ROOT/"model/BIOtagging/data/refcoco_queries_for_annotation.json").read_text())
lu={q.strip().lower():a for q,a in zip(qs,anns)}
TQ=qs[:2500];VQ=qs[2500:2800]

def pre(ql):
    H,MK=[],[]
    for q in tqdm(ql,desc="BEiT3"):
        e=tok(q,return_tensors="pt",max_length=M,truncation=True)
        with torch.no_grad():o=beit3.beit3(visual_tokens=None,textual_tokens=e["input_ids"].to(device),text_padding_position=~e["attention_mask"].to(device))
        h=o["encoder_out"][0].cpu()
        s=h.size(0);mk=torch.ones(M,dtype=torch.bool);mk[s:]=False
        if s<M:h=F.pad(h,(0,0,0,M-s))
        H.append(h[:M]);MK.append(mk)
    return torch.stack(H),torch.stack(MK)

def lab(ql):
    r=[]
    for q in tqdm(ql,desc="Labels"):
        s=lu[q.strip().lower()];wt=q.lower().split();wb=structure_to_bio_tags(s,wt)
        e=tok(q,return_tensors="pt",max_length=M,truncation=True);wids=e.word_ids()
        L=[-100]*M;last=-1
        for i,wid in enumerate(wids[:M]):
            if wid is None or wid>=len(wb):L[i]=-100
            elif wid!=last:last=wid;L[i]=BIO_TAG_TO_ID.get(wb[wid],0)
            else:it=wb[wid].replace("B-","I-") if wb[wid].startswith("B-") else wb[wid];L[i]=BIO_TAG_TO_ID.get(it,0)
        r.append(torch.tensor(L))
    return torch.stack(r)

print("Precomputing...");TH,TM=pre(TQ);VH,VM=pre(VQ);TL=lab(TQ);VL=lab(VQ)
print(f"Train:{TH.shape} Val:{VH.shape}")

cnt=Counter()
for q in TQ:
    for l in tokens_to_bio_labels(q.lower().split(),lu[q.strip().lower()]):
        if l<N:cnt[l]+=1
t=sum(cnt.values());cw=torch.ones(N)*0.1
for c,n in cnt.items():
    if n>0:cw[c]=max(0.05,t/(N*n))
cw[0]*=0.3;cw=cw.to(device)

m=QueryParserHead(1024,N,2,8,2048,0.1).to(device)
opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)

print("Training...");losses=[];accs=[];best=0
for ep in range(30):
    m.train();p=torch.randperm(len(TH));el=0.0
    for i in range(0,len(TH),B):
        idx=p[i:i+B];h=TH[idx].to(device);l=TL[idx].to(device);mk=TM[idx].to(device)
        lo=m(h,mk);loss=F.cross_entropy(lo.view(-1,N),l.view(-1),ignore_index=-100,weight=cw)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        el+=loss.item()
    losses.append(el/max(1,len(TH)//B))
    m.eval();c=0;t_=0
    with torch.no_grad():
        for i in range(0,len(VH),B):
            h=VH[i:i+B].to(device);l=VL[i:i+B].to(device);mk=VM[i:i+B].to(device)
            p=m(h,mk).argmax(-1);msk=l>=0
            c+=(p[msk]==l[msk]).sum().item();t_+=msk.sum().item()
    acc=c/max(t_,1);accs.append(acc)
    if acc>best:best=acc;torch.save(m.state_dict(),OUT/"parser_head_best.pt")
    print(f"Epoch {ep+1:3d} loss={losses[-1]:.4f} val={acc:.4f} best={best:.4f}")

cm=torch.zeros(N,N,dtype=torch.long)
with torch.no_grad():
    for i in range(0,len(VH),B):
        h=VH[i:i+B].to(device);l=VL[i:i+B];mk=VM[i:i+B].to(device)
        p=m(h,mk).argmax(-1).cpu()
        for pp,ll in zip(p.view(-1),l.view(-1)):
            if ll>=0:cm[ll,pp]+=1
lines=["═"*50,f"  Best val acc: {best:.4f}",f"  Model: {OUT/'parser_head_best.pt'}","","  Per-Tag Metrics:","  "+f"{'Tag':<16s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Sup':>6s}"]
for i in range(N):
    tp=cm[i,i].item();fp=cm[:,i].sum().item()-tp;fn=cm[i,:].sum().item()-tp
    prec=tp/(tp+fp+1e-8);rec=tp/(tp+fn+1e-8);f1=2*prec*rec/(prec+rec+1e-8);sup=int(cm[i,:].sum())
    if sup>0:lines.append(f"  {ID_TO_BIO_TAG[i]:<16s} {prec:7.4f} {rec:7.4f} {f1:7.4f} {sup:6d}")
rpt="\n".join(lines);print(rpt);(OUT/"metrics.txt").write_text(rpt)

fig,ax1=plt.subplots(figsize=(10,5))
ax1.plot(losses,color="steelblue",lw=1.5,label="Loss");ax1.legend(loc="upper left")
ax2=ax1.twinx();ax2.plot(accs,color="darkorange",marker="o",ms=3,lw=1.5,label="Val Acc");ax2.legend(loc="upper right")
fig.tight_layout();fig.savefig(OUT/"stage1_loss.png",dpi=150)
print(f"\nSaved to {OUT}")
