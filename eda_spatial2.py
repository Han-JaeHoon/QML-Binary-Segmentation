"""
OSCD EDA -- Part 3: two final confirmations before model design.
1. Spatial sweep on the MEDIAN-CORRECTED |dB| (do median-correction and spatial
   context stack?).
2. Hard-negative analysis under city-grouped CV (leave-region-out), not random.

Usage: python eda_spatial2.py --data_dir /path/to/OneraDataset
"""
import argparse, os
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, maximum_filter

BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
TRAIN = ["aguasclaras","bercy","bordeaux","nantes","paris","rennes","saclay_e",
         "abudhabi","cupertino","pisa","beihai","hongkong","beirut","mumbai"]
BI = {b:i for i,b in enumerate(BANDS)}
RNG = np.random.RandomState(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--out", default="results")
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, GroupKFold

    img = os.path.join(args.data_dir,"images","Onera Satellite Change Detection dataset - Images")
    lbl = os.path.join(args.data_dir,"train_labels","Onera Satellite Change Detection dataset - Train Labels")
    ld = lambda c,s: np.stack([np.array(Image.open(os.path.join(img,c,s,f"{b}.tif")),dtype=np.float32) for b in BANDS],-1)
    ly = lambda c: (np.array(Image.open(os.path.join(lbl,c,"cm",f"{c}-cm.tif")))==2).astype(np.int8)

    pool=[]
    for c in TRAIN:
        pool.append(ld(c,"imgs_1_rect").reshape(-1,13)); pool.append(ld(c,"imgs_2_rect").reshape(-1,13))
    pool=np.vstack(pool); P1=np.percentile(pool,1,0); P99=np.percentile(pool,99,0); del pool
    norm=lambda x:np.clip((x-P1)/(P99-P1),0,1)

    WINS=[1,3,5,7]
    raw_m={w:[] for w in WINS}; cor_m={w:[] for w in WINS}
    hardX=[]; hardY=[]; hardG=[]
    Y=[]; G=[]
    def ndi(N,a,b): return (N[...,BI[a]]-N[...,BI[b]])/(N[...,BI[a]]+N[...,BI[b]]+1e-6)
    for cid,c in enumerate(TRAIN):
        y=ly(c); H,W=y.shape
        t1=ld(c,"imgs_1_rect")[:H,:W]; t2=ld(c,"imgs_2_rect")[:H,:W]
        n1=norm(t1); n2=norm(t2); dpix=n2-n1
        adb=np.abs(dpix)                                        # raw |dB|
        med=np.median(dpix.reshape(-1,13),0)
        adbc=np.abs(dpix-med)                                   # median-corrected |dB|
        valid=~(((np.concatenate([t1,t2],-1))==0).any(-1))
        wm_raw={w:uniform_filter(adb ,size=(w,w,1),mode="reflect") for w in WINS}
        wm_cor={w:uniform_filter(adbc,size=(w,w,1),mode="reflect") for w in WINS}
        ch=valid&(y==1); nch=valid&(y==0)
        ci=np.argwhere(ch); ni=np.argwhere(nch); k=len(ci)
        sel=ni[RNG.choice(len(ni),min(len(ni),max(k,20000)),replace=False)]
        idx=np.vstack([ci,sel]); r,cc=idx[:,0],idx[:,1]
        for w in WINS: raw_m[w].append(wm_raw[w][r,cc]); cor_m[w].append(wm_cor[w][r,cc])
        Y.append(np.concatenate([np.ones(k),np.zeros(len(sel))])); G.append(np.full(len(idx),cid))
        # hard negatives with GROUP ids retained
        mag=adbc.mean(-1); thr=np.percentile(mag[valid],80); hi=valid&(mag>=thr)
        hci=np.argwhere(hi&(y==1)); hni=np.argwhere(hi&(y==0)); m=min(len(hci),len(hni))
        if m>50:
            hci=hci[RNG.choice(len(hci),m,replace=False)]; hni=hni[RNG.choice(len(hni),m,replace=False)]
            hidx=np.vstack([hci,hni]); hr,hcc=hidx[:,0],hidx[:,1]
            ctx=np.stack([ndi(n1,"B08","B04"),ndi(n2,"B08","B04"),
                          ndi(n1,"B11","B08"),ndi(n2,"B11","B08")],-1)
            hardX.append(np.hstack([n1[hr,hcc],n2[hr,hcc],ctx[hr,hcc]]))
            hardY.append(np.concatenate([np.ones(len(hci)),np.zeros(len(hni))]))
            hardG.append(np.full(len(hidx),cid))

    Y=np.concatenate(Y); G=np.concatenate(G); gkf=GroupKFold(5)
    auc=lambda F,y,g,k=5: cross_val_score(LogisticRegression(max_iter=500),F,y,cv=GroupKFold(k),groups=g,scoring="roc_auc").mean()

    md=["\n## Part 3 -- median-corrected spatial sweep + grouped hard negatives\n",
        "\n### Spatial sweep: raw vs median-corrected |dB| (mean-pool, grouped CV)\n",
        "| window | raw |dB| | median-corrected |dB| |\n|---|---|---|\n"]
    raw_auc=[]; cor_auc=[]
    for w in WINS:
        ar=auc(np.vstack(raw_m[w]),Y,G); ac=auc(np.vstack(cor_m[w]),Y,G)
        raw_auc.append(ar); cor_auc.append(ac); md.append(f"| {w}x{w} | {ar:.3f} | {ac:.3f} |\n")

    HX=np.vstack(hardX); HY=np.concatenate(hardY); HG=np.concatenate(hardG)
    ngrp=len(np.unique(HG))
    a_rand=cross_val_score(LogisticRegression(max_iter=500),HX,HY,cv=5,scoring="roc_auc").mean()
    a_grp =auc(HX,HY,HG,min(5,ngrp))
    md+=["\n### Hard negatives (urban vs natural, changed-a-lot): random vs city-grouped CV\n",
         f"- cities contributing hard-neg set: {ngrp}\n",
         f"- random 5-fold AUC     : {a_rand:.3f}\n",
         f"- city-grouped CV AUC   : {a_grp:.3f}  (leave-region-out; honest estimate)\n"]

    fig,ax=plt.subplots(figsize=(6,3.6))
    ax.plot(WINS,raw_auc,"o--",label="raw |dB|")
    ax.plot(WINS,cor_auc,"s-",label="median-corrected |dB|")
    ax.set_xlabel("window size"); ax.set_ylabel("grouped-CV AUC")
    ax.set_title("Median correction + spatial context (do they stack?)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{args.out}/part3_corrected_sweep.png",dpi=130); plt.close(fig)

    with open(f"{args.out}/RESULTS.md","a") as f: f.writelines(md)
    print("raw   :",["%.3f"%a for a in raw_auc])
    print("corr  :",["%.3f"%a for a in cor_auc])
    print("hardneg random %.3f | grouped %.3f (%d cities)"%(a_rand,a_grp,ngrp))

if __name__=="__main__":
    main()
