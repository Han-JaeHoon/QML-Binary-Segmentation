"""
OSCD EDA -- Part 2: domain-shift correction, hard negatives, spatial context.
Follow-up to eda.py. Decides (a) whether per-image median correction helps,
(b) whether land-cover state separates urban vs natural change within the
"changed-a-lot" group, and (c) how much spatial context (patch size) adds --
i.e. whether a purely spectral VQC suffices or a spatial model is warranted.

Usage: python eda_spatial.py --data_dir /path/to/OneraDataset
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
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, GroupKFold

    img = os.path.join(args.data_dir,"images","Onera Satellite Change Detection dataset - Images")
    lbl = os.path.join(args.data_dir,"train_labels","Onera Satellite Change Detection dataset - Train Labels")
    ld  = lambda c,s: np.stack([np.array(Image.open(os.path.join(img,c,s,f"{b}.tif")),dtype=np.float32) for b in BANDS],-1)
    ly  = lambda c: (np.array(Image.open(os.path.join(lbl,c,"cm",f"{c}-cm.tif")))==2).astype(np.int8)

    # frozen robust normalization params (train, T1+T2 pooled)
    pool=[]
    for c in TRAIN:
        pool.append(ld(c,"imgs_1_rect").reshape(-1,13)); pool.append(ld(c,"imgs_2_rect").reshape(-1,13))
    pool=np.vstack(pool); P1=np.percentile(pool,1,0); P99=np.percentile(pool,99,0); del pool
    norm=lambda x:np.clip((x-P1)/(P99-P1),0,1)

    WINS=[1,3,5,7]
    feat={"raw":[], "corr":[]}                 # |dB| at pixel: raw vs median-corrected
    win_mean={w:[] for w in WINS}; win_max={w:[] for w in WINS}
    hardX=[]; hardY=[]; state_all=[]
    Y=[]; G=[]
    for cid,c in enumerate(TRAIN):
        t1=ld(c,"imgs_1_rect"); y=ly(c); H,W=y.shape
        t1=t1[:H,:W]; t2=ld(c,"imgs_2_rect")[:H,:W]
        n1=norm(t1); n2=norm(t2)
        adb=np.abs(n2-n1)                                   # H,W,13  |dB|
        med=np.median((n2-n1).reshape(-1,13),0)             # per-image baseline (dominated by no-change)
        adb_corr=np.abs((n2-n1)-med)
        valid=~(((np.concatenate([t1,t2],-1))==0).any(-1))
        # spatial aggregation of |dB| (before sampling), per band
        wm={w:uniform_filter(adb,size=(w,w,1),mode="reflect") for w in WINS}
        wx={w:maximum_filter(adb,size=(w,w,1),mode="reflect") for w in WINS}
        # sample: all change + equal-ish no-change
        ch=valid&(y==1); nch=valid&(y==0)
        ci=np.argwhere(ch); ni=np.argwhere(nch); k=len(ci)
        sel=ni[RNG.choice(len(ni),min(len(ni),max(k,20000)),replace=False)]
        idx=np.vstack([ci,sel]); r,cc=idx[:,0],idx[:,1]
        yy=np.concatenate([np.ones(k),np.zeros(len(sel))])
        feat["raw"].append(adb[r,cc]); feat["corr"].append(adb_corr[r,cc])
        for w in WINS: win_mean[w].append(wm[w][r,cc]); win_max[w].append(wx[w][r,cc])
        state_all.append(np.hstack([n1[r,cc],n2[r,cc]]))     # absolute state 26-dim
        Y.append(yy); G.append(np.full(len(idx),cid))
        # hard negatives: pixels that changed a lot (top-20% pixel-mean |dB|) among VALID
        mag=adb.mean(-1); thr=np.percentile(mag[valid],80)
        hi=valid&(mag>=thr)
        hci=np.argwhere(hi&(y==1)); hni=np.argwhere(hi&(y==0))
        m=min(len(hci),len(hni))
        if m>50:
            hci=hci[RNG.choice(len(hci),m,replace=False)]; hni=hni[RNG.choice(len(hni),m,replace=False)]
            hidx=np.vstack([hci,hni]); hr,hcc=hidx[:,0],hidx[:,1]
            # features to tell urban(1) from natural(0) *within* changed-a-lot:
            def ndi(N,a,b): return (N[...,BI[a]]-N[...,BI[b]])/(N[...,BI[a]]+N[...,BI[b]]+1e-6)
            ctx=np.stack([ndi(n1,"B08","B04"),ndi(n2,"B08","B04"),   # NDVI T1,T2
                          ndi(n1,"B11","B08"),ndi(n2,"B11","B08")],-1) # NDBI T1,T2
            hardX.append(np.hstack([n1[hr,hcc],n2[hr,hcc],ctx[hr,hcc]]))
            hardY.append(np.concatenate([np.ones(len(hci)),np.zeros(len(hni))]))

    Y=np.concatenate(Y); G=np.concatenate(G)
    gkf=GroupKFold(5); lr=lambda:LogisticRegression(max_iter=500)
    auc=lambda F,y,g: cross_val_score(lr(),F,y,cv=gkf,groups=g,scoring="roc_auc").mean()

    md=["\n## Part 2 -- Domain shift, hard negatives, spatial context\n"]

    # (1) median correction
    a_raw=auc(np.vstack(feat["raw"]),Y,G); a_corr=auc(np.vstack(feat["corr"]),Y,G)
    md+= [f"\n### Per-image median correction (removes city/season domain shift)\n",
          f"- |dB| raw            : AUC {a_raw:.3f}\n",
          f"- |dB| median-corrected: AUC {a_corr:.3f}\n"]

    # (2) spatial sweep
    md+=["\n### Spatial-context sweep (window aggregation of |dB|, leakage-free)\n",
         "| window | mean-pool AUC | max-pool AUC |\n|---|---|---|\n"]
    ms=[]; xs=[]
    for w in WINS:
        am=auc(np.vstack(win_mean[w]),Y,G); ax=auc(np.vstack(win_max[w]),Y,G)
        ms.append(am); xs.append(ax); md.append(f"| {w}x{w} | {am:.3f} | {ax:.3f} |\n")
    fig,axp=plt.subplots(figsize=(6,3.6))
    axp.plot(WINS,ms,"o-",label="mean-pool |dB|"); axp.plot(WINS,xs,"s-",label="max-pool |dB|")
    axp.set_xlabel("window size"); axp.set_ylabel("grouped-CV AUC")
    axp.set_title("Spatial context vs pixel-only (1x1)"); axp.legend(); axp.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{args.out}/part2_patch_sweep.png",dpi=130); plt.close(fig)

    # (3) hard negatives: urban vs natural WITHIN changed-a-lot
    HX=np.vstack(hardX); HY=np.concatenate(hardY)
    md+=["\n### Hard negatives: urban vs *natural* change within the changed-a-lot group\n",
         "_(only pixels in the top-20% |dB| magnitude; balanced urban vs natural)_\n"]
    # grouped CV needs groups; approximate with random 5-fold here (per-city already balanced)
    from sklearn.model_selection import cross_val_score as cvs
    a_state=cvs(lr(),HX,HY,cv=5,scoring="roc_auc").mean()
    md.append(f"- separability using [B_T1,B_T2,NDVI_T1/T2,NDBI_T1/T2]: AUC {a_state:.3f} "
              f"(0.5 = magnitude alone can't tell them apart)\n")

    with open(f"{args.out}/RESULTS.md","a") as f: f.writelines(md)
    print("median: raw %.3f corr %.3f | sweep mean %s | hardneg %.3f"%(
          a_raw,a_corr,["%.3f"%m for m in ms],a_state))

if __name__=="__main__":
    main()
