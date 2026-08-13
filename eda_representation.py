"""
Representation check (pre-pipeline): on the SAME median-corrected 13-band base
|dB_corr|, compare All-13 vs Physical-4 {B04,B05,B12,B08} vs PCA-4 under the
identical 5-fold city-grouped CV. Decides the main QML input representation.

Leakage discipline: per-image median correction is unsupervised (per-image),
PCA is re-fit inside each fold (leakage-free pipeline).

Usage: python eda_representation.py --data_dir /path/to/OneraDataset
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold

BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
TRAIN = ["aguasclaras","bercy","bordeaux","nantes","paris","rennes","saclay_e",
         "abudhabi","cupertino","pisa","beihai","hongkong","beirut","mumbai"]
BI = {b:i for i,b in enumerate(BANDS)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True)
    args = ap.parse_args(); rng = np.random.RandomState(0)
    img = os.path.join(args.data_dir,"images","Onera Satellite Change Detection dataset - Images")
    lbl = os.path.join(args.data_dir,"train_labels","Onera Satellite Change Detection dataset - Train Labels")
    ld = lambda c,s: np.stack([np.array(Image.open(os.path.join(img,c,s,f"{b}.tif")),dtype=np.float32) for b in BANDS],-1)
    ly = lambda c: (np.array(Image.open(os.path.join(lbl,c,"cm",f"{c}-cm.tif")))==2).astype(np.int8)

    pool=[]
    for c in TRAIN:
        pool.append(ld(c,"imgs_1_rect").reshape(-1,13)); pool.append(ld(c,"imgs_2_rect").reshape(-1,13))
    pool=np.vstack(pool); P1=np.percentile(pool,1,0); P99=np.percentile(pool,99,0); del pool
    norm=lambda x:np.clip((x-P1)/(P99-P1),0,1)

    D_l=[];Y_l=[];G_l=[]
    for cid,c in enumerate(TRAIN):
        y=ly(c);H,W=y.shape
        t1=ld(c,"imgs_1_rect")[:H,:W]; t2=ld(c,"imgs_2_rect")[:H,:W]
        d=norm(t2)-norm(t1); valid=~(((np.concatenate([t1,t2],-1))==0).any(-1))
        D=np.abs(d-np.median(d[valid],0))                      # |dB_corr|_13, per-image median
        ch=valid&(y==1); nch=valid&(y==0)
        ci=np.argwhere(ch); ni=np.argwhere(nch); k=len(ci)
        sel=ni[rng.choice(len(ni),min(len(ni),max(k,20000)),replace=False)]
        idx=np.vstack([ci,sel]); r,cc=idx[:,0],idx[:,1]
        D_l.append(D[r,cc]); Y_l.append(np.concatenate([np.ones(k),np.zeros(len(sel))])); G_l.append(np.full(len(idx),cid))
    D=np.vstack(D_l); Y=np.concatenate(Y_l).astype(int); G=np.concatenate(G_l)

    phys=[BI[b] for b in ["B04","B05","B12","B08"]]
    cv=lambda est,F: cross_val_score(est,F,Y,cv=GroupKFold(5),groups=G,scoring="roc_auc")
    lr=LogisticRegression(max_iter=500)
    pca4=make_pipeline(StandardScaler(),PCA(4),LogisticRegression(max_iter=500))
    print("Median-corrected |dB|_13 base, 5-fold city-grouped CV (ROC-AUC):\n")
    for name,res in [("All-13",cv(lr,D)),
                     ("Physical-4 {B04,B05,B12,B08}",cv(lr,D[:,phys])),
                     ("PCA-4 (per-fold, leakage-free)",cv(pca4,D))]:
        print(f"  {name:34} {res.mean():.3f} +/- {res.std():.3f}   folds={np.round(res,3)}")

if __name__=="__main__":
    main()
