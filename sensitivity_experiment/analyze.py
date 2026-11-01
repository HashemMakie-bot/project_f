"""Select a higher-recall threshold using validation predictions only."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

def metrics(frame, threshold):
    y=frame.label.to_numpy(dtype=int); p=frame.score.to_numpy(dtype=float); positive=p>=threshold
    tp=int(((y==1)&positive).sum()); fn=int(((y==1)&~positive).sum())
    fp=int(((y==0)&positive).sum()); tn=int(((y==0)&~positive).sum())
    return dict(tp=tp,fn=fn,fp=fp,tn=tn,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp))

def select(frame,target):
    if not 0<target<=1 or set(frame.label.unique())!={0,1}: raise ValueError('Both labels and a valid sensitivity target are required')
    if not np.isfinite(frame.score).all() or not frame.score.between(0,1).all(): raise ValueError('Invalid scores')
    options=[]
    for threshold in np.unique(np.r_[0.,frame.score.to_numpy(),1.]):
        m=metrics(frame,float(threshold))
        if m['sensitivity']>=target: options.append((m['specificity'],float(threshold),m))
    _,threshold,m=max(options,key=lambda row:(row[0],row[1]))
    return threshold,m

def bootstrap_fixed(frame,threshold,repeats=2000,seed=42):
    groups=[g for _,g in frame.groupby('patient_id')]; rng=np.random.default_rng(seed); values=[]
    for _ in range(repeats):
        sample=pd.concat([groups[i] for i in rng.integers(0,len(groups),len(groups))],ignore_index=True)
        if sample.label.nunique()!=2: continue
        m=metrics(sample,threshold); values.append([m['sensitivity'],m['specificity']])
    a=np.asarray(values)
    return {'valid':len(values),'requested':repeats,'sensitivity_95_percentile':np.quantile(a[:,0],[.025,.975]).tolist(),
            'specificity_95_percentile':np.quantile(a[:,1],[.025,.975]).tolist()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('validation_csv'); ap.add_argument('--out',required=True); ap.add_argument('--target',type=float,default=.95)
    args=ap.parse_args(); path=Path(args.validation_csv); frame=pd.read_csv(path)
    if not {'nodule_id','patient_id','label','score'}.issubset(frame.columns) or frame[['nodule_id','patient_id','label','score']].isna().any().any(): raise ValueError('Missing identities or scores')
    if frame.nodule_id.duplicated().any(): raise ValueError('Duplicate nodule IDs')
    threshold,m=select(frame,args.target); fixed=bootstrap_fixed(frame,threshold)
    output={'data_role':'validation only; test not read','selection':'maximum specificity at target sensitivity',
            'target_sensitivity':args.target,'threshold':threshold,'validation':m,'patients':int(frame.patient_id.nunique()),
            'nodules':len(frame),'validation_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'bootstrap_conditional_on_selected_threshold':fixed,
            'limitations':'Selected and assessed on the same validation data; intervals are conditional and optimistic; external evaluation is required. Scores are not calibrated cancer probabilities.'}
    Path(args.out).write_text(json.dumps(output,indent=2),encoding='utf-8'); print(json.dumps(output,indent=2))

if __name__=='__main__': main()
