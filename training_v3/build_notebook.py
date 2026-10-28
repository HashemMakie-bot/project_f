import json
from pathlib import Path
cells=[]
def add(kind,s):
 c={'cell_type':kind,'metadata':{},'source':s.strip().splitlines(keepends=True)}
 if kind=='code': c.update(execution_count=None,outputs=[])
 cells.append(c)
add('markdown', '''# Project3: تدريب التجزئة والتصنيف على Colab T4

اختر T4 GPU في Colab ثم Run all واربط Google Drive. يستخدم الدفتر رقع المشروع الموجودة في `project3/processed/patches/training_64_with_masks` وتقسيم المرضى الأصلي. يحفظ المخرجات في مجلد تدريب جديد، ويستأنف من آخر epoch مكتملة. غيّر RUN_NAME عند تغيير الإعدادات. صور Test لا تُفتح إلا عند تفعيل RUN_TEST بعد تجميد النتائج. النموذج تجريبي لتصنيف درجات الأطباء، ولا يدرب كاشف الأشعة الكاملة أو يثبت تحسن الدقة.''')
add('code', '''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
ROOT=Path('/content/drive/MyDrive/project3')
assert ROOT.exists(), f'المجلد غير موجود: {ROOT}'
RUN_NAME='resunet_context_t4_v2'
RUN_TEST=False
BATCH=2
ACCUM=4
SEG_EPOCHS=80
CLF_EPOCHS=60
PATIENCE=12
SEED=42
OUT=ROOT/'training_runs'/RUN_NAME
OUT.mkdir(parents=True,exist_ok=True)''')
add('code', '''import os, json, random, math, hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
assert torch.cuda.is_available(), 'اختر GPU في إعدادات Colab'
DEVICE=torch.device('cuda')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
manifest=ROOT/'training_patch_mask_manifest_patient_aware_corrected.csv'
if not manifest.exists(): manifest=ROOT/'processed/metadata/training_patch_mask_manifest_patient_aware_corrected.csv'
df=pd.read_csv(manifest)
required={'physical_nodule_id','patient_id','split','label_code','segmentation_eligible'}
assert required.issubset(df.columns) and not df[list(required)].isna().any().any()
assert df.physical_nodule_id.is_unique and (df.groupby('patient_id').split.nunique()==1).all()
assert set(df.split)=={'train','validation','test'} and df.label_code.isin([0,1]).all()
assert df.segmentation_eligible.astype(str).str.lower().isin(['true','1']).all()
assert df.split.value_counts().to_dict()=={'train':648,'validation':134,'test':131}
df['path']=df.physical_nodule_id.map(lambda s: ROOT/'processed/patches/training_64_with_masks'/f'{s}.npz')
train=df[df.split=='train'].reset_index(drop=True); val=df[df.split=='validation'].reset_index(drop=True); test=df[df.split=='test'].reset_index(drop=True)
print(df.groupby('split').agg(nodules=('patient_id','size'),patients=('patient_id','nunique'),positive=('label_code','sum')))

def read_patch(path):
 with np.load(path,allow_pickle=False) as z:
  x=z['patch'].astype(np.float32); m=z['mask'].astype(np.float32)
 if x.shape!=(64,64,64) or m.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(m).all(): raise ValueError(f'Invalid data: {path}')
 if x.min() < -1e-6 or x.max()>1.000001 or not np.isin(m,[0,1]).all() or not m.any(): raise ValueError(f'Invalid range/mask: {path}')
 return x,m

# فحص جميع صور التدريب والتحقق فقط؛ التوقف عند أول ملف تالف.
audit=[]
for row in pd.concat([train,val]).itertuples():
 x,m=read_patch(row.path)
 if hasattr(row,'mask_voxels') and int(m.sum())!=int(row.mask_voxels): raise ValueError(f'Mask mismatch: {row.path}')
 digest=hashlib.sha256(Path(row.path).read_bytes()).hexdigest()
 audit.append((row.physical_nodule_id,row.split,digest))
 if len(audit)%100==0: print('Audited:',len(audit),flush=True)
fingerprint=hashlib.sha256(json.dumps(audit).encode()+manifest.read_bytes()).hexdigest()
pd.DataFrame(audit,columns=['nodule_id','split','sha256']).to_csv(OUT/'data_audit.csv',index=False)
print('Data audit complete:',len(audit),'test images read: 0')''')
add('code', '''class Patches(Dataset):
 def __init__(self,frame,training=False): self.frame=frame; self.training=training; self.epoch=0
 def __len__(self): return len(self.frame)
 def __getitem__(self,i):
  row=self.frame.iloc[i]; x,m=read_patch(row.path)
  if self.training:
   rng=np.random.default_rng(SEED+self.epoch*1000003+i)
   for axis in range(3):
    if rng.random()<.5: x=np.flip(x,axis); m=np.flip(m,axis)
   axes=[(0,1),(0,2),(1,2)][int(rng.integers(3))]; k=int(rng.integers(4))
   x=np.rot90(x,k,axes); m=np.rot90(m,k,axes)
   if rng.random()<.3: x=np.clip(x+rng.normal(0,.015,x.shape),0,1)
  return torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32))[None],torch.from_numpy(np.ascontiguousarray(m,dtype=np.float32))[None],torch.tensor(float(row.label_code)),str(row.physical_nodule_id),str(row.patient_id)
def loader(frame,training=False,epoch=0):
 ds=Patches(frame,training); ds.epoch=epoch
 return DataLoader(ds,batch_size=BATCH,shuffle=training,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(SEED+epoch))
class Block(nn.Module):
 def __init__(self,a,b,stride=1):
  super().__init__(); self.c1=nn.Conv3d(a,b,3,stride,padding=1,bias=False); self.n1=nn.GroupNorm(math.gcd(8,b),b); self.c2=nn.Conv3d(b,b,3,padding=1,bias=False); self.n2=nn.GroupNorm(math.gcd(8,b),b); self.skip=nn.Identity() if a==b and stride==1 else nn.Conv3d(a,b,1,stride,bias=False)
 def forward(self,x): return F.leaky_relu(self.n2(self.c2(F.leaky_relu(self.n1(self.c1(x)),.1)))+self.skip(x),.1)
class UNet(nn.Module):
 def __init__(self,b=12):
  super().__init__(); c=[b,b*2,b*4,b*8]; self.enc=nn.ModuleList([Block(1,c[0])]+[Block(c[i-1],c[i],2) for i in range(1,4)]); self.dec=nn.ModuleList([Block(c[i+1]+c[i],c[i]) for i in [2,1,0]]); self.head=nn.Conv3d(b,1,1)
 def forward(self,x):
  features=[]
  for layer in self.enc: x=layer(x); features.append(x)
  for layer,skip in zip(self.dec,reversed(features[:-1])): x=layer(torch.cat([F.interpolate(x,size=skip.shape[2:],mode='trilinear',align_corners=False),skip],1))
  return self.head(x)
class Classifier(nn.Module):
 def __init__(self,b=12):
  super().__init__(); c=[b,b*2,b*4,b*8]; self.enc=nn.Sequential(Block(3,c[0],2),Block(c[0],c[1],2),Block(c[1],c[2],2),Block(c[2],c[3],2)); self.head=nn.Sequential(nn.Linear(c[3]*3,64),nn.LeakyReLU(.1),nn.Dropout(.35),nn.Linear(64,1))
 def forward(self,x,p):
  p=p.detach(); f=self.enc(torch.cat([x,x*p,p],1)); w=F.interpolate(p,size=f.shape[2:],mode='trilinear',align_corners=False); roi=(f*w).sum((2,3,4))/w.sum((2,3,4)).clamp_min(1e-6)
  return self.head(torch.cat([f.mean((2,3,4)),f.amax((2,3,4)),roi],1)).squeeze(1)
def dice(p,m): return (2*(p*m).sum((1,2,3,4))+1e-6)/(p.sum((1,2,3,4))+m.sum((1,2,3,4))+1e-6)
def segloss(logits,m): return .7*(1-dice(logits.float().sigmoid(),m).mean())+.3*F.binary_cross_entropy_with_logits(logits.float(),m)
print('Models ready:',torch.cuda.get_device_name())''')
add('code', '''def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(obj,path):
 tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); os.replace(tmp,path)
def epoch(model,frame,task,segmenter=None,optimizer=None,scaler=None,number=0):
 training=optimizer is not None; model.train(training)
 if segmenter is not None: segmenter.eval()
 batches=loader(frame,training,number); losses=[]; scores=[]; labels=[]; dices=[]; pending=0
 if training: optimizer.zero_grad(set_to_none=True)
 weight=float((train.label_code==0).sum()/(train.label_code==1).sum())
 for step,(x,m,y,_,_) in enumerate(batches):
  x=x.to(DEVICE); m=m.to(DEVICE); y=y.to(DEVICE)
  with torch.set_grad_enabled(training):
   with torch.autocast('cuda',dtype=torch.float16):
    if task=='seg': logits=model(x); loss=segloss(logits,m)
    else:
     with torch.no_grad(): p=segmenter(x).float().sigmoid()
     logits=model(x,p); loss=F.binary_cross_entropy_with_logits(logits.float(),y,pos_weight=torch.tensor(weight,device=DEVICE))
   if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
   if training:
    scaler.scale(loss*len(x)).backward(); pending+=len(x)
    if (step+1)%ACCUM==0 or step+1==len(batches):
     scaler.unscale_(optimizer)
     for param in model.parameters():
      if param.grad is not None: param.grad.div_(pending)
     norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.float()) for p in model.parameters() if p.grad is not None]))
     if torch.isfinite(norm): torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
     scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); pending=0
  losses.extend([float(loss.detach())]*len(x))
  if task=='seg': dices.extend(dice(logits.detach().float().sigmoid(),m).cpu().tolist())
  else: scores.extend(logits.detach().float().sigmoid().cpu().tolist()); labels.extend(y.cpu().tolist())
 result={'loss':float(np.mean(losses))}
 if task=='seg': result['score']=float(np.mean(dices))
 else:
  result['score']=float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None
 return result

def fit(model,task,epochs,segmenter=None):
 model=model.to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=1e-5); scaler=torch.amp.GradScaler('cuda')
 signature={'data':fingerprint,'task':task,'batch':BATCH,'accum':ACCUM,'epochs':epochs,'seed':SEED,'seg_sha':sha(OUT/'seg_best.pt') if task=='clf' else None}
 last=OUT/f'{task}_last.pt'; best=OUT/f'{task}_best.pt'; start=1; bestscore=-1.; stale=0; history=[]; stopped=False
 if last.exists():
  c=torch.load(last,map_location=DEVICE,weights_only=False)
  if c['signature']!=signature: raise ValueError('Run configuration changed; choose a new RUN_NAME')
  model.load_state_dict(c['model']); opt.load_state_dict(c['optimizer']); scheduler.load_state_dict(c['scheduler']); scaler.load_state_dict(c['scaler']); start=c['epoch']+1; bestscore=c['bestscore']; stale=c['stale']; history=c['history']; stopped=c['stopped']; torch.set_rng_state(c['rng'].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in c['cuda_rng']])
 for n in range(start,epochs+1):
  if stopped: break
  tr=epoch(model,train,task,segmenter,opt,scaler,n)
  with torch.no_grad(): v=epoch(model,val,task,segmenter)
  if v['score'] is None: raise ValueError('Validation requires both classes')
  row={'epoch':n,'train_loss':tr['loss'],'val_loss':v['loss'],'val_score':v['score']}; history.append(row)
  improved=v['score']>bestscore+1e-6
  if improved: bestscore=v['score']; stale=0
  else: stale+=1
  stopped=n>=15 and stale>=PATIENCE; scheduler.step()
  c={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':n,'signature':signature,'bestscore':bestscore,'stale':stale,'history':history,'stopped':stopped,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
  if improved: save(c,best)
  save(c,last); pd.DataFrame(history).to_csv(OUT/f'{task}_history.csv',index=False)
  print(task,row,'stale',stale,flush=True)
 model.load_state_dict(torch.load(best,map_location=DEVICE,weights_only=False)['model']); return model.eval()

# اختبار خطوة فعلية بأوزان مؤقتة قبل التدريب الطويل.
x,m,y,_,_=next(iter(loader(train))); x=x.to(DEVICE); m=m.to(DEVICE); y=y.to(DEVICE)
s=UNet().to(DEVICE); c=Classifier().to(DEVICE); opt=torch.optim.AdamW(list(s.parameters())+list(c.parameters()))
opt.zero_grad(); logits=s(x); loss=segloss(logits,m)+F.binary_cross_entropy_with_logits(c(x,logits.detach().sigmoid()),y); loss.backward(); opt.step()
assert torch.isfinite(loss); print('Training step passed:',float(loss.detach()),'GPU GiB:',torch.cuda.max_memory_allocated()/1024**3)
del s,c,opt,x,m,y,logits,loss
torch.cuda.empty_cache()''')
add('code', '''torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
segmenter=fit(UNet(),'seg',SEG_EPOCHS)
for p in segmenter.parameters(): p.requires_grad_(False)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
classifier=fit(Classifier(),'clf',CLF_EPOCHS,segmenter)''')
add('code', '''@torch.no_grad()
def predictions(frame,thresholds):
 segmenter.eval(); classifier.eval(); records=[]; totals=np.zeros(len(thresholds))
 for x,m,y,ids,patients in loader(frame):
  x=x.to(DEVICE); m=m.to(DEVICE)
  with torch.autocast('cuda',dtype=torch.float16): p=segmenter(x).float().sigmoid(); score=classifier(x,p).float().sigmoid()
  all_dice=[]
  for j,t in enumerate(thresholds):
   d=dice((p>=t).float(),m).cpu().numpy(); totals[j]+=d.sum(); all_dice.append(d)
  for i in range(len(x)): records.append({'nodule_id':ids[i],'patient_id':patients[i],'label':int(y[i]),'score':float(score[i]),'dice':[float(d[i]) for d in all_dice]})
 return records,totals/len(frame)
seg_thresholds=np.linspace(.1,.9,33); rows,mean_dice=predictions(val,seg_thresholds); bestidx=int(np.argmax(mean_dice)); seg_threshold=float(seg_thresholds[bestidx])
labels=np.array([r['label'] for r in rows]); scores=np.array([r['score'] for r in rows]); choices=[]
for t in np.unique(np.r_[0.,scores,1.]):
 tn,fp,fn,tp=confusion_matrix(labels,scores>=t,labels=[0,1]).ravel(); sens=tp/(tp+fn); spec=tn/(tn+fp)
 if sens>=.9: choices.append((spec,float(t),sens))
spec,clf_threshold,sens=max(choices)
frozen={'seg_threshold':seg_threshold,'clf_threshold':clf_threshold,'validation_dice':float(mean_dice[bestidx]),'validation_auroc':float(roc_auc_score(labels,scores)),'validation_ap':float(average_precision_score(labels,scores)),'validation_sensitivity':float(sens),'validation_specificity':float(spec),'seg_sha256':sha(OUT/'seg_best.pt'),'clf_sha256':sha(OUT/'clf_best.pt'),'data_fingerprint':fingerprint,'test_used':False}
(OUT/'frozen_validation.json').write_text(json.dumps(frozen,indent=2)); pd.DataFrame([{**r,'dice':r['dice'][bestidx]} for r in rows]).to_csv(OUT/'validation_predictions.csv',index=False)
print(json.dumps(frozen,indent=2))''')
add('code', '''if RUN_TEST:
 frozen=json.loads((OUT/'frozen_validation.json').read_text())
 assert sha(OUT/'seg_best.pt')==frozen['seg_sha256'] and sha(OUT/'clf_best.pt')==frozen['clf_sha256']
 rows,d=predictions(test,[frozen['seg_threshold']]); labels=np.array([r['label'] for r in rows]); scores=np.array([r['score'] for r in rows]); tn,fp,fn,tp=confusion_matrix(labels,scores>=frozen['clf_threshold'],labels=[0,1]).ravel()
 metrics={'auroc':float(roc_auc_score(labels,scores)),'average_precision':float(average_precision_score(labels,scores)),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),'mean_dice':float(d[0]),'tp':int(tp),'tn':int(tn),'fp':int(fp),'fn':int(fn)}
 pd.DataFrame(rows).to_csv(OUT/'test_predictions.csv',index=False); (OUT/'test_metrics.json').write_text(json.dumps(metrics,indent=2)); print(metrics)
else: print('Test evaluation skipped. Outputs:',OUT)''')
for i,c in enumerate(cells):
 c['id']=f'project3-{i:03d}'
 if c['cell_type']=='code': compile(''.join(c['source']),f'cell-{i}','exec')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'accelerator':'GPU','colab':{'name':'Project3_Training_T4.ipynb','provenance':[]}},'cells':cells}
Path('Project3_Training_T4.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print('Notebook generated,',len(cells),'cells')
