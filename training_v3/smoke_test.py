import json, pathlib, math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(2)
nb=json.loads(pathlib.Path('Project3_Training_T4.ipynb').read_text())
for c in nb['cells']:
 if c['cell_type']=='code': compile(''.join(c['source']),'notebook','exec')
ns=globals(); ns['SEED']=42; ns['DEVICE']=torch.device('cpu')
src=''.join(nb['cells'][3]['source']).replace("print('Models ready:',torch.cuda.get_device_name())",'')
exec(src,ns)
p=next(pathlib.Path('/workspace/attachments').rglob('PHYSICAL_NODULE_003832.npz'))
with np.load(p,allow_pickle=False) as z: image=z['patch'].astype('float32'); mask=z['mask'].astype('float32')
assert image.shape==mask.shape==(64,64,64) and np.isfinite(image).all() and set(np.unique(mask))<={0,1}
x=torch.tensor(image)[None,None]; m=torch.tensor(mask)[None,None]
seg=UNet(4); clf=Classifier(4); opt=torch.optim.AdamW(list(seg.parameters())+list(clf.parameters()))
pred=seg(x); assert pred.shape==x.shape
score=clf(x,pred.detach().sigmoid()); loss=segloss(pred,m)+F.binary_cross_entropy_with_logits(score,torch.ones(1)); opt.zero_grad(); loss.backward(); opt.step()
assert torch.isfinite(loss)
print('PASS: notebook cells compile, one real NPZ loaded, segmentation and classification forward/backward completed on CPU; loss',float(loss.detach()))
