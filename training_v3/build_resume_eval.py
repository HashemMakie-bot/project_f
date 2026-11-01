import json
from pathlib import Path
n=json.loads(Path('Project3_Training_T4.ipynb').read_text())
def code(s): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.strip().splitlines(keepends=True)}
def markdown(s): return {'cell_type':'markdown','metadata':{},'source':s.strip().splitlines(keepends=True)}
s=lambda i:''.join(n['cells'][i]['source'])
cells=[markdown('''# متابعة Project3: تثبيت العتبات وتقييم النموذج

انتهى التدريب السابق بالتوقف المبكر. هذا الدفتر يحمّل أفضل الأوزان المحفوظة ويختار العتبات من Validation، دون إعادة التدريب. افتحه في Colab واختر T4 ثم Run all. تقييم Test معطّل افتراضيًا؛ غيّر RUN_TEST إلى True بعد مراجعة نتائج Validation. سبق استخدام مجموعة Test في المشروع، لذا فهي ليست تحققًا خارجيًا مستقلًا.'''),n['cells'][1],code(s(2).split('# فحص جميع صور التدريب والتحقق فقط')[0]+'''\n
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
audit=pd.read_csv(OUT/'data_audit.csv',dtype=str)
assert audit.nodule_id.is_unique
expected=set(pd.concat([train,val]).physical_nodule_id.astype(str))
assert set(audit.nodule_id)==expected, 'Audit does not match train/validation manifest'
# Verify the 134 files actually used for threshold selection against the completed run.
lookup=audit.set_index('nodule_id').sha256
for row in val.itertuples():
 if sha(row.path)!=lookup[str(row.physical_nodule_id)]: raise ValueError(f'Validation data changed: {row.path}')
print('Validation data verified:',len(val),'test files read: 0')'''),n['cells'][3],code('''
seg_ckpt=torch.load(OUT/'seg_best.pt',map_location=DEVICE,weights_only=False)
clf_ckpt=torch.load(OUT/'clf_best.pt',map_location=DEVICE,weights_only=False)
assert seg_ckpt['signature']['task']=='seg' and clf_ckpt['signature']['task']=='clf'
assert seg_ckpt['signature']['data']==clf_ckpt['signature']['data'], 'Data fingerprints differ'
assert clf_ckpt['signature']['seg_sha']==sha(OUT/'seg_best.pt'), 'Classifier was trained with a different segmenter'
fingerprint=seg_ckpt['signature']['data']
segmenter=UNet().to(DEVICE); segmenter.load_state_dict(seg_ckpt['model']); segmenter.eval()
classifier=Classifier().to(DEVICE); classifier.load_state_dict(clf_ckpt['model']); classifier.eval()
print('Best segmentation epoch:',seg_ckpt['epoch'],'| best classification epoch:',clf_ckpt['epoch'])
'''),n['cells'][6],markdown('''## تقييم Test اختياري

راجع ملف `frozen_validation.json` أولًا. إذا قررت التقييم، عدّل RUN_TEST=True في خلية الإعدادات وأعد التشغيل. عتبات Test لا تُضبط على الاختبار. نتائج الرقع لا تمثل أداء كشف العقيدات في صور DICOM الكاملة.'''),n['cells'][7]]
for i,c in enumerate(cells):
 c['id']=f'resume-{i:03d}'
 if c['cell_type']=='code':compile(''.join(c['source']),f'cell-{i}','exec')
n['cells']=cells;n['metadata']['colab']['name']='Project3_Resume_Validation.ipynb'
Path('Project3_Resume_Validation.ipynb').write_text(json.dumps(n,ensure_ascii=False,indent=1))
print('Generated and compiled',len(cells),'cells')
