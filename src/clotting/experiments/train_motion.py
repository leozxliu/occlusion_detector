"""Fixed causal frame-difference baseline; no tuning on held-out videos."""
from pathlib import Path
import json,cv2,numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score,roc_auc_score
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=Path('data/six_controls'))
parser.add_argument('--out',type=Path,default=Path('runs/six_channel_comparison'))
args=parser.parse_args()
OUT=args.out/'motion';OUT.mkdir(parents=True,exist_ok=True)
BASE=args.data;labels=json.loads((BASE/'frame_labels.json').read_text());d=json.loads((BASE/'dataset.json').read_text());features={};meta={}
protocol={'split':'same four leave-one-video-out folds as ResNet','features':'mean and 90th-percentile absolute grayscale differences at lags 1 and 3 frames; each raw feature plus past-only 15-frame mean and standard deviation (12 features)','preprocessing':'same normalized 224x224 channel crops; interior pixels [22:202,22:202]; grayscale /255; no intensity/static appearance features','classifier':'StandardScaler + class-balanced LogisticRegression C=1 max_iter=2000 seed=0','training':'every tenth frame; exclude +/-5 experimental seconds around OT; same labels as ResNet','test':'all eligible post-entry frames','threshold':0.5,'sustain_frames':5,'initial_history':'zero difference where history unavailable; no future frames','clock':'video time zero, 10x','tuning':'none'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
for v in d['videos']:
 case=v['case']
 for a in v['channel_assignments']:
  side=a['roi_id'];rr=labels[case][side];history=[];raw=[];x=[]
  for r in rr:
   img=cv2.imread(str(BASE/'frames'/r['path']),0)[22:202,22:202].astype(np.float32)/255;vec=[]
   for lag in [1,3]:
    delta=np.abs(img-history[-lag]) if len(history)>=lag else np.zeros_like(img)
    vec.extend([float(delta.mean()),float(np.quantile(delta,.9))])
   raw.append(vec);w=np.asarray(raw[-15:]);x.append(vec+w.mean(0).tolist()+w.std(0).tolist());history.append(img)
   if len(history)>3:history.pop(0)
  features[(case,side)]=np.asarray(x);meta[(case,side)]=a
  np.savez_compressed(OUT/f'features_{case}_{side}.npz',features=x,frame_indices=[r['frame_idx'] for r in rr]);print('features',case,side,len(rr),flush=True)
def metrics(y,p):
 pred=np.array(p)>.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return dict(n_frames=len(y),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),precision=float(precision_score(y,pred,zero_division=0)),recall_sensitivity=float(recall_score(y,pred,zero_division=0)),specificity=float(tn/(tn+fp)),f1=float(f1_score(y,pred,zero_division=0)),accuracy=float(accuracy_score(y,pred)),auroc=float(roc_auc_score(y,p)) if len(set(y))>1 else None)
def onset(ts,p):
 for i in range(len(p)-4):
  if np.all(p[i:i+5]>.5):return ts[i]
 return None
folds=[];ay=[];ap=[]
for test in labels:
 xs=[];ys=[]
 for (case,side),x in features.items():
  if case==test:continue
  rr=labels[case][side];ot=meta[(case,side)]['ot_seconds'];mask=np.array([r['frame_idx']%10==0 and abs(r['time_s']*10-ot)>5 for r in rr]);xs.append(x[mask]);ys.extend(r['label'] for r,m in zip(rr,mask) if m)
 model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=0));model.fit(np.concatenate(xs),ys);joblib.dump(model,OUT/f'model_test_{test}.joblib');fold={'test_case':test,'training_cases':[c for c in labels if c!=test],'training_frames':len(ys),'channels':[]};fy=[];fp=[]
 for side,rr in labels[test].items():
  pr=model.predict_proba(features[(test,side)])[:,1];ts=[r['time_s'] for r in rr];y=[r['label'] for r in rr];a=meta[(test,side)];on=onset(ts,pr);ot=a['ot_seconds']
  c={'case':test,'side':side,'channel':a['workbook_channel'],'ot_s':ot,'observable':True,'metrics':metrics(y,pr),'predicted_onset_video_s':on,'predicted_onset_experiment_s':on*10 if on is not None else None,'error_experiment_s':on*10-ot if on is not None else None};fold['channels'].append(c);fy+=y;fp+=pr.tolist();(OUT/f'signal_{test}_{side}.json').write_text(json.dumps({'video_s':ts,'probability':pr.tolist(),'label':y}))
 fold['metrics']=metrics(fy,fp);folds.append(fold);ay+=fy;ap+=fp;print('RESULT',test,fold['metrics'],flush=True)
errors=[c['error_experiment_s'] for f in folds for c in f['channels'] if c['error_experiment_s'] is not None]
s={'protocol':protocol,'folds':folds,'pooled_frame_metrics':metrics(ay,ap),'onset_mae_experimental_s_detected_only':float(np.mean(np.abs(errors))) if errors else None,'observable_events':6,'observable_events_with_trigger':len(errors),'missed_observable_events':6-len(errors)};(OUT/'summary.json').write_text(json.dumps(s,indent=2));print('DONE',s['pooled_frame_metrics'],flush=True)
