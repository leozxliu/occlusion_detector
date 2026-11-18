"""Four-fold video-held-out evaluation. Protocol fixed before inspecting predictions."""
import os,sys,json,hashlib,random
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=Path('data/six_controls'))
parser.add_argument('--out',type=Path,default=Path('runs/six_channel_comparison'))
args=parser.parse_args()

from clotting.models.encoder import FrameClassifier
from clotting.data.dataset import FrameRecord,FrameDataset,class_weights
from clotting.eval.onset import detect_onset
OUT=args.out/'resnet';OUT.mkdir(parents=True,exist_ok=True)
FRAMES=args.data/'frames'
d=json.loads((args.data/'dataset.json').read_text())
protocol={'split':'leave-one-video-out, both channels together; four folds','clock':'video zero; experiment_s=10*video_s','initialization':'fresh ImageNet ResNet18 each fold, no prior occlusion weights','epochs':12,'seed':0,'training_sampling_fps':3,'test_sampling':'all frames at 30 fps after visible blood entry','train_transition_exclusion_experiment_s':5,'test_transition_exclusion':False,'optimizer':'AdamW lr=0.0003 weight_decay=0.0001','batch_size':32,'threshold':0.5,'sustain_frames':5,'model_selection':'fixed final epoch; no test-driven tuning','labels':'0 before workbook OT, 1 at/after OT; assumes persistent occlusion after OT','roi':'fixed x=[70,240] left and [480,670] right; full height; no label-based ROI fitting','onset_definition':'start of first run of 5 positive frames; confirmation arrives 4 frames later','orientation':'top_to_bottom; source orientation from manifest; no vertical augmentation','scope':'OT-based experimental labels, not independent frame-by-frame annotations'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
records={};meta={}
manifest=json.loads((args.data/'frame_labels.json').read_text())
for v in d['videos']:
 case=v['case'];records[case]={}
 for assignment in v['channel_assignments']:
  side=assignment['roi_id'];rr=[FrameRecord(**r) for r in manifest[case][side]]
  assert rr and all((FRAMES/r.path).exists() for r in rr)
  records[case][side]=rr;ot=assignment['ot_seconds'];last=rr[-1].time_s
  assignment['ot_observable']=ot/10<=last
  meta[(case,side)]={'case':case,'side':side,'channel':assignment['workbook_channel'],'ot_s':ot,'entry_frame':rr[0].frame_idx,'last_video_s':last,'observable':ot/10<=last,'fps':30.0}
 assert set(records[case])==set(manifest[case])
assert sum(len(r) for sides in records.values() for r in sides.values())==5142
(OUT/'dataset.json').write_text(json.dumps(d,indent=2)+'\n')
(OUT/'frame_labels.json').write_text(json.dumps(manifest))
print('Loaded 5142 verified normalized frames for six approved channels',flush=True)
device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu');torch.set_num_threads(4);all_y=[];all_p=[];results=[]
def metrics(y,p):
 y=np.asarray(y);p=np.asarray(p);pred=p>0.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'n_frames':len(y),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall_sensitivity':float(recall_score(y,pred,zero_division=0)),'specificity':float(tn/(tn+fp)) if tn+fp else None,'f1':float(f1_score(y,pred,zero_division=0)),'auroc':float(roc_auc_score(y,p)) if len(set(y))==2 else None}
for case in records:
 random.seed(0);np.random.seed(0);torch.manual_seed(0)
 train=[]
 for c in records:
  if c==case:continue
  for side,rr in records[c].items():
   ot=meta[(c,side)]['ot_s'];train.extend(r for r in rr if r.frame_idx%10==0 and abs(r.time_s*10-ot)>5)
 assert all(not r.path.startswith(case) for r in train)
 model=FrameClassifier('resnet18',True,.2).to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights(train).to(device))
 loader=DataLoader(FrameDataset(FRAMES,train,224,True),batch_size=32,shuffle=True,num_workers=0)
 print('FOLD',case,'train cases',[c for c in records if c!=case],'frames',len(train),flush=True)
 history=[]
 for epoch in range(12):
  model.train();total=0
  for x,y,*_ in loader:
   x=x.to(device);y=y.long().to(device);opt.zero_grad();loss=loss_fn(model(x),y);loss.backward();opt.step();total+=loss.item()*len(x)
  history.append(total/len(train));print(case,'epoch',epoch+1,'loss',round(history[-1],5),flush=True)
 torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'protocol':protocol,'held_out_video':case,'training_cases':[c for c in records if c!=case]},OUT/f'model_test_{case}.pt')
 model.eval();fold={'test_case':case,'training_cases':[c for c in records if c!=case],'training_frames':len(train),'loss':history,'channels':[]};fy=[];fp=[]
 for side,rr in records[case].items():
  probs=[];loader=DataLoader(FrameDataset(FRAMES,rr,224,False),batch_size=64,shuffle=False,num_workers=0)
  with torch.inference_mode():
   for x,*_ in loader:probs.extend(torch.softmax(model(x.to(device)),dim=1)[:,1].cpu().tolist())
  y=[r.label for r in rr];ts=[r.time_s for r in rr];onset=detect_onset(np.array(ts),np.array(probs),.5,5);m=dict(meta[(case,side)]);m.update({'metrics':metrics(y,probs),'predicted_onset_video_s':onset,'predicted_onset_experiment_s':onset*10 if onset is not None else None,'error_experiment_s':onset*10-m['ot_s'] if onset is not None and m['observable'] else None,'false_alarm_on_censored_channel':onset is not None and not m['observable']})
  (OUT/f'signal_{case}_{side}.json').write_text(json.dumps({'video_s':ts,'probability':probs,'label':y}));fold['channels'].append(m);fy+=y;fp+=probs
 fold['metrics']=metrics(fy,fp);results.append(fold);all_y+=fy;all_p+=fp
 (OUT/'progress.json').write_text(json.dumps(results,indent=2));print('RESULT',case,fold['metrics'],flush=True)
 del model,opt
 if device=='mps':torch.mps.empty_cache()
channels=[c for f in results for c in f['channels']];observable=[c for c in channels if c['observable']];errors=[c['error_experiment_s'] for c in observable if c['error_experiment_s'] is not None]
summary={'protocol':protocol,'device':device,'folds':results,'pooled_frame_metrics':metrics(all_y,all_p),'macro_video_f1':float(np.mean([f['metrics']['f1'] for f in results])),'observable_events':len(observable),'observable_events_with_trigger':len(errors),'missed_observable_events':sum(c['predicted_onset_video_s'] is None for c in observable),'censored_channels':sum(not c['observable'] for c in channels),'false_alarms_on_censored_channels':sum(c['false_alarm_on_censored_channel'] for c in channels),'onset_mae_experimental_s_detected_only':float(np.mean(np.abs(errors))) if errors else None,'onset_median_abs_error_experimental_s_detected_only':float(np.median(np.abs(errors))) if errors else None,'notes':['Video-held-out cross-validation, not an untouched external test cohort.','Frames within a video are correlated; n=4 independent recordings.','OT clock alignment supplied by user; labels assume occlusion persists after OT.','No hyperparameters selected using test predictions.']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(4,2,figsize=(13,12))
for row,f in enumerate(results):
 for col,c in enumerate(f['channels']):
  ax=axs[row,col];s=json.loads((OUT/f"signal_{c['case']}_{c['side']}.json").read_text());ax.plot(np.array(s['video_s'])*10,s['probability']);ax.axvline(c['ot_s'],color='green',label='Workbook OT');ax.axhline(.5,color='gray',ls=':')
  if c['predicted_onset_experiment_s'] is not None:ax.axvline(c['predicted_onset_experiment_s'],color='red',ls='--',label='Predicted onset')
  ax.set_xlim(0,max(c['last_video_s']*10,c['ot_s']+5));ax.set_ylim(-.03,1.03);ax.set_title(f"{c['case']} {c['side']} / ch{c['channel']}");ax.set_xlabel('Experimental seconds from video zero');ax.set_ylabel('P(occluded)');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'heldout_predictions.png',dpi=150);print('DONE',json.dumps({k:v for k,v in summary.items() if k not in ['folds','protocol']}),flush=True)
