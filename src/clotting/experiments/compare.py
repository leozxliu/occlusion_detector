from pathlib import Path
import json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=Path('data/six_controls'))
parser.add_argument('--out',type=Path,default=Path('runs/six_channel_comparison'))
args=parser.parse_args()
ROOT=args.out;reports={}
fig,axs=plt.subplots(6,1,figsize=(11,17));keys=None
for name in ['resnet','motion']:
 out=ROOT/name;s=json.loads((out/'summary.json').read_text());rows=[];episodes=0;exposure=0;early=0;errors=[];training_counts={}
 for f in s['folds']:
  assert f['test_case'] not in f['training_cases'];training_counts[f['test_case']]=f['training_frames']
  for c in f['channels']:
   sig=json.loads((out/f"signal_{c['case']}_{c['side']}.json").read_text());t=np.array(sig['video_s'])*10;p=np.array(sig['probability']);y=np.array(sig['label']);assert len(t)==len(y)==len(p)
   assert np.array_equal(y,(t>=c['ot_s']).astype(int));pre=t<c['ot_s'];exposure+=float(pre.sum()/30*10)
   starts=[];i=0
   while i<len(p):
    if p[i]<=.5:i+=1;continue
    start=i
    while i<len(p) and p[i]>.5:i+=1
    if i-start>=5 and t[start+4]<c['ot_s']:starts.append(float(t[start]))
   episodes+=len(starts);err=c['error_experiment_s'];early+=int(err is not None and err < -5)
   if err is not None:errors.append(err)
   row={'case':c['case'],'side':c['side'],'channel':c['channel'],'ot_s':c['ot_s'],'predicted_onset_s':c['predicted_onset_experiment_s'],'error_s':err,'pre_ot_alarm_episodes':len(starts),'pre_ot_alarm_starts_s':starts,'recall':c['metrics']['recall_sensitivity']};rows.append(row)
 if keys is None:keys=[(r['case'],r['side']) for r in rows]
 assert keys==[(r['case'],r['side']) for r in rows]
 report={'frame_metrics':s['pooled_frame_metrics'],'onset_mae_s':float(np.mean(np.abs(errors))) if errors else None,'onset_median_absolute_error_s':float(np.median(np.abs(errors))) if errors else None,'triggered_channels':len(errors),'eligible_channels':6,'first_triggers_more_than_5s_early':early,'pre_ot_sustained_alarm_episodes':episodes,'pre_ot_exposure_experimental_minutes':exposure/60,'pre_ot_alarm_episodes_per_experimental_minute':episodes/(exposure/60),'channels':rows,'training_counts':training_counts};reports[name]=report
 for i,r in enumerate(rows):
  sig=json.loads((out/f"signal_{r['case']}_{r['side']}.json").read_text());ax=axs[i];ax.plot(np.array(sig['video_s'])*10,sig['probability'],label=name,alpha=.8)
  if name=='motion':ax.axvline(r['ot_s'],color='black',ls='--',label='Workbook OT');ax.axhline(.5,color='gray',ls=':');ax.set_title(f"{r['case']} {r['side']} / ch{r['channel']}");ax.set_ylabel('P(occluded)');ax.set_xlabel('Experimental seconds from video zero');ax.legend()
assert reports['resnet']['training_counts']==reports['motion']['training_counts']
for case,side in keys:
 a=json.loads((ROOT/'resnet'/f'signal_{case}_{side}.json').read_text());b=json.loads((ROOT/'motion'/f'signal_{case}_{side}.json').read_text());assert a['label']==b['label'] and a['video_s']==b['video_s']
fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=140)
result={'methods':reports,'false_alarm_definition':'One run of >=5 consecutive p>0.5 frames, with confirmation frame before workbook OT; separate runs counted separately. Rate denominator is eligible pre-OT experimental time.','timing_definition':'First sustained trigger, start-of-run timestamp; later triggers not substituted for early false alarms.','limitations':['Four videos, six selected channels; development cross-validation after inspecting these cases.','All selected channels eventually occlude; no independent never-occluded video evaluation.','Labels assume video-zero mass alignment and persistent occlusion after OT.','Motion features are frame differences, not calibrated physical velocity.','Excluded channels are omitted from both training and testing.'],'verification':'Identical train counts, held-out video identities, test frame timestamps and labels verified.'}
shared=[(a,b) for a,b in zip(reports['resnet']['channels'],reports['motion']['channels']) if a['error_s'] is not None and b['error_s'] is not None]
result['common_triggered_channels']={'n':len(shared),'resnet_mae_s':float(np.mean([abs(a['error_s']) for a,b in shared])) if shared else None,'motion_mae_s':float(np.mean([abs(b['error_s']) for a,b in shared])) if shared else None}
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2))
lines=['SIX APPROVED CHANNELS: RESNET VS MOTION','Same four leave-one-video-out folds, 10x, video-zero clock, top-to-bottom, no vertical augmentation.',result['verification'],result['false_alarm_definition'],result['timing_definition'],'']
for name,r in reports.items():
 lines.extend([name.upper(),json.dumps({k:v for k,v in r.items() if k!='channels'},indent=2)])
 for c in r['channels']:lines.append(json.dumps(c))
lines.extend(result['limitations']);(ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(reports,indent=2))
