import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from clotting.experiments.prepare import normalize_crop, frame_record, prepare
from clotting.data.dataset import FrameDataset


class DataProtocolTests(unittest.TestCase):
    def test_flip_preserves_sides(self):
        a = np.zeros((224,448,3), dtype=np.uint8)
        a[:112,:224] = 50
        a[112:,:224] = 200
        a[:,224:] = 99
        left = normalize_crop(a, [0,0,224,224], 'bottom_to_top')
        self.assertEqual(int(left[0,0,0]), 200)
        self.assertEqual(int(left[-1,0,0]), 50)
        np.testing.assert_array_equal(normalize_crop(a,[0,0,224,224],'top_to_bottom'),a[:,:224])

    def test_clock_no_entry_offset(self):
        self.assertEqual(frame_record('x','left',299,30,10,100)['label'],0)
        self.assertEqual(frame_record('x','left',300,30,10,100)['label'],1)
        self.assertEqual(frame_record('x','left',300,30,10,100)['time_s'],10)

    def test_approved_selection_and_expected_counts(self):
        config=json.loads(Path('configs/six_controls.json').read_text())
        selected={(v['case'],a['roi_id']) for v in config['videos'] for a in v['channel_assignments']}
        self.assertEqual(len(selected),6)
        self.assertNotIn(('041026','left'),selected)
        self.assertNotIn(('041426','right'),selected)
        self.assertEqual(sum(v['frame_count']-a['entry_frame'] for v in config['videos'] for a in v['channel_assignments']),5142)

    def test_augmentation_keeps_vertical_order(self):
        img=np.repeat(np.arange(224,dtype=np.uint8)[:,None,None],224,axis=1)
        img=np.repeat(img,3,axis=2)
        ds=FrameDataset('.',[],train=True)
        for _ in range(50):
            out=ds._augment(img)
            self.assertLess(out[0].mean(),out[-1].mean())

    def test_prepare_synthetic_video(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);w=cv2.VideoWriter(str(p/'tiny.avi'),cv2.VideoWriter_fourcc(*'MJPG'),30,(32,32))
            for i in range(12):w.write(np.full((32,32,3),i*10,dtype=np.uint8))
            w.release()
            config={'speedup':10,'videos':[{'case':'test','filename':'tiny.avi','fps':30,'frame_count':12,'flow_direction':'top_to_bottom','channel_assignments':[{'roi_id':'right','entry_frame':3,'ot_seconds':2,'roi_xyxy':[16,0,32,32]}]}]}
            prepare(config,p,p/'out');r=json.loads((p/'out/frame_labels.json').read_text())['test']['right']
            self.assertEqual(len(r),9);self.assertEqual(r[0]['frame_idx'],3);self.assertEqual(r[0]['label'],0);self.assertEqual(r[3]['label'],1)
            self.assertEqual(len(list((p/'out/frames').glob('*.jpg'))),9)

if __name__=='__main__':unittest.main()
