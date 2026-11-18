"""Extract approved channel crops and OT labels from original control AVIs."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2


def normalize_crop(frame, roi, direction):
    x0, y0, x1, y1 = roi
    crop = frame[y0:y1, x0:x1]
    if direction == 'bottom_to_top':
        crop = cv2.flip(crop, 0)
    elif direction != 'top_to_bottom':
        raise ValueError(f'Unknown flow direction: {direction}')
    return cv2.resize(crop, (224, 224))


def frame_record(case, side, index, fps, speedup, ot):
    return {'frame_idx': index, 'time_s': index / fps,
            'label': int(index / fps * speedup >= ot),
            'path': f'{case}_{side}_{index:05d}.jpg'}


def prepare(config, videos_root, out):
    if (out / 'frame_labels.json').exists():
        raise ValueError('Output already contains labels; choose a new --out directory.')
    out.mkdir(parents=True, exist_ok=True)
    frames = out / 'frames'
    frames.mkdir(exist_ok=True)
    labels, provenance = {}, []
    for video in config['videos']:
        matches = list(videos_root.rglob(video['filename']))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {video['filename']}; found {len(matches)}")
        source = matches[0]
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f'Cannot open {source.name}; ensure the original AVI is downloaded locally.')
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if abs(fps - video['fps']) > .001 or count != video['frame_count']:
            cap.release()
            raise ValueError(f'Unexpected FPS/frame count for {source.name}: {fps}, {count}')
        case = video['case']
        labels[case] = {a['roi_id']: [] for a in video['channel_assignments']}
        for i in range(count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Unexpected decode failure: {source.name} frame {i}')
            for a in video['channel_assignments']:
                if i < a['entry_frame']:
                    continue
                side = a['roi_id']
                r = frame_record(case, side, i, fps, config['speedup'], a['ot_seconds'])
                crop = normalize_crop(frame, a['roi_xyxy'], video['flow_direction'])
                if not cv2.imwrite(str(frames / r['path']), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise IOError(f"Could not save {r['path']}")
                labels[case][side].append(r)
        cap.release()
        provenance.append({'case': case, 'filename': source.name,
                           'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    (out / 'dataset.json').write_text(json.dumps(config, indent=2) + '\n')
    (out / 'frame_labels.json').write_text(json.dumps(labels))
    (out / 'source_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f"Prepared {sum(len(r) for sides in labels.values() for r in sides.values())} frames in {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/six_controls.json'))
    parser.add_argument('--videos', type=Path, required=True, help='Folder containing the four original AVIs (searched recursively)')
    parser.add_argument('--out', type=Path, default=Path('data/six_controls'))
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.videos, args.out)
