# Clotting: occlusion-onset detection in microfluidic thrombosis video

Detect the instant blood flow stops (occlusion onset) in brightfield microscopy
video of microfluidic channels, where a growing thrombus eventually blocks the vessel.

## Quick demo

The model (trained only on the control video) applied to a **new, unseen**
sample. The curve is the per-frame occlusion probability; the orange line is the
detected onset; the filmstrip shows the channel flipping from translucent flow
(green) to packed thrombus (red) exactly at that instant.

## Data

Data comes from our in-house brightfield microscopy videos of blood flowing through
microfluidic channels until a thrombus occludes the vessel. Raw videos live in
`data/raw/` (30 fps, 694x510, playback is **10x sped up**, so
`real_time_s = video_time_s * 10`).

Each video's two channels are treated as **independent datasets**.

Temporal labeling of the training video (real-time seconds):

- `t < 90 s` -> **flowing** (0)
- `t > 200 s` -> **occluded** (1)
- `90-200 s` -> **excluded** (human onset judgement is unreliable there).

There is deliberately **no onset "ground truth"**: the true stop instant is unknown,
so the detector reports the *detected* onset rather than an error against a guess.

## Method — per-frame CNN classifier

A timm ResNet classifies each cropped channel frame as flowing/occluded.
Evaluation is **leave-one-channel-out**: train on one channel, test on the other.
Onset is triggered when occlusion probability stays above threshold for N frames.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements from pyproject
```



## Run

> ```bash
> export PYTHONPATH=src
> # 1) extract per-channel frames + labels
> python -m clotting.io.extract --config configs/data.yaml
> # 2) train + evaluate the classifier (leave-one-channel-out)
> python -m clotting.train.classify --data-config configs/data.yaml --config configs/baseline.yaml
> # 3) visualize results
> python -m clotting.eval.plot_signal
> python -m clotting.eval.demo
> # 4) run the trained model on a new video (auto-detects channels + orientation)
> python -m clotting.infer.predict --video "data/raw/example_video.avi" --out runs/prp
> ```

Outputs land in `runs/stage1/`:

- `model_test_<ch>.pt` — model for each fold
- `signal_test_<ch>.csv` — per-frame occlusion probability over the whole video
- `summary.json` — frame F1 and detected onset per fold



## License

Released under the [MIT License](LICENSE).