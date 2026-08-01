# G1 Text-to-Motion

Type a text prompt, watch the Unitree G1 humanoid perform it in MuJoCo. This wires
together [TEXEDO](https://github.com/JianuoCao/TEXEDO) (text-conditioned humanoid
motion generation) with NVIDIA's [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
SONIC sim2sim MuJoCo deployment.

```
prompt -> TEXEDO pipeline.generate -> [pipeline.score -> select_best_of_n] ->
texedo_to_sonic.py -> SONIC visualize_motion.py
```

## Layout

- `scripts/text_to_motion_live.py` — interactive loop: type a prompt, get a live MuJoCo viewer
- `scripts/texedo_to_sonic.py` — converts a TEXEDO `(T, 36)` motion `.npy` into SONIC's
  root-only reference-motion CSV format
- `scripts/inspect_joint_activity.py` — ranks which of the G1's 29 joints actually moved
  in a generated motion (range + RMS velocity), since TEXEDO has no explicit
  text→joint mapping to inspect directly
- `verified_motions/` — saved motions (source `.npy` + converted SONIC CSVs + the
  prompt/generation params used), one folder per motion name

## Prerequisites

Expects both dependencies cloned as siblings under `~/robotics/platforms/unitree/`:

```
~/robotics/platforms/unitree/
├── TEXEDO/                      (with a .venv, see its own setup instructions)
├── GR00T-WholeBodyControl/      (SONIC deploy code + assets)
└── g1-text-to-motion/           (this repo)
```

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 scripts/text_to_motion_live.py
python3 scripts/text_to_motion_live.py --num-samples 4   # best-of-N scoring/selection
```

Type a prompt (e.g. `a person waves`), wait for the MuJoCo viewer to open, optionally
save the result under a name to `verified_motions/`. Empty line quits.

```bash
python3 scripts/inspect_joint_activity.py verified_motions/jogging/motion.npy --prompt "jogging"
```

## Notes

- Saving to `verified_motions/` only stores the source data (Tier 1) — it does not
  run or record a SONIC sim2sim rollout, so it doesn't by itself certify the motion
  as safe to execute on a real balance-correcting controller.
- Each new prompt overwrites the same live-motion folder and restarts the viewer
  subprocess, so the MuJoCo window will briefly flicker/reopen per prompt.
