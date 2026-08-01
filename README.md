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

## Example Prompts (validated on real G1 hardware)

The TEXEDO paper ([arXiv:2606.22998](https://arxiv.org/abs/2606.22998), Table 11) deployed
30 text prompts on a physical Unitree G1 and reports **30/30 successful executions**
(no falls, no early termination). These are a good starting point since they're
known-good phrasing for this pipeline:

- `a person walks in a circle clockwise.`
- `a person walks forward slowly.`
- `A person throws a right hook ahead.`
- `A person dances happily ahead.`
- `the person is stretching, with their arms above their head.`
- `a man is doing jumping jacks.`
- `A person pivots right while moving cheerfully.`
- `A person executes a left jab forward with force and then throws a left hook ahead with full force.`
- `The person moves while boxing and walking forward.`
- `A person carries an object sideways to the right balancing carefully.`
- `The person stumbles backward in a slow, zombie-like shamble.`

(Full list of 30 in the paper's Table 11, spanning locomotion, upper-body gestures,
and combinations of the two.)

## Prompt Phrasing Limitations

TEXEDO's generator is trained on captioned motion-capture data (AMASS + CLAW), and
its behavior on prompts that drift from that caption style is not guaranteed:

- **Always include an explicit subject.** Every validated prompt above starts with
  "a person," "the person," or "a man." A bare imperative or subject-less phrase
  (e.g. `wave the right hand` instead of `a person waves with the right hand`) is
  outside the training distribution and may generate a less reliable motion.
- **Sequential compound actions (`X, then Y`) tend to work.** e.g. the validated
  `"...executes a left jab forward with force and then throws a left hook ahead
  with full force"` — the generator handles these as one action following another.
- **Simultaneous, unrelated compound actions are less reliable.** A single 36-dim
  frame has to encode one coherent whole-body pose, so combining two independent
  primitives on unrelated body regions in the same instant (e.g. `"jumping while
  clapping"`) asks the generator to blend two things it wasn't necessarily trained
  to co-occur. The paper's own qualitative failure case (Fig. 5) is a similar
  compound clause — `"A person rotates left 45 degrees while moving with
  difficulty"` — which under a weaker (single-verifier) selection strategy produced
  a dynamically stable but semantically mismatched motion (`R_text = 0.014`).
  TEXEDO's full best-of-N selection (`--num-samples > 1`) exists precisely to filter
  out these misses, so treat compound/simultaneous prompts as candidates worth
  sampling more heavily rather than assuming the first candidate is representative.

## Notes

- Saving to `verified_motions/` only stores the source data (Tier 1) — it does not
  run or record a SONIC sim2sim rollout, so it doesn't by itself certify the motion
  as safe to execute on a real balance-correcting controller.
- Each new prompt overwrites the same live-motion folder and restarts the viewer
  subprocess, so the MuJoCo window will briefly flicker/reopen per prompt.
