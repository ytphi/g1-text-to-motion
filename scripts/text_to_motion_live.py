"""
Interactive "type text, watch the G1 move" loop.

Wires together the 4 TEXEDO pipeline steps + our SONIC converter + SONIC's
MuJoCo viewer (see Obsidian note "TEXEDO Pipeline Explained" for what each
step actually does):

    prompt -> pipeline.generate -> [pipeline.score -> select_best_of_n] ->
    texedo_to_sonic.py -> visualize_motion.py (relaunched per prompt)

Each new prompt overwrites the same live-motion folder and restarts the
viewer subprocess, so the MuJoCo window will flicker/reopen briefly on every
prompt (see the "yes, but with a tradeoff" discussion before this was built:
a persistent, non-restarting window would need patching visualize_motion.py
to hot-swap its loaded motion instead of loading once at startup, which is a
bigger change than this v1 loop needed).

Usage:
    python3 scripts/text_to_motion_live.py
    python3 scripts/text_to_motion_live.py --num-samples 4
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TEXEDO_DIR = Path.home() / "robotics/platforms/unitree/TEXEDO"
TEXEDO_VENV_PYTHON = TEXEDO_DIR / ".venv/bin/python"
SONIC_DEPLOY_DIR = Path.home() / "robotics/platforms/unitree/GR00T-WholeBodyControl/gear_sonic_deploy"
CONVERTER = Path(__file__).resolve().parent / "texedo_to_sonic.py"
REPO_ROOT = Path(__file__).resolve().parent.parent
# The SONIC MuJoCo viewer (visualize_motion.py) needs mujoco/scipy/mink/etc. Launch it
# with a Python that has them: $SONIC_VIEWER_PYTHON, else the texedo310 conda env if
# present, else this interpreter.
_TEXEDO310 = Path.home() / "dev/python/miniconda3/envs/texedo310/bin/python"
SONIC_VIEWER_PYTHON = Path(os.environ.get("SONIC_VIEWER_PYTHON",
                                          _TEXEDO310 if _TEXEDO310.exists() else sys.executable))
VERIFIED_MOTIONS_DIR = REPO_ROOT / "verified_motions"

CANDIDATES_DIR = Path("/tmp/texedo_live_candidates")
LIVE_MOTION_NAME = "live"

# HF models (e.g. flan-t5-base) are already cached locally; force offline mode
# so a stale/invalid ~/.cache/huggingface/token doesn't trip a 401 on lookup.
TEXEDO_ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=1,
                    help="TEXEDO candidates per prompt. 1 skips scoring/selection "
                         "entirely (fastest); >1 runs the full best-of-N pipeline.")
    p.add_argument("--alpha", type=float, default=1.0, help="best-of-N: dynamic-reward weight")
    p.add_argument("--beta", type=float, default=1.0, help="best-of-N: semantic-match weight")
    return p.parse_args()


def generate(prompt: str, num_samples: int) -> None:
    if CANDIDATES_DIR.exists():
        for f in CANDIDATES_DIR.glob("*"):
            if f.is_file():
                f.unlink()
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(TEXEDO_VENV_PYTHON), "-m", "pipeline.generate",
         "--prompt", prompt, "--num-samples", str(num_samples),
         "--out-dir", str(CANDIDATES_DIR)],
        cwd=TEXEDO_DIR, check=True, env=TEXEDO_ENV,
    )


def select_best(prompt: str, alpha: float, beta: float) -> Path:
    npy_files = sorted(glob.glob(str(CANDIDATES_DIR / "*.npy")))
    if not npy_files:
        raise RuntimeError(f"pipeline.generate produced no .npy files in {CANDIDATES_DIR}")
    if len(npy_files) == 1:
        return Path(npy_files[0])

    scores_csv = CANDIDATES_DIR / "scores.csv"
    subprocess.run(
        [str(TEXEDO_VENV_PYTHON), "-m", "pipeline.score",
         "--motion-dir", str(CANDIDATES_DIR), "--caption", prompt,
         "--output", str(scores_csv)],
        cwd=TEXEDO_DIR, check=True, env=TEXEDO_ENV,
    )
    best_dir = CANDIDATES_DIR / "best"
    subprocess.run(
        [str(TEXEDO_VENV_PYTHON), "-m", "pipeline.select_best_of_n",
         "--scores", str(scores_csv), "--alpha", str(alpha), "--beta", str(beta),
         "--motion-dir", str(CANDIDATES_DIR), "--copy-best-to", str(best_dir)],
        cwd=TEXEDO_DIR, check=True, env=TEXEDO_ENV,
    )
    winners = sorted(best_dir.glob("*.npy"))
    if not winners:
        raise RuntimeError(f"select_best_of_n produced no winner in {best_dir}")
    return winners[0]


def convert(winner_npy: Path) -> Path:
    out_dir = SONIC_DEPLOY_DIR / "reference/my_motions" / LIVE_MOTION_NAME
    subprocess.run(
        [sys.executable, str(CONVERTER), str(winner_npy), "--out", str(out_dir)],
        check=True,
    )
    return out_dir


def relaunch_viewer(viewer_proc, motion_dir: Path):
    if viewer_proc is not None and viewer_proc.poll() is None:
        viewer_proc.terminate()
        try:
            viewer_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            viewer_proc.kill()
            viewer_proc.wait()

    rel_motion_dir = motion_dir.relative_to(SONIC_DEPLOY_DIR)
    return subprocess.Popen(
        [str(SONIC_VIEWER_PYTHON), "visualize_motion.py", "--motion_dir", str(rel_motion_dir)],
        cwd=SONIC_DEPLOY_DIR,
    )


def save_motion(name: str, prompt: str, args, winner_npy: Path, csv_dir: Path) -> None:
    """Copy the current clip's source npy + SONIC CSVs + prompt/gen params into
    verified_motions/<name>/, per the save convention in the TEXEDO-SONIC-Integration
    Obsidian note. This is Tier 1 only (source data) -- it does not run or record a
    SONIC sim2sim rollout, so it doesn't by itself mean the motion is safe to
    execute on a real balance-correcting controller, only that it's saved for later."""
    dest = VERIFIED_MOTIONS_DIR / name
    if dest.exists():
        print(f"       '{name}' already exists at {dest} -- pick a different name, not overwriting")
        return

    dest_csv = dest / "sonic_csv"
    dest_csv.mkdir(parents=True)
    shutil.copy(winner_npy, dest / "motion.npy")
    for f in csv_dir.glob("*"):
        shutil.copy(f, dest_csv / f.name)

    with open(dest / "prompt.json", "w") as f:
        json.dump({
            "prompt": prompt,
            "num_samples": args.num_samples,
            "selected_candidate": winner_npy.name,
            "alpha": args.alpha if args.num_samples > 1 else None,
            "beta": args.beta if args.num_samples > 1 else None,
            "note": "single-sample generation, no best-of-N scoring/selection was run"
                     if args.num_samples == 1 else "best-of-N scoring/selection was run",
        }, f, indent=2)

    print(f"       saved -> {dest}")


def main():
    args = parse_args()
    viewer_proc = None
    print("Type a motion prompt (e.g. 'a person waves'), or empty line to quit.")
    try:
        while True:
            prompt = input("\nprompt> ").strip()
            if not prompt:
                break
            try:
                print(f"[1/3] generating {args.num_samples} candidate(s)...")
                generate(prompt, args.num_samples)

                print("[2/3] selecting best candidate...")
                winner = select_best(prompt, args.alpha, args.beta)
                print(f"       winner: {winner.name}")

                print("[3/3] converting + relaunching viewer...")
                motion_dir = convert(winner)
                viewer_proc = relaunch_viewer(viewer_proc, motion_dir)

                save_name = input("       save as (name, blank=skip): ").strip()
                if save_name:
                    save_motion(save_name, prompt, args, winner, motion_dir)
            except subprocess.CalledProcessError as e:
                print(f"step failed ({e.cmd[1] if len(e.cmd) > 1 else e.cmd}), try another prompt: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        if viewer_proc is not None and viewer_proc.poll() is None:
            viewer_proc.terminate()


if __name__ == "__main__":
    main()
