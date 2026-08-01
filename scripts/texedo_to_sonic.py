"""
Convert a TEXEDO (T,36) motion .npy into SONIC's root-only reference-motion
CSV format, so it can be played back via SONIC's MuJoCo sim2sim deployment.

TEXEDO format (see TEXEDO/docs/FORMAT.md), 50 fps:
    motion[:, 0:3]  — root position (x, y, z), meters
    motion[:, 3:7]  — root quaternion (w, x, y, z), unit-norm
    motion[:, 7:36] — 29 joint angles, radians

SONIC format (see GR00T-WholeBodyControl/docs/source/references/motion_reference.md),
50 Hz. Joint order has been verified index-for-index against SONIC's own
G1_ISAACLAB_JOINTS (gear_sonic/envs/manager_env/robots/g1.py) — no reordering
or resampling needed between the two formats.

Usage:
    python scripts/texedo_to_sonic.py motion.npy --out reference/my_motions/wave_hand/
"""

import argparse
from pathlib import Path

import numpy as np

FPS = 50.0
NUM_JOINTS = 29


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("npy_path", help="Path to a TEXEDO (T,36) motion .npy file")
    p.add_argument("--out", required=True, help="Output motion folder (SONIC reference/<name>/)")
    return p.parse_args()


def joint_velocities(joint_pos: np.ndarray, dt: float) -> np.ndarray:
    """Central finite difference, one-sided at the boundaries."""
    vel = np.zeros_like(joint_pos)
    vel[1:-1] = (joint_pos[2:] - joint_pos[:-2]) / (2 * dt)
    vel[0] = (joint_pos[1] - joint_pos[0]) / dt
    vel[-1] = (joint_pos[-1] - joint_pos[-2]) / dt
    return vel


def write_csv(path: Path, array: np.ndarray, headers: list[str]) -> None:
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in array:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def convert(npy_path: str, out_dir: str) -> None:
    motion = np.load(npy_path)
    if motion.ndim != 2 or motion.shape[1] != 36:
        raise ValueError(f"expected (T, 36) TEXEDO motion, got {motion.shape}")

    root_pos = motion[:, 0:3]
    root_quat = motion[:, 3:7]
    joint_pos = motion[:, 7:36]
    dt = 1.0 / FPS

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "joint_pos.csv", joint_pos, [f"joint_{i}" for i in range(NUM_JOINTS)])
    write_csv(out / "joint_vel.csv", joint_velocities(joint_pos, dt),
              [f"joint_vel_{i}" for i in range(NUM_JOINTS)])
    write_csv(out / "body_pos.csv", root_pos, ["body_0_x", "body_0_y", "body_0_z"])
    write_csv(out / "body_quat.csv", root_quat, ["body_0_w", "body_0_x", "body_0_y", "body_0_z"])

    with open(out / "metadata.txt", "w") as f:
        f.write(f"Metadata for: {out.name}\n")
        f.write("=" * 30 + "\n\n")
        f.write("Body part indexes:\n[0]\n\n")
        f.write(f"Total timesteps: {motion.shape[0]}\n")

    print(f"wrote {motion.shape[0]} frames -> {out}")


if __name__ == "__main__":
    args = parse_args()
    convert(args.npy_path, args.out)
