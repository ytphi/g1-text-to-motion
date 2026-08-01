"""Rank which of the G1's 29 joints actually move in a TEXEDO-generated motion.

There is no explicit text->joint mapping anywhere in TEXEDO (see the "TEXEDO
generation mechanism and joint mapping" Obsidian note) - the FLAN-T5 generator
learns an implicit association during training. This script answers the
question empirically instead: given a generated (T,36) motion, which joints
were actually driven, by how much, ranked and optionally grouped by body part.

Two activity metrics per joint:
    range_deg     peak-to-peak angular excursion (captures held poses)
    vel_rms_dps   RMS angular velocity via central finite difference
                  (captures oscillatory motion like waving)

Usage:
    python scripts/inspect_joint_activity.py motion.npy
    python scripts/inspect_joint_activity.py motion.npy --prompt "a person waves"
    python scripts/inspect_joint_activity.py motion.npy --baseline idle.npy
    python scripts/inspect_joint_activity.py motion.npy --top 8 --plot activity.png
"""

import argparse

import numpy as np

FPS = 50.0
NUM_JOINTS = 29

# motion[:, 7:36] order, from TEXEDO/docs/FORMAT.md (also texedo_to_sonic.py).
JOINT_NAMES = [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw", "left_hip_roll",
    "right_hip_roll", "waist_roll", "left_hip_yaw", "right_hip_yaw",
    "waist_pitch", "left_knee", "right_knee", "left_shoulder_pitch",
    "right_shoulder_pitch", "left_ankle_pitch", "right_ankle_pitch",
    "left_shoulder_roll", "right_shoulder_roll", "left_ankle_roll",
    "right_ankle_roll", "left_shoulder_yaw", "right_shoulder_yaw",
    "left_elbow", "right_elbow", "left_wrist_roll", "right_wrist_roll",
    "left_wrist_pitch", "right_wrist_pitch", "left_wrist_yaw", "right_wrist_yaw",
]
assert len(JOINT_NAMES) == NUM_JOINTS

GROUPS = {
    "leg": ["hip", "knee", "ankle"],
    "waist": ["waist"],
    "arm": ["shoulder", "elbow", "wrist"],
}


def joint_group(name: str) -> str:
    for group, keywords in GROUPS.items():
        if any(k in name for k in keywords):
            return group
    return "other"


def joint_velocities(joint_pos: np.ndarray, dt: float) -> np.ndarray:
    """Central finite difference, one-sided at the boundaries."""
    vel = np.zeros_like(joint_pos)
    vel[1:-1] = (joint_pos[2:] - joint_pos[:-2]) / (2 * dt)
    vel[0] = (joint_pos[1] - joint_pos[0]) / dt
    vel[-1] = (joint_pos[-1] - joint_pos[-2]) / dt
    return vel


def load_joint_pos(npy_path: str) -> np.ndarray:
    motion = np.load(npy_path)
    if motion.ndim != 2 or motion.shape[1] != 36:
        raise ValueError(f"expected (T, 36) TEXEDO motion, got {motion.shape}")
    return motion[:, 7:36]


def joint_activity(joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / FPS
    vel = joint_velocities(joint_pos, dt)
    range_deg = np.degrees(joint_pos.max(axis=0) - joint_pos.min(axis=0))
    vel_rms_dps = np.degrees(np.sqrt((vel ** 2).mean(axis=0)))
    return range_deg, vel_rms_dps


def print_table(range_deg: np.ndarray, vel_rms_dps: np.ndarray, top: int) -> None:
    order = np.argsort(-vel_rms_dps)  # rank by movement energy, most active first
    print(f"{'joint':<22}{'group':<8}{'range_deg':>12}{'vel_rms_dps':>14}")
    print("-" * 56)
    for rank, idx in enumerate(order):
        marker = "*" if rank < top else " "
        print(f"{marker}{JOINT_NAMES[idx]:<21}{joint_group(JOINT_NAMES[idx]):<8}"
              f"{range_deg[idx]:>12.1f}{vel_rms_dps[idx]:>14.1f}")
    print(f"\n(* = top {top} most active by vel_rms_dps)")


def print_baseline_diff(range_deg, vel_rms_dps, base_range_deg, base_vel_rms_dps, top: int) -> None:
    d_vel = vel_rms_dps - base_vel_rms_dps
    order = np.argsort(-np.abs(d_vel))
    print(f"{'joint':<22}{'group':<8}{'d_range_deg':>13}{'d_vel_rms_dps':>15}")
    print("-" * 58)
    for rank, idx in enumerate(order):
        marker = "*" if rank < top else " "
        print(f"{marker}{JOINT_NAMES[idx]:<21}{joint_group(JOINT_NAMES[idx]):<8}"
              f"{range_deg[idx] - base_range_deg[idx]:>13.1f}{d_vel[idx]:>15.1f}")
    print(f"\n(* = top {top} most changed vs. baseline, by |d_vel_rms_dps|)")
    print("(positive d_vel_rms_dps = more active than baseline)")


def plot_activity(range_deg, vel_rms_dps, prompt: str, out_path: str) -> None:
    import matplotlib.pyplot as plt

    order = np.argsort(-vel_rms_dps)
    names = [JOINT_NAMES[i] for i in order]
    colors = {"leg": "tab:blue", "waist": "tab:orange", "arm": "tab:green", "other": "tab:gray"}
    bar_colors = [colors[joint_group(n)] for n in names]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(range(len(names)), vel_rms_dps[order], color=bar_colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("velocity RMS (deg/s)")
    title = f'joint activity: "{prompt}"' if prompt else "joint activity"
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote plot -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npy_path", help="Path to a TEXEDO (T,36) motion .npy file")
    p.add_argument("--prompt", default="", help="Text prompt that produced this motion (label only)")
    p.add_argument("--baseline", default=None,
                   help="Optional (T,36) npy of a neutral/idle motion to diff against")
    p.add_argument("--top", type=int, default=6, help="How many top joints to flag")
    p.add_argument("--plot", default=None, help="If set, save a bar chart to this path")
    return p.parse_args()


def main():
    args = parse_args()
    joint_pos = load_joint_pos(args.npy_path)
    range_deg, vel_rms_dps = joint_activity(joint_pos)

    label = f' for "{args.prompt}"' if args.prompt else ""
    print(f"{joint_pos.shape[0]} frames ({joint_pos.shape[0] / FPS:.1f}s){label}\n")

    if args.baseline:
        base_joint_pos = load_joint_pos(args.baseline)
        base_range_deg, base_vel_rms_dps = joint_activity(base_joint_pos)
        print_baseline_diff(range_deg, vel_rms_dps, base_range_deg, base_vel_rms_dps, args.top)
    else:
        print_table(range_deg, vel_rms_dps, args.top)

    if args.plot:
        plot_activity(range_deg, vel_rms_dps, args.prompt, args.plot)


if __name__ == "__main__":
    main()
