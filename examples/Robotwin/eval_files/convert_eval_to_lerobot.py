"""
Convert RoboTwin eval output (HDF5) to LeRobot v2.1 format.

No dependency on the `lerobot` package — generates all files directly.
Compatible with starVLA's gr00t_lerobot dataloader.

Reads from eval output directory structure:
    {input_dir}/{task_name}/aloha-agilex_{task_config}_{N}/
        ├── data/episode*.hdf5
        ├── instructions/  (optional)
        ├── video/episode*.mp4
        ├── scene_info.json
        └── seed.txt

Produces a LeRobot v2.1 dataset at {output_dir}/{task_name}/ with:
    data/chunk-000/episode_NNNNNN.parquet
    videos/chunk-000/observation.images.{cam}/episode_NNNNNN.mp4
    meta/info.json
    meta/episodes.jsonl
    meta/tasks.jsonl
    meta/episodes_stats.jsonl
    meta/modality.json

Usage:
    # Convert a single task
    python convert_eval_to_lerobot.py \
        --input-dir /path/to/eval_output_v2 \
        --output-dir /path/to/lerobot_datasets \
        --task-name click_alarmclock

    # Convert all tasks in eval output
    python convert_eval_to_lerobot.py \
        --input-dir /path/to/eval_output_v2 \
        --output-dir /path/to/lerobot_datasets \
        --all-tasks

    # Skip video encoding (faster, parquet-only)
    python convert_eval_to_lerobot.py \
        --input-dir /path/to/eval_output_v2 \
        --output-dir /path/to/lerobot_datasets \
        --all-tasks --no-video

    # Convert only successful episodes
    python convert_eval_to_lerobot.py \
        --input-dir /path/to/eval_output_v1_failure \
        --output-dir /path/to/lerobot_datasets \
        --all-tasks --filter-success success

    # Convert only failed episodes
    python convert_eval_to_lerobot.py \
        --input-dir /path/to/eval_output_v1_failure \
        --output-dir /path/to/lerobot_datasets \
        --all-tasks --filter-success failure
"""

import argparse
import json
import os
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


CAMERA_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

MOTORS = [
    "left_waist", "left_shoulder", "left_elbow",
    "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow",
    "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]

FPS = 15


def decode_jpeg_images(jpeg_bytes_array, target_h=480, target_w=640):
    """Decode JPEG byte strings from HDF5 to numpy BGR arrays, resize to target."""
    imgs = []
    for data in jpeg_bytes_array:
        if isinstance(data, bytes):
            nparr = np.frombuffer(data, np.uint8)
        else:
            nparr = np.frombuffer(bytes(data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            imgs.append(img)
        else:
            imgs.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
    return np.array(imgs)


def load_instruction_from_json(instructions_dir, ep_idx):
    """Load instruction from per-episode JSON file."""
    json_path = os.path.join(instructions_dir, f"episode{ep_idx}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    candidates = data.get("seen", []) + data.get("unseen", [])
    if candidates:
        return candidates[0]
    return None


def load_episode_success(metadata_dir, ep_idx):
    """Load success flag from metadata JSON for a given episode."""
    json_path = os.path.join(metadata_dir, f"episode{ep_idx}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("success", None)


def find_eval_subdirs(input_dir, task_name):
    """Find all aloha-agilex_* subdirectories for a task."""
    task_dir = os.path.join(input_dir, task_name)
    if not os.path.isdir(task_dir):
        return []
    subdirs = []
    for name in sorted(os.listdir(task_dir)):
        full_path = os.path.join(task_dir, name)
        if os.path.isdir(full_path) and name.startswith("aloha-agilex_"):
            data_dir = os.path.join(full_path, "data")
            if os.path.isdir(data_dir):
                subdirs.append(full_path)
    return subdirs


def encode_video_ffmpeg(frames_bgr, output_path, fps=15):
    """Encode BGR frames to MP4 (h264) using ffmpeg."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_frames, H, W, _ = frames_bgr.shape
    frames_rgb = frames_bgr[..., ::-1].copy()  # BGR -> RGB, contiguous

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{W}x{H}",
        "-framerate", str(fps),
        "-i", "-",
        "-pix_fmt", "yuv420p",
        "-vcodec", "libx264",
        "-crf", "23",
        "-g", "2",
        "-threads", "2",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(frames_rgb.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        print(f"  [WARN] ffmpeg failed for {output_path}")


def compute_episode_stats(state, action):
    """Compute per-episode statistics for state and action."""
    stats = {}
    for key, data in [("observation.state", state), ("action", action)]:
        stats[key] = {
            "min": data.min(axis=0).tolist(),
            "max": data.max(axis=0).tolist(),
            "mean": data.mean(axis=0).tolist(),
            "std": data.std(axis=0, ddof=0).tolist(),
            "count": [len(data)],
        }
    return stats


def write_modality_json(output_dir):
    """Write meta/modality.json matching reference format."""
    modality = {
        "action": {
            "left_joints": {"start": 0, "end": 6, "original_key": "action"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "action"},
            "right_joints": {"start": 7, "end": 13, "original_key": "action"},
            "right_gripper": {"start": 13, "end": 14, "original_key": "action"},
        },
        "state": {
            "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
            "right_joints": {"start": 7, "end": 13, "original_key": "observation.state"},
            "right_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
        },
        "video": {
            "cam_high": {"original_key": "observation.images.cam_high"},
            "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
            "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "modality.json"), "w") as f:
        json.dump(modality, f, indent=4)


def write_info_json(output_dir, total_episodes, total_frames, total_tasks, total_videos, fps, encode_video):
    """Write meta/info.json in v2.1 format."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": [14],
            "names": [MOTORS],
        },
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [3, 480, 640],
            "names": ["channels", "height", "width"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    for scalar_key, dtype in [
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ]:
        features[scalar_key] = {
            "dtype": dtype,
            "shape": [1],
            "names": None,
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_videos,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


def process_single_episode(args_tuple):
    """Process a single episode — designed to run in a worker process."""
    (hdf5_path, instruction, ep_global_idx, global_index_start,
     dataset_dir, encode_video, fps) = args_tuple

    try:
        with h5py.File(hdf5_path, "r") as f:
            if "joint_action/vector" not in f:
                return None
            state = f["joint_action/vector"][:].astype(np.float32)
            num_frames = state.shape[0]

            if "action" in f:
                action = f["action"][:].astype(np.float32)
                if action.shape[0] != num_frames:
                    action_padded = np.zeros_like(state)
                    n = min(action.shape[0], num_frames)
                    action_padded[:n] = action[:n]
                    if n < num_frames:
                        action_padded[n:] = action[-1] if n > 0 else 0
                    action = action_padded
            else:
                action = np.zeros_like(state)
                action[:-1] = state[1:]
                action[-1] = state[-1]

            imgs_per_cam = {}
            if encode_video:
                for native_name, lerobot_name in CAMERA_MAP.items():
                    cam_path = f"observation/{native_name}/rgb"
                    if cam_path in f:
                        imgs_per_cam[lerobot_name] = decode_jpeg_images(f[cam_path][:])

    except Exception as e:
        print(f"  [ERROR] Failed to read {hdf5_path}: {e}")
        return None

    # Write per-episode parquet
    data_dir = os.path.join(dataset_dir, "data", "chunk-000")
    rows = {
        "observation.state": [state[i].tolist() for i in range(num_frames)],
        "action": [action[i].tolist() for i in range(num_frames)],
        "timestamp": [i / fps for i in range(num_frames)],
        "frame_index": list(range(num_frames)),
        "episode_index": [ep_global_idx] * num_frames,
        "index": list(range(global_index_start, global_index_start + num_frames)),
        "task_index": [0] * num_frames,  # placeholder, filled after merge
    }
    df = pd.DataFrame(rows)
    parquet_path = os.path.join(data_dir, f"episode_{ep_global_idx:06d}.parquet")
    df.to_parquet(parquet_path, index=False)

    # Encode videos concurrently (3 cameras in parallel via threads)
    if encode_video and imgs_per_cam:
        def _encode_cam(cam):
            if cam in imgs_per_cam and len(imgs_per_cam[cam]) > 0:
                video_path = os.path.join(
                    dataset_dir, "videos", "chunk-000",
                    f"observation.images.{cam}", f"episode_{ep_global_idx:06d}.mp4"
                )
                encode_video_ffmpeg(imgs_per_cam[cam], video_path, fps=fps)

        with ThreadPoolExecutor(max_workers=3) as tex:
            list(tex.map(_encode_cam, CAMERAS))

    # Compute stats
    ep_stats = compute_episode_stats(state, action)

    return {
        "episode_index": ep_global_idx,
        "instruction": instruction,
        "length": num_frames,
        "stats": ep_stats,
    }


def convert_single_task(
    input_dir: str,
    output_dir: str,
    task_name: str,
    encode_video: bool = True,
    fps: int = FPS,
    default_instruction: str = None,
    num_workers: int = 8,
    filter_success: str = None,
):
    """Convert all eval episodes for a single task to LeRobot v2.1 format."""
    subdirs = find_eval_subdirs(input_dir, task_name)
    if not subdirs:
        print(f"[SKIP] No eval data found for task: {task_name}")
        return

    all_hdf5_files = []
    all_instructions = []

    for subdir in subdirs:
        data_dir = os.path.join(subdir, "data")
        instructions_dir = os.path.join(subdir, "instructions")
        metadata_dir = os.path.join(subdir, "metadata")

        hdf5_files = sorted(
            glob(os.path.join(data_dir, "episode*.hdf5")),
            key=lambda x: int(os.path.basename(x).replace("episode", "").replace(".hdf5", "")),
        )

        for hdf5_path in hdf5_files:
            ep_idx = int(os.path.basename(hdf5_path).replace("episode", "").replace(".hdf5", ""))

            if filter_success is not None and os.path.isdir(metadata_dir):
                success = load_episode_success(metadata_dir, ep_idx)
                if success is not None:
                    want_success = (filter_success == "success")
                    if success != want_success:
                        continue

            instruction = None
            if os.path.isdir(instructions_dir):
                instruction = load_instruction_from_json(instructions_dir, ep_idx)
            if instruction is None:
                instruction = default_instruction or f"perform {task_name.replace('_', ' ')}"
            all_hdf5_files.append(hdf5_path)
            all_instructions.append(instruction)

    if not all_hdf5_files:
        print(f"[SKIP] No HDF5 episodes found for task: {task_name}" +
              (f" (filter: {filter_success})" if filter_success else ""))
        return

    filter_info = f", filter={filter_success}" if filter_success else ""
    print(f"[CONVERT] {task_name}: {len(all_hdf5_files)} episodes from {len(subdirs)} subdirs, workers={num_workers}{filter_info}")

    dataset_dir = os.path.join(output_dir, task_name)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    data_dir = os.path.join(dataset_dir, "data", "chunk-000")
    os.makedirs(data_dir, exist_ok=True)

    if encode_video:
        for cam in CAMERAS:
            os.makedirs(os.path.join(dataset_dir, "videos", "chunk-000", f"observation.images.{cam}"), exist_ok=True)

    # Pre-compute global_index offsets (need a sequential scan of frame counts)
    frame_counts = []
    for hdf5_path in all_hdf5_files:
        try:
            with h5py.File(hdf5_path, "r") as f:
                if "joint_action/vector" in f:
                    frame_counts.append(f["joint_action/vector"].shape[0])
                else:
                    frame_counts.append(0)
        except:
            frame_counts.append(0)

    global_index_offsets = []
    running = 0
    for c in frame_counts:
        global_index_offsets.append(running)
        running += c

    # Build args for each worker
    worker_args = []
    for ep_idx, (hdf5_path, instruction) in enumerate(zip(all_hdf5_files, all_instructions)):
        worker_args.append((
            hdf5_path, instruction, ep_idx, global_index_offsets[ep_idx],
            dataset_dir, encode_video, fps,
        ))

    # Process episodes in parallel
    results = [None] * len(worker_args)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_episode, args): i
            for i, args in enumerate(worker_args)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(worker_args), desc=f"  {task_name}"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"  [ERROR] Worker failed for episode {idx}: {e}")

    # Collect results and fix task_index in parquets
    episodes_meta = []
    tasks_list = []
    task_to_index = {}
    episodes_stats_list = []
    total_frames = 0

    for ep_idx, result in enumerate(results):
        if result is None:
            continue

        instruction = result["instruction"]
        if instruction not in task_to_index:
            task_to_index[instruction] = len(task_to_index)
            tasks_list.append(instruction)
        task_idx = task_to_index[instruction]

        # Fix task_index in the already-written parquet if needed
        if task_idx != 0:
            parquet_path = os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet")
            df = pd.read_parquet(parquet_path)
            df["task_index"] = task_idx
            df.to_parquet(parquet_path, index=False)

        episodes_stats_list.append({
            "episode_index": ep_idx,
            "stats": result["stats"],
        })

        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [instruction],
            "length": result["length"],
        })

        total_frames += result["length"]

    if not episodes_meta:
        print(f"[SKIP] No valid episodes for {task_name}")
        shutil.rmtree(dataset_dir)
        return

    # Write meta files
    meta_dir = os.path.join(dataset_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for idx, task in enumerate(tasks_list):
            f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")

    with open(os.path.join(meta_dir, "episodes_stats.jsonl"), "w") as f:
        for ep_stat in episodes_stats_list:
            f.write(json.dumps(ep_stat) + "\n")

    write_modality_json(dataset_dir)

    total_videos = len(episodes_meta) * len(CAMERAS) if encode_video else 0
    write_info_json(
        dataset_dir,
        total_episodes=len(episodes_meta),
        total_frames=total_frames,
        total_tasks=len(tasks_list),
        total_videos=total_videos,
        fps=fps,
        encode_video=encode_video,
    )

    print(f"[DONE] {task_name}: {len(episodes_meta)} episodes, {total_frames} frames -> {dataset_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert RoboTwin eval output to LeRobot v2.1 format (no lerobot dependency)")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to eval output directory (e.g. starVLA/datasets/eval_output_v2)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to output LeRobot dataset directory")
    parser.add_argument("--task-name", type=str, default=None,
                        help="Single task name to convert (e.g. click_alarmclock)")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Convert all tasks found in input-dir")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video encoding (faster, parquet-only)")
    parser.add_argument("--fps", type=int, default=FPS,
                        help=f"Dataset FPS (default: {FPS})")
    parser.add_argument("--default-instruction", type=str, default=None,
                        help="Default instruction if none found in data")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers for episode processing (default: 8)")
    parser.add_argument("--filter-success", type=str, choices=["success", "failure"], default=None,
                        help="Only convert successful or failed episodes (reads metadata/episodeN.json)")
    args = parser.parse_args()

    if not args.all_tasks and args.task_name is None:
        parser.error("Must specify --task-name or --all-tasks")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all_tasks:
        task_names = sorted([
            d for d in os.listdir(args.input_dir)
            if os.path.isdir(os.path.join(args.input_dir, d)) and not d.startswith(".")
        ])
        print(f"Found {len(task_names)} tasks: {task_names}")
    else:
        task_names = [args.task_name]

    for task_name in task_names:
        convert_single_task(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            task_name=task_name,
            encode_video=not args.no_video,
            fps=args.fps,
            default_instruction=args.default_instruction,
            num_workers=args.workers,
            filter_success=args.filter_success,
        )

    print(f"\n[ALL DONE] Converted {len(task_names)} tasks to {args.output_dir}")


if __name__ == "__main__":
    main()
