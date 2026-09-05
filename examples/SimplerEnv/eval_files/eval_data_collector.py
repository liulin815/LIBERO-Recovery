"""Collect per-step evaluation data and save in LeRobot dataset format.

Output structure (matching bridge_orig_1.0.0_lerobot):
  <output_dir>/
    success/
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/observation.images.image_0/episode_000000.mp4
      meta/info.json, stats.json, episodes.jsonl, tasks.jsonl
    failure/
      ... (same structure)
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np
from transforms3d.euler import quat2euler


CHUNKS_SIZE = 1000
FPS_DEFAULT = 5
VIDEO_CODEC = "libx264"
VIDEO_PIX_FMT = "yuv420p"
IMAGE_SIZE = (256, 256)


class LeRobotDatasetWriter:
    """Writes episodes directly to LeRobot v2 format (parquet + mp4 + meta)."""

    def __init__(self, output_dir: str, fps: int = FPS_DEFAULT, image_size: tuple = IMAGE_SIZE):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.image_size = image_size
        self.chunks_size = CHUNKS_SIZE

        self.data_dir = self.output_dir / "data"
        self.video_dir = self.output_dir / "videos"
        self.meta_dir = self.output_dir / "meta"

        for d in [self.data_dir, self.video_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._episode_count = self._load_existing_episode_count()
        self._global_frame_index = self._load_existing_frame_count()
        self._tasks_map: dict[str, int] = self._load_existing_tasks()

        self._all_actions: list[np.ndarray] = []
        self._all_states: list[np.ndarray] = []
        self._all_timestamps: list[float] = []

    def _load_existing_episode_count(self) -> int:
        ep_file = self.meta_dir / "episodes.jsonl"
        if not ep_file.exists():
            return 0
        count = 0
        with open(ep_file, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _load_existing_frame_count(self) -> int:
        ep_file = self.meta_dir / "episodes.jsonl"
        if not ep_file.exists():
            return 0
        total = 0
        with open(ep_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    total += entry.get("length", 0)
        return total

    def _load_existing_tasks(self) -> dict[str, int]:
        tasks_file = self.meta_dir / "tasks.jsonl"
        if not tasks_file.exists():
            return {}
        tasks = {}
        with open(tasks_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    tasks[entry["task"]] = entry["task_index"]
        return tasks

    def _get_or_create_task_index(self, task: str) -> int:
        if task in self._tasks_map:
            return self._tasks_map[task]
        idx = len(self._tasks_map)
        self._tasks_map[task] = idx
        with open(self.meta_dir / "tasks.jsonl", "a") as f:
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
        return idx

    def _get_chunk_idx(self, episode_index: int) -> int:
        return episode_index // self.chunks_size

    def write_episode(
        self,
        images: list[np.ndarray],
        actions: list[np.ndarray],
        states: list[np.ndarray],
        task_description: str,
    ):
        """Write a single episode in LeRobot format."""
        ep_idx = self._episode_count
        chunk_idx = self._get_chunk_idx(ep_idx)
        num_frames = len(actions)
        task_idx = self._get_or_create_task_index(task_description)

        chunk_data_dir = self.data_dir / f"chunk-{chunk_idx:03d}"
        chunk_data_dir.mkdir(parents=True, exist_ok=True)

        chunk_video_dir = self.video_dir / f"chunk-{chunk_idx:03d}" / "observation.images.image_0"
        chunk_video_dir.mkdir(parents=True, exist_ok=True)

        self._write_parquet(
            chunk_data_dir / f"episode_{ep_idx:06d}.parquet",
            actions, states, task_idx, ep_idx, num_frames,
        )

        self._write_video(
            chunk_video_dir / f"episode_{ep_idx:06d}.mp4",
            images[:num_frames],
        )

        for a in actions:
            self._all_actions.append(a)
        for s in states:
            self._all_states.append(s)
        for i in range(num_frames):
            self._all_timestamps.append(float(i) / self.fps)

        with open(self.meta_dir / "episodes.jsonl", "a") as f:
            entry = {
                "episode_index": ep_idx,
                "tasks": [task_description],
                "length": num_frames,
            }
            f.write(json.dumps(entry) + "\n")

        self._global_frame_index += num_frames
        self._episode_count += 1

    def _write_parquet(
        self,
        path: Path,
        actions: list[np.ndarray],
        states: list[np.ndarray],
        task_idx: int,
        episode_idx: int,
        num_frames: int,
    ):
        """Write episode data to parquet file."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = []
        for i in range(num_frames):
            row = {
                "observation.state": states[i].tolist(),
                "action": actions[i].tolist(),
                "timestamp": float(i) / self.fps,
                "frame_index": i,
                "episode_index": episode_idx,
                "index": self._global_frame_index + i,
                "task_index": task_idx,
            }
            rows.append(row)

        state_array = pa.array([r["observation.state"] for r in rows], type=pa.list_(pa.float32()))
        action_array = pa.array([r["action"] for r in rows], type=pa.list_(pa.float32()))
        timestamp_array = pa.array([r["timestamp"] for r in rows], type=pa.float32())
        frame_index_array = pa.array([r["frame_index"] for r in rows], type=pa.int64())
        episode_index_array = pa.array([r["episode_index"] for r in rows], type=pa.int64())
        index_array = pa.array([r["index"] for r in rows], type=pa.int64())
        task_index_array = pa.array([r["task_index"] for r in rows], type=pa.int64())

        table = pa.table({
            "observation.state": state_array,
            "action": action_array,
            "timestamp": timestamp_array,
            "frame_index": frame_index_array,
            "episode_index": episode_index_array,
            "index": index_array,
            "task_index": task_index_array,
        })
        pq.write_table(table, str(path))

    def _write_video(self, path: Path, images: list[np.ndarray]):
        """Write images to mp4 video using opencv."""
        h, w = self.image_size
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(str(path), fourcc, self.fps, (w, h))
        for img in images:
            resized = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
            writer.write(cv.cvtColor(resized, cv.COLOR_RGB2BGR))
        writer.release()

        temp_path = path.with_suffix(".tmp.mp4")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(path),
                    "-c:v", VIDEO_CODEC, "-pix_fmt", VIDEO_PIX_FMT,
                    "-loglevel", "error",
                    str(temp_path),
                ],
                check=True, capture_output=True,
            )
            temp_path.replace(path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            if temp_path.exists():
                temp_path.unlink()

    def finalize(self):
        """Write info.json and stats.json after all episodes are written."""
        self._write_info_json()
        self._write_stats_json()

    def _write_info_json(self):
        info = {
            "codebase_version": "v2.0",
            "robot_type": "widowx",
            "total_episodes": self._episode_count,
            "total_frames": self._global_frame_index,
            "total_tasks": len(self._tasks_map),
            "total_videos": self._episode_count,
            "total_chunks": self._get_chunk_idx(max(0, self._episode_count - 1)) + 1 if self._episode_count > 0 else 0,
            "chunks_size": self.chunks_size,
            "fps": self.fps,
            "splits": {
                "train": f"0:{self._episode_count}"
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.image_0": {
                    "dtype": "video",
                    "shape": [self.image_size[0], self.image_size[1], 3],
                    "names": ["height", "width", "rgb"],
                    "info": {
                        "video.fps": float(self.fps),
                        "video.height": self.image_size[0],
                        "video.width": self.image_size[1],
                        "video.channels": 3,
                        "video.codec": VIDEO_CODEC,
                        "video.pix_fmt": VIDEO_PIX_FMT,
                        "video.is_depth_map": False,
                        "has_audio": False,
                    }
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [8],
                    "names": {
                        "motors": ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
                    }
                },
                "action": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": {
                        "motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
                    }
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": None
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
            }
        }
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=4)

    def _write_stats_json(self):
        """Compute mean/std/max/min for all numeric features."""
        stats = {}

        if self._all_actions:
            actions_arr = np.array(self._all_actions, dtype=np.float64)
            stats["action"] = {
                "mean": actions_arr.mean(axis=0).tolist(),
                "std": actions_arr.std(axis=0).tolist(),
                "max": actions_arr.max(axis=0).tolist(),
                "min": actions_arr.min(axis=0).tolist(),
            }

        if self._all_states:
            states_arr = np.array(self._all_states, dtype=np.float64)
            stats["observation.state"] = {
                "mean": states_arr.mean(axis=0).tolist(),
                "std": states_arr.std(axis=0).tolist(),
                "max": states_arr.max(axis=0).tolist(),
                "min": states_arr.min(axis=0).tolist(),
            }

        if self._all_timestamps:
            ts_arr = np.array(self._all_timestamps, dtype=np.float64)
            stats["timestamp"] = {
                "mean": [float(ts_arr.mean())],
                "std": [float(ts_arr.std())],
                "max": [float(ts_arr.max())],
                "min": [float(ts_arr.min())],
            }

        total_frames = self._global_frame_index
        if total_frames > 0:
            indices = np.arange(total_frames, dtype=np.float64)
            stats["index"] = {
                "mean": [float(indices.mean())],
                "std": [float(indices.std())],
                "max": [float(indices.max())],
                "min": [float(indices.min())],
            }

            ep_file = self.meta_dir / "episodes.jsonl"
            frame_indices = []
            episode_indices = []
            task_indices = []
            if ep_file.exists():
                with open(ep_file, "r") as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            ep_len = entry["length"]
                            ep_idx = entry["episode_index"]
                            for fi in range(ep_len):
                                frame_indices.append(fi)
                                episode_indices.append(ep_idx)
                tasks_file = self.meta_dir / "tasks.jsonl"
                if tasks_file.exists():
                    with open(ep_file, "r") as f:
                        for line in f:
                            if line.strip():
                                entry = json.loads(line)
                                task_name = entry["tasks"][0] if entry["tasks"] else ""
                                t_idx = self._tasks_map.get(task_name, 0)
                                for _ in range(entry["length"]):
                                    task_indices.append(t_idx)

            if frame_indices:
                fi_arr = np.array(frame_indices, dtype=np.float64)
                stats["frame_index"] = {
                    "mean": [float(fi_arr.mean())],
                    "std": [float(fi_arr.std())],
                    "max": [float(fi_arr.max())],
                    "min": [float(fi_arr.min())],
                }
            if episode_indices:
                ei_arr = np.array(episode_indices, dtype=np.float64)
                stats["episode_index"] = {
                    "mean": [float(ei_arr.mean())],
                    "std": [float(ei_arr.std())],
                    "max": [float(ei_arr.max())],
                    "min": [float(ei_arr.min())],
                }
            if task_indices:
                ti_arr = np.array(task_indices, dtype=np.float64)
                stats["task_index"] = {
                    "mean": [float(ti_arr.mean())],
                    "std": [float(ti_arr.std())],
                    "max": [float(ti_arr.max())],
                    "min": [float(ti_arr.min())],
                }

        # image stats: placeholder [0,1] range since we normalize to [0,1]
        stats["observation.images.image_0"] = {
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.25]], [[0.25]], [[0.25]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "min": [[[0.0]], [[0.0]], [[0.0]]],
        }

        with open(self.meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)


class SimplerEnvDataCollector:
    """Collects eval episode data and writes to LeRobot format, splitting success/failure."""

    def __init__(
        self,
        output_dir: str,
        env_name: str,
        fps: int = FPS_DEFAULT,
        image_size: tuple = IMAGE_SIZE,
    ):
        self.fps = fps
        self.image_size = image_size
        self.env_name = env_name

        base_dir = Path(output_dir) / env_name
        self._success_writer = LeRobotDatasetWriter(
            str(base_dir / "success"), fps=fps, image_size=image_size
        )
        self._failure_writer = LeRobotDatasetWriter(
            str(base_dir / "failure"), fps=fps, image_size=image_size
        )

        self._images: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._task_description: str = ""

    def begin_episode(self, task_description: str, scene_name: str):
        self._images = []
        self._actions = []
        self._states = []
        self._task_description = task_description

    def record_step(self, image: np.ndarray, raw_action: Optional[dict], env):
        """Record one step. First step has raw_action=None (initial obs)."""
        self._images.append(image.copy())
        self._states.append(self._extract_state(env))
        if raw_action is not None:
            action = np.concatenate([
                raw_action["world_vector"].flatten(),
                raw_action["rotation_delta"].flatten(),
                raw_action["open_gripper"].flatten(),
            ]).astype(np.float32)
            self._actions.append(action)

    def end_episode(self, success: bool):
        if not self._actions:
            return

        num_actions = len(self._actions)
        images = self._images[:num_actions]
        states = self._states[:num_actions]

        writer = self._success_writer if success else self._failure_writer
        writer.write_episode(
            images=images,
            actions=self._actions,
            states=states,
            task_description=self._task_description,
        )

        self._images = []
        self._actions = []
        self._states = []

    def close(self):
        """Finalize both datasets (write info.json and stats.json)."""
        self._success_writer.finalize()
        self._failure_writer.finalize()
        print(f"[DataCollector] Dataset saved:")
        print(f"  Success episodes: {self._success_writer._episode_count}")
        print(f"  Failure episodes: {self._failure_writer._episode_count}")
        print(f"  Output: {self._success_writer.output_dir.parent}")

    @staticmethod
    def _extract_state(env) -> np.ndarray:
        """Extract 8D state [x,y,z,roll,pitch,yaw,pad,gripper] from env TCP pose."""
        tcp_pose_at_base = env.agent.robot.pose.inv() * env.tcp.pose
        pos = tcp_pose_at_base.p
        quat = tcp_pose_at_base.q
        roll, pitch, yaw = quat2euler(quat)

        qpos = env.agent.robot.get_qpos()
        qlimits = env.agent.robot.get_qlimits()
        finger_qpos = qpos[-2]
        finger_low = qlimits[-2, 0]
        finger_high = qlimits[-2, 1]
        if finger_high - finger_low > 1e-6:
            gripper_openness = float((finger_qpos - finger_low) / (finger_high - finger_low))
        else:
            gripper_openness = 0.0

        return np.array(
            [pos[0], pos[1], pos[2], roll, pitch, yaw, 0.0, gripper_openness],
            dtype=np.float32,
        )
