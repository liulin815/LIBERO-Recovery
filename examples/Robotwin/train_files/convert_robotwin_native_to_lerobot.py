"""
Convert RoboTwin2.0 native HDF5 dataset to LeRobot v2 format for starVLA training.

Reads from:
    {input_dir}/data/episode*.hdf5
    {input_dir}/instructions/episode*.json

Produces a LeRobot dataset at HF_LEROBOT_HOME/{repo_id}/ with:
    observation.state (14-D from joint_action/vector)
    action (14-D from joint_action/vector shifted by 1 step)
    observation.images.cam_high (from observation/head_camera/rgb)
    observation.images.cam_left_wrist (from observation/left_camera/rgb)
    observation.images.cam_right_wrist (from observation/right_camera/rgb)
    task (instruction text)

Usage:
    python convert_robotwin_native_to_lerobot.py \
        --input-dir /path/to/aloha-agilex_clean_50 \
        --repo-id my_eval_data/adjust_bottle \
        --mode image
"""

import dataclasses
import json
import os
import shutil
from glob import glob
from pathlib import Path
from typing import Literal

import cv2
import h5py
import numpy as np
import torch
import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


CAMERA_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist", "left_shoulder", "left_elbow",
        "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper",
        "right_waist", "right_shoulder", "right_elbow",
        "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    ]
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=15,
        robot_type="aloha",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def decode_jpeg_images(jpeg_bytes_array):
    """Decode JPEG byte strings from HDF5 to numpy RGB arrays."""
    imgs = []
    for data in jpeg_bytes_array:
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        imgs.append(img)  # BGR from cv2
    return np.array(imgs)


def load_instruction(instructions_dir, ep_idx):
    """Load instruction from per-episode JSON file."""
    json_path = os.path.join(instructions_dir, f"episode{ep_idx}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    candidates = data.get("seen", []) + data.get("unseen", [])
    if candidates:
        return np.random.choice(candidates)
    return None


def convert(
    input_dir: Path,
    repo_id: str,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Convert RoboTwin2.0 native format to LeRobot."""
    data_dir = input_dir / "data"
    instructions_dir = input_dir / "instructions"

    hdf5_files = sorted(glob(str(data_dir / "episode*.hdf5")),
                        key=lambda x: int(os.path.basename(x).replace("episode", "").replace(".hdf5", "")))

    if not hdf5_files:
        raise FileNotFoundError(f"No episode HDF5 files found in {data_dir}")

    print(f"Found {len(hdf5_files)} episodes in {data_dir}")

    dataset = create_empty_dataset(repo_id, mode=mode, dataset_config=dataset_config)

    for ep_idx, hdf5_path in enumerate(tqdm.tqdm(hdf5_files, desc="Converting")):
        with h5py.File(hdf5_path, "r") as f:
            # State: joint_action/vector (T, 14)
            state = torch.from_numpy(f["joint_action/vector"][:].astype(np.float32))

            # Action: prefer explicit /action field (eval data saves model predictions);
            # fall back to next-step state (expert demo convention)
            if "action" in f:
                action = torch.from_numpy(f["action"][:].astype(np.float32))
            else:
                action = torch.zeros_like(state)
                action[:-1] = state[1:]
                action[-1] = state[-1]

            # Load images from cameras
            imgs_per_cam = {}
            for native_name, lerobot_name in CAMERA_MAP.items():
                cam_path = f"observation/{native_name}/rgb"
                if cam_path in f:
                    imgs_per_cam[lerobot_name] = decode_jpeg_images(f[cam_path][:])

            num_frames = state.shape[0]

        # Load instruction
        instruction = load_instruction(str(instructions_dir), ep_idx)
        if instruction is None:
            instruction = "perform the task"

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }
            for cam_name, img_array in imgs_per_cam.items():
                frame[f"observation.images.{cam_name}"] = img_array[i]

            dataset.add_frame(frame, task=instruction)

        dataset.save_episode()

    print(f"Dataset saved to {HF_LEROBOT_HOME / repo_id}")
    print(f"Total episodes: {len(hdf5_files)}, total frames: {len(dataset)}")


if __name__ == "__main__":
    tyro.cli(convert)
