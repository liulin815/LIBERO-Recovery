"""
Evaluation data saver: records trajectory data during RoboTwin evaluation
and saves in RoboTwin2.0 native dataset format (identical to
datasets/RoboTwin2.0/dataset/<task>/aloha-agilex_<config>_<N>/).

Output HDF5 structure (per episode):
    endpose/left_endpose          (T, 7)
    endpose/left_gripper          (T,)
    endpose/right_endpose         (T, 7)
    endpose/right_gripper         (T,)
    joint_action/left_arm         (T, 6)
    joint_action/left_gripper     (T,)
    joint_action/right_arm        (T, 6)
    joint_action/right_gripper    (T,)
    joint_action/vector           (T, 14)
    observation/<cam>/rgb         (T,) JPEG bytes S<maxlen>
    observation/<cam>/intrinsic_cv   (T, 3, 3)
    observation/<cam>/extrinsic_cv   (T, 3, 4)
    observation/<cam>/cam2world_gl   (T, 4, 4)
    pointcloud                    (T, 0)

Output directory structure:
    {save_dir}/{task_name}/aloha-agilex_{task_config}_{num_episodes}/
    ├── data/episode0.hdf5
    ├── video/episode0.mp4
    ├── instructions/episode0.json
    ├── scene_info.json
    └── seed.txt
"""

import os
import json
import subprocess

import cv2
import h5py
import numpy as np


def images_encoding(imgs):
    """Encode images as JPEG bytes and pad to uniform length (matches pkl2hdf5.py)."""
    encode_data = []
    max_len = 0
    for img in imgs:
        success, encoded_image = cv2.imencode(".jpg", img)
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    padded_data = [d.ljust(max_len, b"\0") for d in encode_data]
    return padded_data, max_len


def images_to_video(imgs_rgb, out_path, fps=30.0):
    """Generate MP4 video from RGB frames."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if len(imgs_rgb.shape) != 4:
        return
    n_frames, H, W, C = imgs_rgb.shape
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{W}x{H}",
            "-framerate", str(fps),
            "-i", "-",
            "-pix_fmt", "yuv420p",
            "-vcodec", "libx264",
            "-crf", "23",
            out_path,
        ],
        stdin=subprocess.PIPE,
    )
    ffmpeg.stdin.write(imgs_rgb.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()


class EvalDataSaver:
    """Records evaluation trajectories and saves in RoboTwin2.0 dataset format.

    Each episode is written to disk immediately upon end_episode(), so data is
    preserved even if the evaluation is interrupted mid-run.
    """

    def __init__(self, save_dir, task_name, task_config, ckpt_setting, save_mode="all"):
        self.save_dir = save_dir
        self.task_name = task_name
        self.task_config = task_config
        self.ckpt_setting = ckpt_setting
        self.save_mode = save_mode

        self.episodes_meta = []
        self.current_episode = None
        self.saved_count = 0

    def start_episode(self, episode_idx, seed, instruction, episode_info=None):
        """Begin recording a new episode."""
        self.current_episode = {
            "episode_idx": episode_idx,
            "seed": seed,
            "instruction": instruction,
            "episode_info": episode_info,
            "steps": [],
        }

    def record_step(self, observation, action):
        """Record a single timestep.

        Args:
            observation: dict from TASK_ENV.get_obs()
            action: 14-dim action vector (after reindexing)
        """
        if self.current_episode is None:
            return

        obs_cameras = {}
        for cam_name in ["head_camera", "left_camera", "right_camera", "front_camera"]:
            if cam_name in observation["observation"]:
                cam_data = observation["observation"][cam_name]
                obs_cameras[cam_name] = {
                    "rgb": cam_data["rgb"].copy(),
                    "intrinsic_cv": cam_data["intrinsic_cv"].copy(),
                    "extrinsic_cv": cam_data["extrinsic_cv"].copy(),
                    "cam2world_gl": cam_data["cam2world_gl"].copy(),
                }

        step_data = {
            "observation": obs_cameras,
            "joint_action": {
                "left_arm": np.array(observation["joint_action"]["left_arm"]).copy(),
                "left_gripper": float(observation["joint_action"]["left_gripper"]),
                "right_arm": np.array(observation["joint_action"]["right_arm"]).copy(),
                "right_gripper": float(observation["joint_action"]["right_gripper"]),
                "vector": np.array(observation["joint_action"]["vector"]).copy(),
            },
            "endpose": {
                "left_endpose": np.array(observation["endpose"]["left_endpose"]).copy(),
                "left_gripper": float(observation["endpose"]["left_gripper"]),
                "right_endpose": np.array(observation["endpose"]["right_endpose"]).copy(),
                "right_gripper": float(observation["endpose"]["right_gripper"]),
            },
            "action": np.array(action).copy(),
        }
        self.current_episode["steps"].append(step_data)

    def _get_output_dir(self):
        """Get the output directory path (created lazily on first write)."""
        return os.path.join(self.save_dir, self.task_name, f"aloha-agilex_{self.task_config}")

    def end_episode(self, success):
        """End the current episode. If save_mode matches, write to disk immediately.

        The success field is always recorded in metadata/episodeN.json regardless
        of save_mode, so users can re-define success criteria during data conversion.
        """
        if self.current_episode is None:
            return

        should_save = (
            self.save_mode == "all"
            or (self.save_mode == "success" and success)
            or (self.save_mode == "fail" and not success)
        )

        if should_save and len(self.current_episode["steps"]) > 0:
            self.current_episode["success"] = success
            ep_idx = self.saved_count

            output_dir = self._get_output_dir()
            data_dir = os.path.join(output_dir, "data")
            video_dir = os.path.join(output_dir, "video")
            instructions_dir = os.path.join(output_dir, "instructions")
            metadata_dir = os.path.join(output_dir, "metadata")
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(instructions_dir, exist_ok=True)
            os.makedirs(metadata_dir, exist_ok=True)

            self._save_episode_hdf5(self.current_episode, ep_idx, data_dir)
            self._save_episode_video(self.current_episode, ep_idx, video_dir)
            self._save_episode_instructions(self.current_episode, ep_idx, instructions_dir)
            self._save_episode_metadata(self.current_episode, ep_idx, metadata_dir, success)

            self.episodes_meta.append({
                "seed": self.current_episode["seed"],
                "episode_info": self.current_episode.get("episode_info"),
                "success": success,
            })
            self.saved_count += 1
            print(f"[EvalDataSaver] Episode {ep_idx} saved to {data_dir}/episode{ep_idx}.hdf5 (success={success})")

        self.current_episode = None

    def save_all(self):
        """Finalize: write scene_info.json and seed.txt for all saved episodes."""
        if self.saved_count == 0:
            print("[EvalDataSaver] No episodes were saved.")
            return

        output_dir = self._get_output_dir()
        self._save_scene_info(output_dir)
        self._save_seeds(output_dir)

        final_dir = os.path.join(
            self.save_dir,
            self.task_name,
            f"aloha-agilex_{self.task_config}_{self.saved_count}",
        )
        if output_dir != final_dir:
            if os.path.exists(final_dir):
                import shutil
                shutil.rmtree(final_dir)
            os.rename(output_dir, final_dir)

        print(f"[EvalDataSaver] Done. {self.saved_count} episodes saved to {final_dir}")

    def _save_episode_metadata(self, episode, ep_idx, metadata_dir, success):
        """Save per-episode metadata JSON with success flag and episode info.

        This file is meant to be read during data conversion so users can
        re-define success/failure criteria without re-running evaluation.
        """
        meta_path = os.path.join(metadata_dir, f"episode{ep_idx}.json")
        meta = {
            "episode_idx": ep_idx,
            "seed": episode["seed"],
            "instruction": episode["instruction"],
            "success": success,
            "num_steps": len(episode["steps"]),
            "task_name": self.task_name,
            "task_config": self.task_config,
            "ckpt_setting": self.ckpt_setting,
        }
        if episode.get("episode_info") is not None:
            meta["episode_info"] = episode["episode_info"]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _save_episode_hdf5(self, episode, ep_idx, data_dir):
        """Save a single episode as HDF5 in RoboTwin2.0 format."""
        steps = episode["steps"]
        T = len(steps)
        hdf5_path = os.path.join(data_dir, f"episode{ep_idx}.hdf5")

        camera_names = list(steps[0]["observation"].keys())

        # Encode images per camera (pass RGB directly to imencode, matching
        # the original pkl2hdf5.py behavior which does NOT convert to BGR)
        cam_encoded = {}
        for cam_name in camera_names:
            imgs = [s["observation"][cam_name]["rgb"] for s in steps]
            encoded, max_len = images_encoding(imgs)
            cam_encoded[cam_name] = (encoded, max_len)

        with h5py.File(hdf5_path, "w") as f:
            # endpose
            endpose_grp = f.create_group("endpose")
            endpose_grp.create_dataset(
                "left_endpose",
                data=np.array([s["endpose"]["left_endpose"] for s in steps]),
            )
            endpose_grp.create_dataset(
                "left_gripper",
                data=np.array([s["endpose"]["left_gripper"] for s in steps]),
            )
            endpose_grp.create_dataset(
                "right_endpose",
                data=np.array([s["endpose"]["right_endpose"] for s in steps]),
            )
            endpose_grp.create_dataset(
                "right_gripper",
                data=np.array([s["endpose"]["right_gripper"] for s in steps]),
            )

            # joint_action
            ja_grp = f.create_group("joint_action")
            ja_grp.create_dataset(
                "left_arm",
                data=np.array([s["joint_action"]["left_arm"] for s in steps]),
            )
            ja_grp.create_dataset(
                "left_gripper",
                data=np.array([s["joint_action"]["left_gripper"] for s in steps]),
            )
            ja_grp.create_dataset(
                "right_arm",
                data=np.array([s["joint_action"]["right_arm"] for s in steps]),
            )
            ja_grp.create_dataset(
                "right_gripper",
                data=np.array([s["joint_action"]["right_gripper"] for s in steps]),
            )
            ja_grp.create_dataset(
                "vector",
                data=np.array([s["joint_action"]["vector"] for s in steps]),
            )

            # observation cameras
            obs_grp = f.create_group("observation")
            for cam_name in camera_names:
                encoded, max_len = cam_encoded[cam_name]
                cam_grp = obs_grp.create_group(cam_name)
                cam_grp.create_dataset("rgb", data=encoded, dtype=f"S{max_len}")
                cam_grp.create_dataset(
                    "intrinsic_cv",
                    data=np.array([s["observation"][cam_name]["intrinsic_cv"] for s in steps]),
                )
                cam_grp.create_dataset(
                    "extrinsic_cv",
                    data=np.array([s["observation"][cam_name]["extrinsic_cv"] for s in steps]),
                )
                cam_grp.create_dataset(
                    "cam2world_gl",
                    data=np.array([s["observation"][cam_name]["cam2world_gl"] for s in steps]),
                )

            # action (model-predicted 14-D action vector)
            actions = [s.get("action") for s in steps]
            if actions[0] is not None:
                f.create_dataset("action", data=np.array(actions, dtype=np.float64))

            # pointcloud (empty, for format alignment)
            f.create_dataset("pointcloud", data=np.empty((T, 0), dtype=np.float64))

    def _save_episode_video(self, episode, ep_idx, video_dir):
        """Generate MP4 from head camera RGB frames."""
        steps = episode["steps"]
        if "head_camera" not in steps[0]["observation"]:
            return
        frames = np.array([s["observation"]["head_camera"]["rgb"] for s in steps])
        video_path = os.path.join(video_dir, f"episode{ep_idx}.mp4")
        images_to_video(frames, video_path, fps=30.0)

    def _save_episode_instructions(self, episode, ep_idx, instructions_dir):
        """Save instruction as JSON with seen/unseen format (matching RoboTwin2.0)."""
        instruction = episode["instruction"]
        instr_path = os.path.join(instructions_dir, f"episode{ep_idx}.json")
        data = {
            "seen": [instruction],
            "unseen": [],
        }
        with open(instr_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_scene_info(self, output_dir):
        """Save scene_info.json matching RoboTwin2.0 format."""
        scene_info_path = os.path.join(output_dir, "scene_info.json")
        scene_info = {}
        for ep_idx, meta in enumerate(self.episodes_meta):
            ep_info = {
                "cluttered_table_info": [],
                "texture_info": {"wall_texture": None, "table_texture": None},
            }
            if meta.get("episode_info") is not None:
                ep_info["info"] = meta["episode_info"]
            else:
                ep_info["info"] = {}
            scene_info[f"episode_{ep_idx}"] = ep_info
        with open(scene_info_path, "w", encoding="utf-8") as f:
            json.dump(scene_info, f, ensure_ascii=False, indent=2)

    def _save_seeds(self, output_dir):
        """Save seed.txt (space-separated seeds on one line, matching RoboTwin2.0)."""
        seed_path = os.path.join(output_dir, "seed.txt")
        seeds = [str(meta["seed"]) for meta in self.episodes_meta]
        with open(seed_path, "w") as f:
            f.write(" ".join(seeds))
