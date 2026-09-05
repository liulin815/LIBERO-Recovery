"""Custom-scene evaluation for starVLA on LIBERO.

Independent counterpart to ``eval_libero_custom_bddl.py``. It evaluates the user's
own scenes (``*.bddl`` + sibling ``*_sim_state.npy``) with the following protocol
per scene:

  * ``num_close_trials`` rollouts that start with the robot gripper CLOSED,
  * ``num_open_trials`` rollouts that start with the robot gripper OPEN,
  * every rollout applies a fresh, small random xy perturbation to the free-joint
    object positions of the scene.

The original ``.bddl`` / ``.npy`` files on disk are NEVER modified: the initial
state is loaded into an in-memory copy and perturbations are applied to that copy.

This script is fully standalone and does not import or alter any other eval
script. The companion launcher is ``eval_libero_custom_scene.sh``.
"""

import dataclasses
import glob
import json
import logging
import math
import os
import pathlib
import re

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.errors import RandomizationError

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = 520
CONTACT_FORCE_THRESHOLD = 2.0


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def _gripper_env_to_dataset(gripper_env: float) -> float:
    """Convert env gripper value {+1=close, -1=open} to dataset convention {0=close, 1=open}."""
    return (1.0 - gripper_env) / 2.0


def _parse_language_from_bddl(bddl_path: str) -> str:
    with open(bddl_path, "r") as f:
        for line in f:
            m = re.search(r"\(:language\s+(.+?)\s*\)", line)
            if m:
                return m.group(1).strip()
    return os.path.splitext(os.path.basename(bddl_path))[0]


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# =====================================================================================
# Scene-perturbation helpers (independent of the rest of the codebase)
# =====================================================================================

def build_object_xy_qpos_adrs(env) -> list[tuple[str, int, int]]:
    """Return ``[(obj_name, qpos_idx_x, qpos_idx_y)]`` for every free-joint object
    that lives in ``env.env.obj_body_id``.

    Only MuJoCo *free* joints (``jnt_type == 0``) are translatable in xy; hinge /
    slide joints (e.g. a stove button, a drawer slide) are intentionally left
    untouched so we never corrupt a non-spatial DoF.
    """
    sim = env.env.sim
    model = sim.model
    obj_body_id = getattr(env.env, "obj_body_id", {})
    out = []
    for name, body_id in obj_body_id.items():
        try:
            jnt_adr = int(model.body_jntadr[body_id])
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if jnt_adr < 0:
            continue
        if int(model.jnt_type[jnt_adr]) != 0:  # 0 == free joint
            continue
        qpos_adr = int(model.jnt_qposadr[jnt_adr])
        # free-joint qpos layout: [x, y, z, qw, qx, qy, qz]
        out.append((name, qpos_adr, qpos_adr + 1))
    return out


def perturb_object_xy(flat_state: np.ndarray, obj_xy_adrs, noise_xy: float, rng: np.random.RandomState):
    """Return a copy of ``flat_state`` with a small uniform xy offset added to each
    free-joint object, plus a dict of the applied deltas ``{name: (dx, dy)}``.

    The input array is never mutated; the on-disk ``.npy`` is therefore untouched.
    """
    s = np.array(flat_state, dtype=np.float64, copy=True)
    deltas = {}
    for name, ax, ay in obj_xy_adrs:
        dx = float(rng.uniform(-noise_xy, noise_xy))
        dy = float(rng.uniform(-noise_xy, noise_xy))
        s[ax] += dx
        s[ay] += dy
        deltas[name] = (dx, dy)
    return s, deltas


def settle_gripper_to(env, close: bool, num_steps: int) -> None:
    """Drive the gripper to a fully-closed (``close=True``) or fully-open state by
    stepping the env with a zero arm-delta action.

    Direct qpos writes do NOT work for the robosuite Panda gripper because the two
    fingers are coupled by an equality constraint that restores the captured state
    on ``forward()``. Driving the gripper through its actuator for ~15-20 steps
    reliably and physically reaches the target state. The arm itself is held in
    place because OSC_POSE with a zero delta keeps the EEF goal fixed.
    """
    grip_cmd = 1.0 if close else -1.0
    action = [0.0] * 6 + [float(grip_cmd)]
    for _ in range(int(num_steps)):
        env.step(action)


def reset_controller_goal_to_current(env) -> None:
    """After restoring a sim state, point the OSC controller's goal at the current
    EEF pose so the arm does not drift on the first ``step()``.

    Mirrors the workaround already used in ``eval_libero_custom_bddl.py``.
    """
    for robot in env.robots:
        robot.controller.update()
        robot.controller.update_initial_joints(robot.controller.joint_pos)
        robot.controller.reset_goal()


def scene_tag_from_path(bddl_path: str, scene_dir: str) -> str:
    """A filesystem-safe, unique tag for a scene (the bddl path relative to
    ``scene_dir``, without the ``.bddl`` suffix, ``/`` -> ``__``)."""
    try:
        rel = os.path.relpath(bddl_path, scene_dir)
    except ValueError:
        rel = os.path.basename(bddl_path)
    if rel.endswith(".bddl"):
        rel = rel[: -len(".bddl")]
    return rel.replace(os.sep, "__").replace("/", "__")


def _construct_env(env_args, tag, seed_base, max_attempts):
    """Build the off-screen env, retrying with a reseeded global RNG on
    ``RandomizationError``.

    The placement sampler that runs inside the env's first internal reset is
    RNG-driven and, for the tight (2 cm) object regions used by these custom
    scenes, can fail to find a collision-free layout within its 5000 attempts.
    That layout is throwaway here because ``set_init_state`` overwrites it with
    the recorded sim state, so on failure we simply reseed ``np.random`` and try
    again. Returns the env, or ``None`` if every attempt failed.
    """
    for attempt in range(max_attempts):
        np.random.seed(seed_base + attempt)
        try:
            return OffScreenRenderEnv(**env_args)
        except RandomizationError as exc:
            logging.warning(
                f"[{tag}] placement failed during env build "
                f"(attempt {attempt + 1}/{max_attempts}): {exc}"
            )
    return None


def _reset_env(env, tag, trial_idx, grip_label, seed_base, max_attempts):
    """Reset the env, retrying with a reseeded global RNG on
    ``RandomizationError``. Returns True on success, False if every attempt
    failed (caller should skip the trial). Same rationale as ``_construct_env``:
    the randomized placement is overwritten by ``set_init_state``."""
    for attempt in range(max_attempts):
        np.random.seed(seed_base + attempt)
        try:
            env.reset()
            return True
        except RandomizationError as exc:
            logging.warning(
                f"[{tag}] trial{trial_idx}({grip_label}) placement failed during reset "
                f"(attempt {attempt + 1}/{max_attempts}): {exc}"
            )
    return False


# =====================================================================================
# Args + main
# =====================================================================================

@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    # Directory searched (recursively) for *.bddl scenes. Each bddl is paired with
    # its sibling ``<stem>_sim_state.npy``. May point at a whole suite folder or a
    # single task/episode folder.
    scene_dir: str = "/H20_vepfs/liulin/starVLA/assets/scenes/selected_bddl_scenes_goal"

    num_steps_wait: int = 0  # extra dummy steps at the start of each rollout (object settling)

    # ---- Custom-scene protocol ---------------------------------------------------
    # If > 0, randomly sample this many scenes from scene_dir (seeded, reproducible).
    # <= 0 means: evaluate ALL discovered scenes.
    num_scenes: int = 30
    num_close_trials: int = 5  # rollouts starting with the gripper CLOSED
    num_open_trials: int = 5   # rollouts starting with the gripper OPEN
    obj_noise_xy: float = 0.02  # max xy perturbation per object, in metres
    gripper_settle_steps: int = 20  # steps used to drive the gripper to the target state
    # Retries (each reseeds the global RNG) when the throwaway placement sampler
    # inside env build/reset raises RandomizationError. With ~50% per-attempt
    # success on tight scenes, 12 attempts => ~0.02% failure probability.
    placement_max_attempts: int = 12

    video_out_path: str = "experiments/libero/logs"
    seed: int = 7
    pretrained_path: str = ""
    unnorm_key: str | None = None
    post_process_action: bool = True
    job_name: str = "test"

    save_dataset: str = ""
    dataset_out_path: str = ""


def eval_libero_custom_scene(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)

    bddl_files = sorted(glob.glob(os.path.join(args.scene_dir, "**", "*.bddl"), recursive=True))
    # Also support a flat scene_dir that directly contains *.bddl
    bddl_files = sorted(set(bddl_files + sorted(glob.glob(os.path.join(args.scene_dir, "*.bddl")))))
    if not bddl_files:
        logging.error(f"No .bddl files found under {args.scene_dir}")
        return

    num_discovered = len(bddl_files)
    if args.num_scenes is not None and args.num_scenes > 0 and num_discovered > args.num_scenes:
        # Seeded sampling so the scene subset is reproducible across runs with the
        # same seed. A dedicated RNG is used so this does not perturb the
        # per-trial perturbation RNG sequence below.
        sample_rng = np.random.RandomState(args.seed)
        chosen = sorted(sample_rng.choice(num_discovered, size=args.num_scenes, replace=False).tolist())
        bddl_files = [bddl_files[i] for i in chosen]
        logging.info(
            f"Sampled {args.num_scenes} of {num_discovered} discovered scenes "
            f"(seed={args.seed}); the rest are skipped."
        )
    else:
        logging.info(f"Evaluating all {num_discovered} discovered scenes.")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client_model = ModelClient(host=args.host, port=args.port, unnorm_key=args.unnorm_key)

    _save_dataset = args.save_dataset.lower() in ("true", "1", "yes")
    if _save_dataset:
        dataset_out = pathlib.Path(args.dataset_out_path) if args.dataset_out_path else pathlib.Path(args.video_out_path) / "raw_dataset"
        raw_dir = dataset_out / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = raw_dir / "manifest.jsonl"
        manifest_f = open(manifest_path, "a", encoding="utf-8")
        global_episode_idx = 0
        logging.info(f"Dataset saving enabled. Output: {raw_dir}")

    # Build the per-scene trial plan once: first all CLOSE trials, then all OPEN.
    trial_plan = [("close", True)] * args.num_close_trials + [("open", False)] * args.num_open_trials
    trials_per_scene = len(trial_plan)
    if trials_per_scene == 0:
        logging.error("num_close_trials + num_open_trials == 0; nothing to do.")
        return

    total_episodes, total_successes = 0, 0
    total_close_episodes, total_close_successes = 0, 0
    total_open_episodes, total_open_successes = 0, 0
    per_scene_results = []  # for the summary file

    for scene_idx, bddl_path in enumerate(tqdm.tqdm(bddl_files, desc="Scenarios")):
        stem = bddl_path[: -len(".bddl")]
        npy_path = stem + "_sim_state.npy"
        tag = scene_tag_from_path(bddl_path, args.scene_dir)

        if not os.path.isfile(npy_path):
            logging.warning(f"Missing sim state for {bddl_path}, skipping")
            continue

        task_description = _parse_language_from_bddl(bddl_path)
        base_init_state = np.load(npy_path)

        env_args = {
            "bddl_file_name": bddl_path,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = _construct_env(
            env_args, tag, args.seed + 1_000_001 * (scene_idx + 1), args.placement_max_attempts
        )
        if env is None:
            logging.error(
                f"[{tag}] could not place objects after {args.placement_max_attempts} attempts; "
                f"skipping scene."
            )
            continue
        env.seed(args.seed)

        obj_xy_adrs = build_object_xy_qpos_adrs(env)
        logging.info(
            f"\nScenario: {tag} | Task: {task_description} | "
            f"perturbable objects: {[n for n, _, _ in obj_xy_adrs]} | trials: {trials_per_scene} "
            f"({args.num_close_trials} close / {args.num_open_trials} open)"
        )

        scene_dir_out = pathlib.Path(args.video_out_path) / tag
        scene_dir_out.mkdir(parents=True, exist_ok=True)

        scene_episodes, scene_successes = 0, 0
        scene_close_episodes, scene_close_successes = 0, 0
        scene_open_episodes, scene_open_successes = 0, 0

        for trial_idx, (grip_label, grip_close) in enumerate(trial_plan):
            client_model.reset(task_description=task_description)
            if not _reset_env(
                env, tag, trial_idx, grip_label,
                args.seed + 1_000_001 * (scene_idx + 1) + 10_001 * (trial_idx + 1),
                args.placement_max_attempts,
            ):
                logging.error(
                    f"[{tag}] trial{trial_idx}({grip_label}) could not reset env after "
                    f"{args.placement_max_attempts} attempts; skipping trial."
                )
                continue

            # Fresh in-memory copy -> perturb objects. On-disk npy is untouched.
            perturbed_state, obj_deltas = perturb_object_xy(
                base_init_state, obj_xy_adrs, args.obj_noise_xy, rng
            )
            obs = env.set_init_state(perturbed_state)
            reset_controller_goal_to_current(env)

            # Drive the gripper to the requested initial state (not recorded).
            settle_gripper_to(env, close=grip_close, num_steps=args.gripper_settle_steps)

            t = 0
            replay_images = []
            replay_wrist_images = []
            full_actions = []
            full_states = []
            first_contact_step = None
            baseline_force = None
            first_gripper_change_step = None
            prev_gripper_action = None
            step = 0
            done = False

            while t < MAX_STEPS + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                replay_images.append(img)
                replay_wrist_images.append(wrist_img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
                full_states.append(state.astype(np.float32))

                example_dict = {"image": [img, wrist_img], "lang": str(task_description)}
                response = client_model.step(example=example_dict, step=step)

                raw_action = response["raw_action"]
                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )

                delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
                full_actions.append(delta_action)

                obs, reward, done, info = env.step(delta_action.tolist())

                if first_contact_step is None:
                    ee_force = env.env.robots[0].ee_force
                    if baseline_force is None:
                        baseline_force = ee_force.copy()
                    else:
                        force_delta = np.linalg.norm(ee_force - baseline_force)
                        if force_delta > CONTACT_FORCE_THRESHOLD:
                            first_contact_step = step

                current_gripper = gripper[0]
                if first_gripper_change_step is None:
                    if prev_gripper_action is None:
                        prev_gripper_action = current_gripper
                    elif current_gripper != prev_gripper_action:
                        first_gripper_change_step = step

                if done:
                    break
                t += 1
                step += 1

            scene_episodes += 1
            total_episodes += 1
            if grip_close:
                scene_close_episodes += 1
                total_close_episodes += 1
            else:
                scene_open_episodes += 1
                total_open_episodes += 1
            if done:
                scene_successes += 1
                total_successes += 1
                if grip_close:
                    scene_close_successes += 1
                    total_close_successes += 1
                else:
                    scene_open_successes += 1
                    total_open_successes += 1

            suffix = "success" if done else "failure"
            imageio.mimwrite(
                scene_dir_out / f"rollout_trial{trial_idx}_{grip_label}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=20,
            )
            imageio.mimwrite(
                scene_dir_out / f"rollout_trial{trial_idx}_{grip_label}_{suffix}_wrist.mp4",
                [np.asarray(x) for x in replay_wrist_images],
                fps=20,
            )

            if full_actions:
                stacked_actions = np.stack(full_actions)
            if first_contact_step is not None:
                logging.info(
                    f"[{tag}] trial{trial_idx}({grip_label}) "
                    f"first_contact@{first_contact_step} (t={first_contact_step / 20.0:.2f}s)"
                )
            else:
                logging.info(f"[{tag}] trial{trial_idx}({grip_label}) no contact detected")
            if first_gripper_change_step is not None:
                logging.info(
                    f"[{tag}] trial{trial_idx}({grip_label}) "
                    f"first_grip_change@{first_gripper_change_step} (t={first_gripper_change_step / 20.0:.2f}s)"
                )

            if _save_dataset and full_actions and len(full_actions) > 0:
                actions_dataset = stacked_actions.copy()
                actions_dataset[:, 6] = np.vectorize(_gripper_env_to_dataset)(actions_dataset[:, 6])
                np.savez_compressed(
                    raw_dir / f"episode_{global_episode_idx:06d}.npz",
                    images=np.stack(replay_images),
                    wrist_images=np.stack(replay_wrist_images),
                    states=np.stack(full_states),
                    actions=actions_dataset,
                    task_description=np.array(task_description),
                    scene=np.array(tag),
                    gripper_init=np.array(grip_label),
                    obj_deltas=np.array([list(v) for v in obj_deltas.values()], dtype=np.float64)
                    if obj_deltas else np.zeros((0, 2)),
                    first_contact_step=np.array(first_contact_step if first_contact_step is not None else -1),
                    first_gripper_change_step=np.array(first_gripper_change_step if first_gripper_change_step is not None else -1),
                )
                manifest_entry = {
                    "episode_index": global_episode_idx,
                    "task": task_description,
                    "scene": tag,
                    "trial_index": trial_idx,
                    "gripper_init": grip_label,
                    "length": len(replay_images),
                    "success": bool(done),
                    "first_contact_step": first_contact_step,
                    "first_gripper_change_step": first_gripper_change_step,
                    "obj_deltas": obj_deltas,
                }
                manifest_f.write(json.dumps(manifest_entry) + "\n")
                manifest_f.flush()
                global_episode_idx += 1

            logging.info(
                f"[{tag}] trial{trial_idx}({grip_label}) Success: {done} | "
                f"scene {scene_successes}/{scene_episodes} "
                f"(close {scene_close_successes}/{scene_close_episodes}, "
                f"open {scene_open_successes}/{scene_open_episodes})"
            )

        if scene_episodes > 0:
            scene_rate = scene_successes / scene_episodes
            close_rate = (scene_close_successes / scene_close_episodes) if scene_close_episodes else float("nan")
            open_rate = (scene_open_successes / scene_open_episodes) if scene_open_episodes else float("nan")
            logging.info(
                f"Scene [{tag}] rate: {scene_rate:.4f} "
                f"(close {('nan' if math.isnan(close_rate) else f'{close_rate:.4f}')}, "
                f"open {('nan' if math.isnan(open_rate) else f'{open_rate:.4f}')})"
            )
            per_scene_results.append(
                {"scene": tag, "task": task_description, "episodes": scene_episodes,
                 "successes": scene_successes, "rate": scene_rate,
                 "close_episodes": scene_close_episodes, "close_successes": scene_close_successes,
                 "open_episodes": scene_open_episodes, "open_successes": scene_open_successes}
            )

        env.close()

    if _save_dataset:
        manifest_f.close()
        logging.info(f"Dataset saved: {global_episode_idx} episodes -> {raw_dir}")

    # Write a JSON summary next to the videos.
    summary_path = pathlib.Path(args.video_out_path) / "scene_summary.json"
    overall_rate = (total_successes / total_episodes) if total_episodes else float("nan")
    close_overall = (total_close_successes / total_close_episodes) if total_close_episodes else float("nan")
    open_overall = (total_open_successes / total_open_episodes) if total_open_episodes else float("nan")
    summary = {
        "scene_dir": args.scene_dir,
        "num_scenes_requested": args.num_scenes,
        "num_discovered": num_discovered,
        "num_evaluated": len(per_scene_results),
        "trials_per_scene": trials_per_scene,
        "num_close_trials": args.num_close_trials,
        "num_open_trials": args.num_open_trials,
        "obj_noise_xy": args.obj_noise_xy,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_rate": overall_rate,
        "close_episodes": total_close_episodes,
        "close_successes": total_close_successes,
        "close_rate": close_overall,
        "open_episodes": total_open_episodes,
        "open_successes": total_open_successes,
        "open_rate": open_overall,
        "scenes": per_scene_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info(f"Scene summary written to {summary_path}")

    # This line is grepped by the bash launcher's aggregator.
    logging.info(f"Current total success rate: {overall_rate}")
    logging.info(
        f"Total episodes: {total_episodes} | close {total_close_successes}/{total_close_episodes} "
        f"({('nan' if math.isnan(close_overall) else f'{close_overall:.4f}')}) | "
        f"open {total_open_successes}/{total_open_episodes} "
        f"({('nan' if math.isnan(open_overall) else f'{open_overall:.4f}')})"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    tyro.cli(eval_libero_custom_scene)
