import dataclasses
import json
import logging
import math
import os
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
CONTACT_FORCE_THRESHOLD = 2.0  # ee_force delta magnitude threshold (N) above baseline to detect contact


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def _gripper_env_to_dataset(gripper_env: float) -> float:
    """Convert env gripper value {+1=close, -1=open} to dataset convention {0=close, 1=open}."""
    return (1.0 - gripper_env) / 2.0


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"

    #################################################################################################################
    # Dataset saving
    #################################################################################################################
    save_dataset: str = ""  # Set to "True" or "1" to enable saving rollout data for training
    dataset_out_path: str = ""  # Directory to save raw episode data (.npz + manifest)

    save_bddl: str = ""  # Set to "True" or "1" to save scene state as .bddl every second
    bddl_save_interval: int = 20  # Steps between bddl snapshots (20 steps = 1s at 20Hz)


def _format_goal_state(goal_state: list) -> str:
    """Format parsed goal_state back to bddl string."""
    if not goal_state:
        return "()"
    parts = []
    for item in goal_state:
        if isinstance(item, list):
            parts.append(f"({' '.join(str(x) for x in item)})")
        elif isinstance(item, tuple):
            parts.append(f"({' '.join(str(x) for x in item)})")
        else:
            parts.append(str(item))
    if len(parts) == 1:
        return f"(And {parts[0]})"
    return "(And " + " ".join(parts) + ")"


def save_scene_as_bddl(env, task, episode_idx: int, step: int, out_dir: pathlib.Path):
    """Extract current MuJoCo scene state and write as a .bddl file."""
    inner_env = env.env  # BDDLBaseDomain
    parsed = inner_env.parsed_problem

    task_name = task.name if hasattr(task, "name") else task.language.replace(" ", "_")
    ep_dir = out_dir / task_name / f"episode_{episode_idx:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    skip_names = {"main_table", "kitchen_table", "floor", "countertop", "coffee_table",
                  "living_room_table", "study_table"}

    # Determine workspace (table) name
    workspace_name = "main_table"
    for name in ["kitchen_table", "main_table", "coffee_table", "living_room_table", "study_table"]:
        if name in inner_env.fixtures_dict or name in inner_env.obj_body_id:
            workspace_name = name
            break

    # Collect actual object positions and yaw from MuJoCo
    obj_states = {}
    for name, body_id in inner_env.obj_body_id.items():
        if name in skip_names:
            continue
        pos = inner_env.sim.data.body_xpos[body_id].copy()
        quat = inner_env.sim.data.body_xquat[body_id].copy()  # (w, x, y, z)
        yaw = np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))
        obj_states[name] = {"pos": pos, "yaw": float(yaw)}

    half_len = 0.01
    lines = []
    problem_name = parsed["problem_name"]
    lines.append(f"(define (problem {problem_name})")
    lines.append("  (:domain robosuite)")
    lang = parsed["language_instruction"]
    if isinstance(lang, list):
        lang = " ".join(lang)
    lines.append(f"  (:language {lang})")

    # Regions section
    lines.append("    (:regions")
    for name, state in obj_states.items():
        x, y = float(state["pos"][0]), float(state["pos"][1])
        yaw = state["yaw"]
        region_name = f"{name}_init_region"
        lines.append(f"      ({region_name}")
        lines.append(f"          (:target {workspace_name})")
        lines.append("          (:ranges (")
        lines.append(f"              ({x - half_len} {y - half_len} {x + half_len} {y + half_len})")
        lines.append("            )")
        lines.append("          )")
        if abs(yaw) > 0.01:
            lines.append("          (:yaw_rotation (")
            lines.append(f"              ({yaw} {yaw})")
            lines.append("            )")
            lines.append("          )")
        lines.append("      )")

    # Preserve affordance regions (fixture-based regions without XY ranges)
    for region_full_name, region_info in parsed["regions"].items():
        if not region_info.get("ranges"):
            target = region_info["target"]
            # Extract region suffix (e.g. "wooden_cabinet_1_top_region" -> "top_region")
            prefix = target + "_"
            if region_full_name.startswith(prefix):
                region_short = region_full_name[len(prefix):]
            else:
                region_short = region_full_name
            lines.append(f"      ({region_short}")
            lines.append(f"          (:target {target})")
            lines.append("      )")
    lines.append("    )\n")

    # Fixtures
    lines.append("  (:fixtures")
    for cat, names_list in parsed["fixtures"].items():
        for n in names_list:
            lines.append(f"    {n} - {cat}")
    lines.append("  )\n")

    # Objects
    lines.append("  (:objects")
    for cat, names_list in parsed["objects"].items():
        for n in names_list:
            lines.append(f"    {n} - {cat}")
    lines.append("  )\n")

    # Obj of interest
    lines.append("  (:obj_of_interest")
    for o in parsed["obj_of_interest"]:
        lines.append(f"    {o}")
    lines.append("  )\n")

    # Init
    lines.append("  (:init")
    for name in obj_states:
        region_name = f"{name}_init_region"
        lines.append(f"    (On {name} {workspace_name}_{region_name})")
    lines.append("  )\n")

    # Goal
    goal_str = _format_goal_state(parsed["goal_state"])
    lines.append("  (:goal")
    lines.append(f"    {goal_str}")
    lines.append("  )\n")
    lines.append(")")

    out_path = ep_dir / f"step_{step:04d}.bddl"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
    )

    # Dataset saving setup
    _save_dataset = args.save_dataset.lower() in ("true", "1", "yes")
    if _save_dataset:
        dataset_out = pathlib.Path(args.dataset_out_path) if args.dataset_out_path else pathlib.Path(args.video_out_path) / "raw_dataset"
        raw_dir = dataset_out / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = raw_dir / "manifest.jsonl"
        manifest_f = open(manifest_path, "a", encoding="utf-8")
        global_episode_idx = 0
        logging.info(f"Dataset saving enabled. Output: {raw_dir}")

    # BDDL scene saving setup
    _save_bddl = args.save_bddl.lower() in ("true", "1", "yes")
    if _save_bddl:
        bddl_out_dir = pathlib.Path(args.video_out_path) / "bddl_scenes"
        bddl_out_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"BDDL saving enabled (interval={args.bddl_save_interval} steps). Output: {bddl_out_dir}")

    # Optional smoke-test cap (still useful for quick verification with -1 = full run).
    n_eval_tasks = num_tasks_in_suite if args.max_tasks <= 0 else min(args.max_tasks, num_tasks_in_suite)
    logging.info(f"Evaluating {n_eval_tasks} of {num_tasks_in_suite} tasks (max_tasks={args.max_tasks})")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(n_eval_tasks)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            replay_wrist_images = []
            full_actions = []
            full_states = []
            first_contact_step = None
            baseline_force = None
            first_gripper_change_step = None
            prev_gripper_action = None

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Save preprocessed image for replay video
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

                observation = {  #
                    "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API --> two images provided here --> check training
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                }

                start_time = time.time()

                response = client_model.step(example=example_dict, step=step)

                end_time = time.time()
                # print(f"time: {end_time - start_time}")

                # #
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())

                # Detect first contact via EEF force sensor (delta above baseline)
                if first_contact_step is None:
                    ee_force = env.env.robots[0].ee_force
                    if baseline_force is None:
                        baseline_force = ee_force.copy()
                    else:
                        force_delta = np.linalg.norm(ee_force - baseline_force)
                        if force_delta > CONTACT_FORCE_THRESHOLD:
                            first_contact_step = step

                # Detect first gripper state change
                current_gripper = gripper[0]
                if first_gripper_change_step is None:
                    if prev_gripper_action is None:
                        prev_gripper_action = current_gripper
                    elif current_gripper != prev_gripper_action:
                        first_gripper_change_step = step

                # Save scene state as bddl at configured interval
                if _save_bddl and step % args.bddl_save_interval == 0:
                    save_scene_as_bddl(env, task, episode_idx, step, bddl_out_dir)

                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=20,
            )
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}_wrist.mp4",
                [np.asarray(x) for x in replay_wrist_images],
                fps=20,
            )

            full_actions = np.stack(full_actions)

            # Log contact info
            if first_contact_step is not None:
                first_contact_ts = first_contact_step / 20.0
                logging.info(f"First contact at step {first_contact_step} (t={first_contact_ts:.2f}s)")
            else:
                first_contact_ts = None
                logging.info("No contact detected in this episode")

            # Log gripper change info
            if first_gripper_change_step is not None:
                first_gripper_change_ts = first_gripper_change_step / 20.0
                logging.info(f"First gripper change at step {first_gripper_change_step} (t={first_gripper_change_ts:.2f}s)")
            else:
                first_gripper_change_ts = None
                logging.info("No gripper state change in this episode")

            # Save raw episode data for LeRobot conversion
            if _save_dataset:
                ep_len = len(replay_images)
                actions_dataset = full_actions.copy()
                actions_dataset[:, 6] = np.vectorize(_gripper_env_to_dataset)(actions_dataset[:, 6])

                np.savez_compressed(
                    raw_dir / f"episode_{global_episode_idx:06d}.npz",
                    images=np.stack(replay_images),
                    wrist_images=np.stack(replay_wrist_images),
                    states=np.stack(full_states),
                    actions=actions_dataset,
                    task_description=np.array(task_description),
                    first_contact_step=np.array(first_contact_step if first_contact_step is not None else -1),
                    first_gripper_change_step=np.array(first_gripper_change_step if first_gripper_change_step is not None else -1),
                )
                manifest_entry = {
                    "episode_index": global_episode_idx,
                    "task": task_description,
                    "length": ep_len,
                    "success": bool(done),
                    "first_contact_step": first_contact_step,
                    "first_contact_timestamp": first_contact_ts,
                    "first_gripper_change_step": first_gripper_change_step,
                    "first_gripper_change_timestamp": first_gripper_change_ts,
                }
                manifest_f.write(json.dumps(manifest_entry) + "\n")
                manifest_f.flush()
                global_episode_idx += 1

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    if _save_dataset:
        manifest_f.close()
        logging.info(f"Dataset saved: {global_episode_idx} episodes -> {raw_dir}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
