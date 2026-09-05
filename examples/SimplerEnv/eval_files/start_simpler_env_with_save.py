"""SimplerEnv evaluation entry point with optional dataset saving in LeRobot format.

When --save-dataset is specified, evaluation data (images, actions, states) is saved
directly in LeRobot v2 format (parquet + mp4 + meta), with success and failure
episodes stored separately.

Output structure:
  <dataset-out-path>/<env_name>/
    success/   (data/, videos/, meta/)
    failure/   (data/, videos/, meta/)

Action format: delta (same as bridge_orig), shape [7]:
  [dx, dy, dz, droll, dpitch, dyaw, gripper]
"""

import os

import numpy as np
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video

from examples.SimplerEnv.eval_files.custom_argparse import get_args
from examples.SimplerEnv.eval_files.model2simpler_interface import ModelClient
from examples.SimplerEnv.eval_files.eval_data_collector import SimplerEnvDataCollector


def run_maniskill2_eval_single_episode_with_save(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
    collector=None,
):
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    is_final_subtask = env.is_final_subtask()

    if instruction is not None:
        task_description = instruction
    else:
        task_description = env.get_language_instruction()
    print(task_description)

    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    model.reset(task_description)

    if collector:
        collector.begin_episode(task_description, scene_name)
        collector.record_step(image, None, env)

    timestep = 0
    success = "failure"

    while not (predicted_terminated or truncated):
        raw_action, action = model.step(image, task_description)
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                predicted_terminated = False
                env.advance_to_next_subtask()

        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )

        success = "success" if done else "failure"
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.is_final_subtask()

        print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(image)

        if collector:
            collector.record_step(image, raw_action, env)

        timestep += 1

        if done:
            break

    if collector:
        collector.end_episode(success == "success")

    episode_stats = info.get("episode_stats", {})

    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def maniskill2_evaluator_with_save(model, args, collector=None):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                    collector=collector,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode_with_save(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(
                            run_maniskill2_eval_single_episode_with_save(
                                obj_episode_id=obj_episode_id,
                                **kwargs,
                            )
                        )
                else:
                    raise NotImplementedError()

    return success_arr


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if os.getenv("DEBUG", False):
        start_debugpy_once()

    model = ModelClient(
        policy_ckpt_path=args.ckpt_path,
        policy_setup=args.policy_setup,
        port=args.port,
        action_scale=args.action_scale,
        cfg_scale=1.5,
    )

    collector = None
    if args.save_dataset:
        out_path = args.dataset_out_path or os.path.join(args.logging_dir, "eval_dataset_lerobot")
        collector = SimplerEnvDataCollector(
            output_dir=out_path,
            env_name=args.env_name,
            fps=args.control_freq,
            image_size=tuple(args.dataset_image_size),
        )

    success_arr = maniskill2_evaluator_with_save(model, args, collector=collector)

    if collector:
        collector.close()

    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
