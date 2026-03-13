#!/usr/bin/env python
import time
from pathlib import Path
from typing import Optional

import joblib
import mujoco
import numpy as np
from mujoco import viewer


def load_motion_qpos(
    motion_file: Path,
    motion_name: Optional[str] = None,
    field: Optional[str] = None,
):
    data = joblib.load(motion_file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict from {motion_file}, got {type(data)}")

    motion_keys = list(data.keys())
    if len(motion_keys) == 0:
        raise ValueError(f"No motions found in {motion_file}")

    if motion_name is None:
        motion_name = motion_keys[0]
        print(f"[info] No motion_name specified, using first one: {motion_name}")
    elif motion_name not in data:
        raise KeyError(f"motion_name={motion_name} not found. Available keys (first 10): {motion_keys[:10]}")

    motion = data[motion_name]
    if not isinstance(motion, dict):
        raise TypeError(f"Expected motion entry to be dict, got {type(motion)}")

    print(f"[info] Selected motion: {motion_name}")
    print(f"[info] Available fields in this motion: {list(motion.keys())}")

    candidate_fields = []
    for k, v in motion.items():
        if hasattr(v, "shape"):
            candidate_fields.append((k, np.asarray(v).shape))
    print("[info] Tensor-like fields and shapes in this motion:")
    for name, shape in candidate_fields:
        print(f"  - {name}: {shape}")

    qpos_array = None
    if field is not None:
        if field not in motion:
            raise KeyError(f"Specified field '{field}' not found in motion. Available: {list(motion.keys())}")
        qpos_array = np.asarray(motion[field])
        print(f"[info] Using user-specified field '{field}' as qpos, shape={qpos_array.shape}")
    else:
        # 优先处理类似 lafan/adamsp 的结构：root_trans_offset(平移3) + root_rot(四元数4) + dof(关节29)
        if all(k in motion for k in ["root_trans_offset", "root_rot", "dof"]):
            root_trans = np.asarray(motion["root_trans_offset"])
            root_rot_xyzw = np.asarray(motion["root_rot"])
            dof = np.asarray(motion["dof"])
            if root_trans.shape[0] != root_rot_xyzw.shape[0] or root_trans.shape[0] != dof.shape[0]:
                raise ValueError(
                    f"Inconsistent time length among root_trans_offset {root_trans.shape}, "
                    f"root_rot {root_rot_xyzw.shape}, dof {dof.shape}"
                )
            if root_trans.shape[1] != 3 or root_rot_xyzw.shape[1] != 4:
                raise ValueError(
                    f"Expected root_trans_offset shape (T,3) and root_rot shape (T,4), "
                    f"got {root_trans.shape} and {root_rot_xyzw.shape}"
                )
            # motion_lib 中 root_rot 在很多地方以 w_last=True 使用，通常为 (x, y, z, w)（xyzw）。
            # MuJoCo 期望 qpos 中的四元数为 (w, x, y, z)（wxyz），需要转换一下顺序。
            root_rot = np.stack(
                [
                    root_rot_xyzw[:, 3],  # w
                    root_rot_xyzw[:, 0],  # x
                    root_rot_xyzw[:, 1],  # y
                    root_rot_xyzw[:, 2],  # z
                ],
                axis=1,
            )
            qpos_array = np.concatenate([root_trans, root_rot, dof], axis=1)
            print(f"[info] Auto-constructed qpos from [root_trans_offset, root_rot, dof], shape={qpos_array.shape}")
        else:
            preferred_order = ["qpos", "pose_qpos", "qpos_global"]
            for cand in preferred_order:
                if cand in motion:
                    qpos_array = np.asarray(motion[cand])
                    print(f"[info] Auto-selected field '{cand}' as qpos, shape={qpos_array.shape}")
                    break

    if qpos_array is None:
        raise RuntimeError(
            "Could not automatically determine which field to treat as qpos.\n"
            "Please re-run with --field=<name> where <name> is one of the tensor fields printed above."
        )

    if qpos_array.ndim != 2:
        raise ValueError(f"Expected qpos of shape (T, nq), got shape {qpos_array.shape}")

    return motion_name, qpos_array


def play_motion_with_mujoco(
    model_xml: Path,
    motion_file: Path,
    motion_name: Optional[str] = None,
    field: Optional[str] = None,
    fps: float = 30.0,
):
    print(f"[info] Loading MuJoCo model from: {model_xml}")
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)

    print(f"[info] model.nq = {model.nq}, model.nv = {model.nv}, model.njnt = {model.njnt}")
    print("[info] Model joint names (mjOBJ_JOINT):")
    for j in range(model.njnt):
        print(j, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))

    # 预先打印该数据文件中包含的所有 motion 名称，方便浏览
    data_dict_for_log = joblib.load(motion_file)
    motion_keys_for_log = list(data_dict_for_log.keys())
    print(f"[info] Motion file: {motion_file}")
    print(f"[info] Total motions: {len(motion_keys_for_log)}")
    if len(motion_keys_for_log) > 0:
        print("[info] Motion names:")
        for i, name in enumerate(motion_keys_for_log):
            print(f"  [{i}] {name}")
    else:
        print("[info] Motion file contains no motions.")

    dt = 1.0 / fps
    print(f"[info] Starting viewer at ~{fps} FPS. Close the window to stop.")

    # 情况 1：指定了 motion_name，只播放这一条
    if motion_name is not None:
        motion_name, qpos_seq = load_motion_qpos(motion_file, motion_name=motion_name, field=field)
        T, nq = qpos_seq.shape
        print(f"[info] Loaded motion '{motion_name}' with T={T}, nq={nq}")

        if nq != model.nq:
            raise ValueError(
                f"Motion nq ({nq}) does not match model.nq ({model.nq}).\n"
                "If your data only contains joint angles (e.g. 29 DOF) while the model is free-base (e.g. 36 DOF),\n"
                "you need to extend qpos to include root pose before using this script."
            )

        with viewer.launch_passive(model, data) as v:
            frame = 0
            t_last = time.time()
            while v.is_running():
                now = time.time()
                if now - t_last < dt:
                    time.sleep(0.001)
                    continue
                t_last = now

                qpos = qpos_seq[frame % T]
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                v.sync()
                frame += 1

        return

    # 情况 2：未指定 motion_name，顺序循环播放所有 motions
    data_dict = joblib.load(motion_file)
    motion_keys = list(data_dict.keys())
    if len(motion_keys) == 0:
        raise ValueError(f"No motions found in {motion_file}")

    print(f"[info] No motion_name specified. Will loop over all {len(motion_keys)} motions.")

    with viewer.launch_passive(model, data) as v:
        motion_idx = 0
        qpos_seq = None
        T = 0
        frame = 0
        t_last = time.time()

        while v.is_running():
            now = time.time()
            if now - t_last < dt:
                time.sleep(0.001)
                continue
            t_last = now

            # 如果当前 motion 播放结束或还没有加载任何 motion，则切换到下一条
            if qpos_seq is None or frame >= T:
                current_name = motion_keys[motion_idx]
                current_name, qpos_seq = load_motion_qpos(motion_file, motion_name=current_name, field=field)
                T, nq = qpos_seq.shape
                print(f"[info] >>> Now playing motion '{current_name}' (T={T}, nq={nq})")

                if nq != model.nq:
                    raise ValueError(
                        f"Motion '{current_name}' nq ({nq}) does not match model.nq ({model.nq}).\n"
                        "If your data only contains joint angles (e.g. 29 DOF) while the model is free-base (e.g. 36 DOF),\n"
                        "you need to extend qpos to include root pose before using this script."
                    )

                frame = 0
                motion_idx = (motion_idx + 1) % len(motion_keys)

            qpos = qpos_seq[frame]
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            v.sync()
            frame += 1


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Visualize motion from a .pkl file (e.g. adamsp_29dof_lafan1.pkl) "
            "on a specified MuJoCo model."
        )
    )
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--robot",
        type=str,
        default="adamsp/adamsp_29dof",
        help=(
            "Robot name (e.g. 'adamsp/adamsp_29dof', 'g1/g1_29dof_new'). "
            "Used only to choose sensible defaults for model XML and motion file."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to motion .pkl file. If not set, a default is chosen based on --robot.",
    )
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=None,
        help="Path to MuJoCo XML model to visualize with. If not set, chosen based on --robot.",
    )
    parser.add_argument(
        "--motion-name",
        type=str,
        default=None,
        help="Key of the motion sequence inside the .pkl. Default: first key.",
    )
    parser.add_argument(
        "--field",
        type=str,
        default=None,
        help=(
            "Name of the field in the motion dict to use as qpos. "
            "If not set, the script will try 'qpos', 'pose_qpos', 'qpos_global' in this order."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback FPS for visualization.",
    )
    args = parser.parse_args()

    # 根据 robot 选择默认的模型和数据路径
    robot_str = args.robot or ""
    robot_lower = robot_str.lower()

    if "adamsp" in robot_lower:
        default_motion = default_root / "data" / "adamsp_29dof_lafan1.pkl"
        default_model = default_root / "data" / "robots" / "adamsp" / "scene_29dof_freebase_mujoco.xml"
    elif "g1" in robot_lower:
        # 对 G1 使用现有的 lafan 数据和场景 XML（如需精细区分，可再根据 robot_str 细化）
        default_motion = default_root / "data" / "g1_lafan_29dof_10s-clipped.pkl"
        default_model = default_root / "data" / "robots" / "g1" / "scene_29dof_freebase_mujoco.xml"
    else:
        # 回退到 adamsp 默认
        default_motion = default_root / "data" / "adamsp_29dof_lafan1.pkl"
        default_model = default_root / "data" / "robots" / "adamsp" / "scene_29dof_freebase_mujoco.xml"

    motion_file = args.data_path if args.data_path is not None else default_motion
    model_xml = args.model_xml if args.model_xml is not None else default_model

    if not motion_file.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    if not model_xml.exists():
        raise FileNotFoundError(f"Model XML not found: {model_xml}")

    play_motion_with_mujoco(
        model_xml=model_xml,
        motion_file=motion_file,
        motion_name=args.motion_name,
        field=args.field,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

