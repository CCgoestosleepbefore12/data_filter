"""数据 schema 声明（已知事实的单一真相源）。

集中声明 raw pika / raw teleop / processed XVLA 三类 HDF5 的字段、shape 与
关键 attrs 契约，避免在各 check 里重复硬编码 / 重新推断。

所有索引/字段来自实测 schema（见 PLAN.md §1）与 univis 适配器约定。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# raw pika / UMI （手持采集，time-synced，attr 自述 "UMI Pika raw, time-synced"）
# ---------------------------------------------------------------------------
RAW_PIKA = {
    "pose": ["observations/pose_left", "observations/pose_right"],   # 各 (T,6) 欧拉 XYZ，station frame（基站系，逐 session 漂移）
    "gripper": ["observations/gripper_left", "observations/gripper_right"],  # 各 (T,) 距离(m)
    "images": [
        "observations/images/cam_high",
        "observations/images/cam_left_wrist",
        "observations/images/cam_right_wrist",
    ],  # per-frame JPEG bytes (vlen uint8)
    "timestamps": "timestamps",   # (T,) 秒，rebased 到首帧
    # 注意：raw pika 无独立 action、无 joint、无 language
}

# ---------------------------------------------------------------------------
# raw teleop / NAS （ALOHA 风格）
# ---------------------------------------------------------------------------
RAW_TELEOP = {
    "action": "action",                 # (T,14) 命令动作（双臂 7×2）
    "base_action": "base_action",       # (T,2) 移动底盘
    "qpos": "observations/qpos",        # (T,14) 关节位置（state）
    "qvel": "observations/qvel",        # (T,14)
    "effort": "observations/effort",    # (T,14)
    "eef_6d": "observations/eef_6d",    # (T,20) [xyz3+rot6d6+grip1]×2（注意：raw eef_6d 的 rot 约定可能与 processed 不同）
    "eef_quaternion": "observations/eef_quaternion",  # (T,16)
    "language": "language_instruction",  # (1,) object
    "images": [
        "observations/images/cam_high",
        "observations/images/cam_left_wrist",
        "observations/images/cam_right_wrist",
    ],
    "eef_time": ["observations/eef_left_time", "observations/eef_right_time"],  # 各 (T,)
}

# ---------------------------------------------------------------------------
# processed XVLA （训练用，统一 20D qpos）
#   qpos 布局: [left_pose9, left_gripper, right_pose9, right_gripper]
#   pose9    : [xyz(3), rot6d(6)]，rot6d = concat(R[:,0], R[:,1])（真 R 的前两列）
#   gripper  : binary，positive=closed
# ---------------------------------------------------------------------------
PROCESSED_QPOS_KEY = "observations/qpos"   # (T,20)
QPOS_DIM = 20

# 逐维 slice（半开区间）
LEFT_XYZ = slice(0, 3)
LEFT_ROT6D = slice(3, 9)
LEFT_GRIP = 9
RIGHT_XYZ = slice(10, 13)
RIGHT_ROT6D = slice(13, 19)
RIGHT_GRIP = 19

# rot6d 契约（约定 A：构造即正交）——供 checks/rot6d.py 使用
#   判据: finite、‖a‖≈1、‖b‖≈1、a·b≈0（容差 ~1e-3）；identity = [1,0,0, 0,1,0]
#   注: det([a,b,a×b]) = ‖a×b‖² ≥ 0 恒非负，无法判手性/镜像，故不作判据。
IDENTITY_ROT6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

# 各来源 processed 应满足的 attrs 契约
PROCESSED_ATTRS = {
    "pika_umi": {
        "pose_frame": "robot_base_tip2base_piper_tcp_config",
        "tip2base_applied": True,
        "relative_to_first_frame": False,
        "domain_name": {
            "pika_umi_tip2base_abs",
            "pika_extra_tip2base_abs",
            "pika_camera_wrong_tip2base_abs",
        },
        "time_alignment_status": "verified_common_time_axis",
    },
    "nas_teleop": {
        "domain_name": "nas_real_teleop",
        "source_key": "observations/eef_6d",
        "time_alignment_status": "verified_common_time_axis",
    },
}
