import math
from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation as R


def rotation_matrix_from_vector(x: float, y: float) -> np.ndarray:
    norm = np.linalg.norm([x, y])
    if norm < 1e-6:
        return np.eye(3, dtype=np.float32)
    v_x = np.asarray([x, y, 0.0], dtype=np.float32) / norm
    v_y = np.asarray([-v_x[1], v_x[0], 0.0], dtype=np.float32)
    v_y = v_y / (np.linalg.norm(v_y) + 1e-6)
    v_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return np.column_stack((v_x, v_y, v_z)).astype(np.float32)


def transform_point(point: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    return np.dot(point, rotation_matrix)


def to_eularian_angles(q):
    x, y, z, w = q
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = min(max(t2, -1.0), 1.0)
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = math.atan2(t3, t4)
    return pitch, roll, yaw


def euler_to_rotation_matrix(e):
    rotation = R.from_euler("xyz", e, degrees=False)
    return rotation.as_matrix()


def project_this_state2target_state_axis(this_state: Dict, target_state: Dict) -> Dict:
    start_pos = target_state["position"]
    start_eular = to_eularian_angles(target_state["orientation"])
    this_pos = this_state["position"]
    this_eular = to_eularian_angles(this_state["orientation"])
    delta_pos = np.asarray(this_pos) - np.asarray(start_pos)
    delta_eular = np.asarray(this_eular) - np.asarray(start_eular)
    rot = euler_to_rotation_matrix(start_eular)
    delta_pos = rot.T @ delta_pos
    return {"position": delta_pos.tolist(), "orientation": delta_eular.tolist()}

