import json
import socket
import threading
from configparser import ConfigParser
from typing import Dict, Optional, Tuple

import numpy as np
from stable_baselines3 import SAC


# ---- fixed config ----

CONFIG_PATH = "/home/data/zhuyanxiang/UAVDRL/UAV_Navigation_DRL_AirSim/logs_save/test/config/config.ini"
MODEL_PATH = "/home/data/zhuyanxiang/UAVDRL/UAV_Navigation_DRL_AirSim/logs_save/test/models/model_sb3.zip"
HOST = "0.0.0.0"
PORT = 50052

NAVIGATION_3D = False
USING_VELOCITY_STATE = False
STATE_FEATURE_LENGTH = 2


# ---- utilities ----

def _load_model(model_path: str):
    return SAC.load(model_path, device="cpu")


def _normalize_multirotor_state(
    state: Dict[str, float],
    goal_distance: float,
    v_xy_min: float,
    v_xy_max: float,
    v_z_max: float,
    yaw_rate_max_rad: float,
    navigation_3d: bool,
    using_velocity_state: bool,
    max_vertical_difference: float = 5.0,
) -> np.ndarray:
    """
    Expect angles in radians: dyaw (relative yaw), yaw_rate.
    Returns state_norm in 0-255 order as in MultirotorDynamicsAirsim.
    """
    dxy = state.get("dxy")
    dz = state.get("dz", 0.0)
    dyaw = state.get("dyaw")

    if dxy is None or dyaw is None or goal_distance is None:
        raise ValueError("Missing dxy/dyaw/goal_distance for multirotor normalization")

    distance_norm = dxy / goal_distance * 255.0
    vertical_distance_norm = (dz / max_vertical_difference / 2.0 + 0.5) * 255.0
    relative_yaw_norm = (dyaw / np.pi / 2.0 + 0.5) * 255.0

    vxy = state.get("vxy", 0.0)
    vz = state.get("vz", 0.0)
    yaw_rate = state.get("yaw_rate", 0.0)

    linear_velocity_norm = (vxy - v_xy_min) / (v_xy_max - v_xy_min) * 255.0
    linear_velocity_z_norm = (vz / v_z_max / 2.0 + 0.5) * 255.0
    angular_velocity_norm = (yaw_rate / yaw_rate_max_rad / 2.0 + 0.5) * 255.0

    state_norm = np.array(
        [
            distance_norm,
            vertical_distance_norm,
            relative_yaw_norm,
            linear_velocity_norm,
            linear_velocity_z_norm,
            angular_velocity_norm,
        ],
        dtype=np.float32,
    )

    if navigation_3d:
        if not using_velocity_state:
            state_norm = state_norm[:3]
    else:
        state_norm = np.array([state_norm[0], state_norm[2], state_norm[3], state_norm[5]], dtype=np.float32)
        if not using_velocity_state:
            state_norm = state_norm[:2]

    return np.clip(state_norm, 0, 255)


def _build_obs_image(
    state_feature: np.ndarray,
    height: int,
    width: int,
    depth_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    if depth_image is None:
        depth_image = np.zeros((height, width), dtype=np.uint8)
    if depth_image.shape != (height, width):
        raise ValueError(f"depth_image shape must be {(height, width)}")

    state_feature_array = np.zeros((height, width), dtype=np.float32)
    state_feature_array[0, 0:state_feature.shape[0]] = state_feature

    image_with_state = np.array([depth_image, state_feature_array], dtype=np.float32)
    image_with_state = image_with_state.swapaxes(0, 2)
    image_with_state = image_with_state.swapaxes(0, 1)
    return image_with_state


# ---- server ----

class RemoteInferServer:
    def __init__(self, cfg_path: str, model_path: str, host: str, port: int) -> None:
        self.cfg = ConfigParser()
        self.cfg.read(cfg_path)

        self.screen_height = self.cfg.getint("environment", "screen_height")
        self.screen_width = self.cfg.getint("environment", "screen_width")
        self.state_feature_length = STATE_FEATURE_LENGTH

        self.model = _load_model(model_path)

        self.v_xy_min = self.cfg.getfloat("multirotor", "v_xy_min")
        self.v_xy_max = self.cfg.getfloat("multirotor", "v_xy_max")
        self.v_z_max = self.cfg.getfloat("multirotor", "v_z_max")
        self.yaw_rate_max_rad = np.deg2rad(self.cfg.getfloat("multirotor", "yaw_rate_max_deg"))

        self.host = host
        self.port = port

    def _make_state_feature(self, payload: Dict) -> np.ndarray:
        state = payload.get("state", {})
        goal_distance = payload.get("goal_distance")

        return _normalize_multirotor_state(
            state,
            goal_distance,
            self.v_xy_min,
            self.v_xy_max,
            self.v_z_max,
            self.yaw_rate_max_rad,
            NAVIGATION_3D,
            USING_VELOCITY_STATE,
        )

    def _infer_action(self, payload: Dict) -> Dict:
        state_feature = self._make_state_feature(payload)

        depth = payload.get("depth_image")
        if depth is not None:
            depth = np.asarray(depth, dtype=np.uint8)
        obs = _build_obs_image(state_feature, self.screen_height, self.screen_width, depth)

        action, _ = self.model.predict(obs, deterministic=True)
        return {
            "action": action.tolist() if isinstance(action, np.ndarray) else list(action),
            "state_feature_used": state_feature.tolist(),
        }

    def _handle_client(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        with conn:
            buffer = ""
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        result = self._infer_action(payload)
                        resp = {"ok": True, **result}
                    except Exception as e:
                        resp = {"ok": False, "error": str(e)}
                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(5)
            print(f"RemoteInferServer listening on {self.host}:{self.port}")
            while True:
                conn, addr = s.accept()
                t = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
                t.start()
def main() -> None:
    server = RemoteInferServer(CONFIG_PATH, MODEL_PATH, HOST, PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
