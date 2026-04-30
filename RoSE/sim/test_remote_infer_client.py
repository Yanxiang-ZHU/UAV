#!/usr/bin/env python3
import json
import socket
from configparser import ConfigParser

import numpy as np


CONFIG_PATH = "/home/data/userhome/zhuyanxiang/RoSE/RoSE/sim/config.ini"
REMOTE_HOST = "10.134.141.151"
REMOTE_PORT = 50052
GOAL_DISTANCE = 100.0


def build_payload_from_inputs(dxy: float, dyaw: float, depth_obs_image: np.ndarray) -> dict:
    if depth_obs_image is None:
        raise ValueError("depth_obs_image is None")
    if depth_obs_image.dtype != np.uint8:
        depth_obs_image = depth_obs_image.astype(np.uint8)

    payload = {
        "state_is_normalized": False,
        "goal_distance": float(GOAL_DISTANCE),
        "state": {
            "dxy": float(dxy),
            "dyaw": float(dyaw),
        },
        "depth_image": depth_obs_image.tolist(),
    }
    return payload


def send_infer(
    dxy: float,
    dyaw: float,
    depth_obs_image: np.ndarray,
    host: str = REMOTE_HOST,
    port: int = REMOTE_PORT,
    timeout: float = 5.0,
) -> str:
    payload = build_payload_from_inputs(dxy, dyaw, depth_obs_image)
    msg = json.dumps(payload) + "\n"

    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(msg.encode("utf-8"))
        resp = s.recv(4096).decode("utf-8").strip()
    print("[RESP]###")
    print(resp)
    return resp


def main() -> None:
    raise SystemExit("Use send_infer(dxy, dyaw, depth_obs_image) from this module.")


if __name__ == "__main__":
    main()
