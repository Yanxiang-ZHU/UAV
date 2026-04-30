import json
import socket
from configparser import ConfigParser
import numpy as np


CONFIG_PATH = "/home/data/zhuyanxiang/UAVDRL/UAV_Navigation_DRL_AirSim/logs_save/test/config/config.ini"
REMOTE_HOST = "127.0.0.1"
REMOTE_PORT = 50052
GOAL_DISTANCE = 100.0


def build_payload(cfg: ConfigParser) -> dict:
    screen_height = cfg.getint("environment", "screen_height")
    screen_width = cfg.getint("environment", "screen_width")

    dxy = 30.0
    dyaw = 0.1

    depth_image = (np.random.rand(screen_height, screen_width) * 255.0).astype(np.uint8)

    payload = {
        "state_is_normalized": False,
        "goal_distance": float(GOAL_DISTANCE),
        "state": {
            "dxy": float(dxy),
            "dyaw": float(dyaw),
        },
        "depth_image": depth_image.tolist(),
    }
    return payload


def main() -> None:
    cfg = ConfigParser()
    cfg.read(CONFIG_PATH)

    payload = build_payload(cfg)
    msg = json.dumps(payload) + "\n"

    with socket.create_connection((REMOTE_HOST, REMOTE_PORT), timeout=5) as s:
        s.sendall(msg.encode("utf-8"))
        resp = s.recv(4096).decode("utf-8").strip()

    print("Response:", resp)


if __name__ == "__main__":
    main()
