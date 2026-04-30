import os
import argparse
import threading
import time
import synchronizer
import control_drone
import control_car

HOST = "localhost"
# AIRSIM_IP = "localhost"
AIRSIM_IP = "10.134.141.151"

SYNC_PORT = 10001
DATA_PORT = 60002

# Thread to track synchronization code
class SyncThread(threading.Thread):
    def __init__(self, args):
        threading.Thread.__init__(self)
        if args["Vehicle_type"] == "drone":
            control = control_drone.IntermediateDroneApi()
        elif args["Vehicle_type"] == "car":
            control = control_car.IntermediateCarApi()
        else:
            raise ValueError("Unsupported vehicle type")
        self.sync = synchronizer.Synchronizer(
            HOST, SYNC_PORT, DATA_PORT,
            firesim_step=args['Firesim_steps'],
            airsim_step=args['Airsim_steps'],
            control=control,
            airsim_ip=args['Airsim_ip'],
            cycle_limit=args['Cycle_limit'],
            vehicle=args["Vehicle_type"],
            initial_y=args["Initial_y"],
            terminal_x=args["Terminal_x"],
            logname=args["Log_file"]
        )

    def run(self):
        self.sync.run()


if __name__ == "__main__":
    arg_list = argparse.ArgumentParser()

    # Keep only useful args
    arg_list.add_argument("-a", "--Airsim-steps", type=int, default=1, help="airsim steps")
    arg_list.add_argument("-f", "--Firesim-steps", type=int, default=10000, help="firesim steps (used only for sync pacing)")
    arg_list.add_argument("-i", "--Airsim-ip", default=AIRSIM_IP, help="IP address of airsim server")
    arg_list.add_argument("-c", "--Cycle-limit", type=int, default=None)
    arg_list.add_argument("-v", "--Vehicle-type", type=str, default="drone", help="Vehicle to simulate: drone or car")
    arg_list.add_argument("-y", "--Initial-y", type=float, default=0.0, help="Initial y position")
    arg_list.add_argument("-x", "--Terminal-x", type=float, default=None, help="Terminal x position to stop simulation")
    arg_list.add_argument("-l", "--Log-file", type=str, default=None, help="Log file name (no extension)")
    args = vars(arg_list.parse_args())
    
    print("runner start!")
    sync_thread = SyncThread(args)
    print("Starting synchronizer thread")
    sync_thread.start()

    while not sync_thread.sync.server_started:
        print("Waiting for synchronizer server to start...")
        time.sleep(0.1)

    print("Joining synchronizer thread")
    sync_thread.join()
    print("runner_sim: join end")
