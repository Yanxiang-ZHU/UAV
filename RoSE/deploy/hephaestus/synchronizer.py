import socket
from unittest import case
import airsim
import time
import numpy as np
import os
import tempfile
import pprint
import cv2
import sys
import struct
import argparse

import threading
import traceback

import control_drone
import datetime

import collections
import pandas as pd

from dataclasses import dataclass
from heapq import *
import logging
import errno
import subprocess
import select
import signal
import faulthandler

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("tcp_debug.log"),
                              logging.StreamHandler(sys.stdout)])
def dump_conn_state(port):
    try:
        out = subprocess.check_output(["ss", "-tanp"], stderr=subprocess.DEVNULL, text=True)
        lines = [l for l in out.splitlines() if str(port) in l]
        logging.debug("ss -tanp lines for port %s:\n%s", port, "\n".join(lines))
    except Exception as e:
        logging.debug("dump_conn_state failed: %s", e)

def peer_closed(sock):
    try:
        r, _, _ = select.select([sock], [], [], 0)
        if r:
            try:
                data = sock.recv(1, socket.MSG_PEEK)
            except BlockingIOError:
                return False
            except OSError as e:
                logging.debug("peer_closed: recv peek OSError: %s", e)
                return True
            if data == b'':
                return True
        return False
    except Exception as e:
        logging.debug("peer_closed: select error: %s", e)
        return True

# Enable faulthandler so Python crashes produce stack traces in stderr/logs
faulthandler.enable()

# Optional: ensure signals produce logging (KeyboardInterrupt still fine)
def _log_and_exit(signum, frame):
    logging.error("Received signal %s, exiting", signum)
    dump_conn_state(SYNC_PORT)
    sys.exit(1)
for sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(sig, _log_and_exit)
    except Exception:
        pass

CS_GRANT_TOKEN  = 0x80 
CS_REQ_CYCLES   = 0x81 
CS_RSP_CYCLES   = 0x82 
CS_DEFINE_STEP  = 0x83
CS_RSP_STALL    = 0x84
CS_CFG_BW       = 0x85

CS_REQ_WAYPOINT = 0x01
CS_RSP_WAYPOINT = 0x02
CS_SEND_IMU     = 0x03
CS_REQ_ARM      = 0x04
CS_REQ_DISARM   = 0x05
CS_REQ_TAKEOFF  = 0x06

CS_REQ_IMG      = 0x10
CS_RSP_IMG      = 0x11
CS_REQ_IMG_POLL = 0x16
CS_RSP_IMG_POLL = 0x17

CS_REQ_DEPTH    = 0x12
CS_RSP_DEPTH    = 0x13
CS_REQ_DEPTH_STREAM = 0x14
CS_RSP_DEPTH_STREAM = 0x15

CS_SET_TARGETS  = 0x20
# AD_IMG_BACK     = 0x30 

CS_REQ_COOR_X   = 0x30
CS_RSP_COOR_X   = 0x31
CS_REQ_COOR_Y   = 0x32
CS_RSP_COOR_Y   = 0x33
CS_REQ_PHI      = 0x34
CS_RSP_PHI      = 0x35

INTCMDS = [CS_GRANT_TOKEN, CS_REQ_CYCLES, CS_RSP_CYCLES, CS_DEFINE_STEP, CS_RSP_STALL, CS_RSP_IMG, CS_CFG_BW, CS_RSP_IMG_POLL]

#HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
HOST = "localhost" # Private aws IP
#HOST = "172.31.30.244"
#AIRSIM_IP = "zr-desktop.cs.berkeley.edu"
#AIRSIM_IP = "localhost"
AIRSIM_IP = "10.134.141.151"
#PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
SYNC_PORT = 10001  # Port to listen on (non-privileged ports are > 1023)
DATA_PORT = 60002  # Port to listen on (non-privileged ports are > 1023)

INPUT_DIM = 56

def stable_heap_push(heap, item):
    heap.append(item)
    return heap

def stable_heap_pop(heap):
    item = heap.pop(0)
    heap.sort()
    return item

class CoSimLogger:
    
    def __init__(self, client, cycles, frames, filename=None):
        self.client = client
        self.table = {}
        self.df =  pd.DataFrame({"frame": [0]})
        self.started = False 
        self.start_time = time.time()
        self.cycles = cycles
        self.frames = frames
        try:
            f = open('angle.txt', 'r')
            yaw = int(float(f.readline()))
        except:
            yaw = "unknown"

        if filename is None:
            filename = f'./logs/runlog-angle-{yaw}-cycles-{cycles}-frames-{frames}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv'
        self.filename = filename
    
    def start(self):
        self.started = True
        self.start_time = time.time()

    def add_row(self, frame):
        to_add = pd.DataFrame.from_dict({"frame": [frame*self.frames], "cycles": [frame*self.cycles], "real_time": [time.time() - self.start_time]})
        self.df = pd.concat([self.df, to_add], ignore_index=True)
        # print(f"df row added: {self.df}")

    def log_targets(self, dat):
        self.record_row(dat)

    def log_true_kin(self):
        kin = self.client.simGetGroundTruthKinematics()
        dat = {}
        dat['ang_x_accel'] = kin.angular_acceleration.x_val
        dat['ang_y_accel'] = kin.angular_acceleration.y_val
        dat['ang_z_accel'] = kin.angular_acceleration.z_val
        dat['ang_x_vel']   = kin.angular_velocity.x_val
        dat['ang_y_vel']   = kin.angular_velocity.y_val
        dat['ang_z_vel']   = kin.angular_velocity.z_val
        dat['lin_x_accel'] = kin.linear_acceleration.x_val
        dat['lin_y_accel'] = kin.linear_acceleration.y_val
        dat['lin_z_accel'] = kin.linear_acceleration.z_val
        dat['lin_x_vel']   = kin.linear_velocity.x_val
        dat['lin_y_vel']   = kin.linear_velocity.y_val
        dat['lin_z_vel']   = kin.linear_velocity.z_val
        dat['x']   = kin.position.x_val
        dat['y']   = kin.position.y_val
        dat['z']   = kin.position.z_val
        depth = self.client.getDistanceSensorData("Distance").distance
        dat['depth'] = depth

        (pitch, roll, yaw) = airsim.utils.to_eularian_angles(kin.orientation)
        dat['pitch']   = pitch
        dat['roll']    = roll
        dat['yaw']     = yaw

        self.record_row(dat)
    
    # def log_event(self, kind):
    #     if kind not in self.df.columns:
    #         self.df = self.df.assign(**{kind:np.nan}).copy()
    #     if np.isnan(self.df.iloc[-1][kind]):
    #         self.df.iloc[-1][kind] = 1
    #     else:
    #         self.df.iloc[-1][kind] += 1
    

    def log_event(self, kind):
        # Check if the column exists, and if not, create it
        if kind not in self.df.columns:
            self.df[kind] = np.nan
        
        # Check if the last value in the column is NaN
        if np.isnan(self.df.loc[self.df.index[-1], kind]):
            self.df.loc[self.df.index[-1], kind] = 1
        else:
            self.df.loc[self.df.index[-1], kind] += 1

    # def record_row(self, data):
    #     print('----------------------- dat')
    #     print(data)
    #     for k, v in data.items():
    #         if k not in self.df.columns:
    #             self.df = self.df.assign(**{k:np.nan}).copy()
    #     for k, v in data.items():
    #         self.df.iloc[-1][k] = v
    #     print('======================== df')
    #     print(self.df)
    
    def record_row(self, data):
        # print('----------------------- data')
        # print(data)
        
        for k, v in data.items():
            if k not in self.df.columns:
                self.df[k] = np.nan
                
        for k, v in data.items():
            self.df.loc[self.df.index[-1], k] = v
        
        # print('======================== df')
        # print(self.df)

    def log_file(self):
        # print(self.df)
        self.df.to_csv(self.filename)
        # print(f"df logged to {self.filename}: {self.df}")
        print(f"df logged to {self.filename}")

class CoSimPacket:
    cmd_latency_dict = {CS_RSP_IMG: 0.0, CS_RSP_DEPTH: 0.0, CS_RSP_IMG_POLL: 0.0, CS_RSP_DEPTH_STREAM: 0.0}

    def __init__(self):
        self.cmd = None
        self.num_bytes = None
        self.data = None
        self.latency_enabled = None
        self.latency = None

    def __str__(self):
        return "[cmd: 0x{:02X}, num_bytes: {:04d}, data: {}]".format(self.cmd, self.num_bytes, self.data)

    def init(self, cmd, num_bytes, data):
        self.cmd = cmd
        self.num_bytes = num_bytes
        self.data = data
        self.latency_enabled = cmd in CoSimPacket.cmd_latency_dict.keys()
        self.latency = CoSimPacket.cmd_latency_dict.get(cmd, 0)

    def decode(self, buffer):
        self.cmd = int.from_bytes(buffer[0:4], "little", signed="False")
        self.num_bytes = int.from_bytes(buffer[4:8], "little", signed="False")
        data_array = [] 
        for i in range(self.num_bytes // 4):    
            if self.cmd in INTCMDS:
                data_array.append(int.from_bytes(buffer[4 * i + 8 : 4 * i + 12],  "little", signed="False"))
            else:
                data_array.append(struct.unpack("f", buffer[4 * i + 8 : 4 * i + 12])[0])
        self.data = data_array

    def encode(self):
        # Always emit standard wire header: [4B cmd][4B num_bytes]
        nb = 0 if (self.num_bytes is None) else int(self.num_bytes)
        buffer = self.cmd.to_bytes(4, 'little') + nb.to_bytes(4, 'little')
        if nb > 0 and self.data:
            for datum in self.data:
                if self.cmd in INTCMDS:
                    buffer += int(datum).to_bytes(4, "little", signed=False)
                else:
                    buffer += struct.pack("f", float(datum))
        return buffer

class Blob:
    # counter = 0
    def __init__(self, latency, packet):
        self.latency = latency
        self.packet = packet
        # self.counter = Blob.counter
        # Blob.counter += 1

    def __eq__(self, other):
        if isinstance(other, Blob):
            return self.latency == other.latency
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Blob):
            return self.latency < other.latency
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, Blob):
            return self.latency > other.latency
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, Blob):
            return self.latency <= other.latency
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, Blob):
            return self.latency >= other.latency
        return NotImplemented

class SocketThread (threading.Thread):
    def __init__(self, syn):
        threading.Thread.__init__(self)
        self.syn = syn
        self.killed = False
    
    def recv_all(self, n):
        data = b''
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def read_word(self):
        word = self.recv_all(4)
        return word

    def kill(self):
        self.killed = True

    def run(self):
        logging.debug("SocketThread started, connecting to %s:%s", self.syn.sync_host, self.syn.sync_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            self.sock = s
            try:
                s.connect((self.syn.sync_host, self.syn.sync_port))
            except Exception:
                logging.exception("connect failed")
                dump_conn_state(self.syn.sync_port)
                raise
            try:
                self.syn.server_started = True
                logging.debug("synchronizer.server_started set -> True")
            except Exception:
                logging.debug("failed to set server_started")
            logging.info("connected to %s", s.getpeername())

            count = 0
            while True:
                try:
                    if peer_closed(s):
                        logging.warning("peer_closed detected by MSG_PEEK/select -> remote closed or reset")
                        dump_conn_state(self.syn.sync_port)
                        break
                except Exception as e:
                    logging.debug("peer_closed check failed: %s", e)

                if self.killed:
                    logging.info("killed set, closing socket")
                    try:
                        s.close()
                    except:
                        pass
                    return
                try:
                    s.settimeout(0.5)
                    count += 1
                    if count % 200 == 0:
                        logging.debug("<heartbeat> peer=%s fd=%d", s.getpeername(), s.fileno())

                    cmd_data = self.recv_all(4)
                    if not cmd_data:
                        logging.warning("recv_all returned None for cmd -> peer closed or reset")
                        dump_conn_state(self.syn.sync_port)
                        break
                    cmd = int.from_bytes(cmd_data, "little", signed=False)
                    logging.debug("RX cmd=0x%02X", cmd)

                    num_bytes_data = self.recv_all(4)
                    if not num_bytes_data:
                        logging.warning("recv_all returned None for num_bytes -> peer closed")
                        dump_conn_state(self.syn.sync_port)
                        break
                    num_bytes = int.from_bytes(num_bytes_data, "little", signed=False)
                    logging.debug("RX num_bytes=%d", num_bytes)

                    payload = []
                    if num_bytes > 0:
                        raw = self.recv_all(num_bytes)
                        if raw is None:
                            logging.warning("recv_all returned None for payload -> peer closed")
                            dump_conn_state(self.syn.sync_port)
                            break
                        for i in range(0, len(raw), 4):
                            word = raw[i:i+4]
                            if cmd in INTCMDS:
                                payload.append(int.from_bytes(word, "little", signed=False))
                            else:
                                payload.append(struct.unpack("f", word)[0])

                    packet = CoSimPacket()
                    packet.init(cmd, num_bytes, payload)
                    queue = self.syn.sync_rxqueue if cmd > 0x80 else self.syn.data_rxqueue
                    queue.append(packet)
                    if cmd > 0x80:
                        self.syn.sync_rxqueue = queue
                    else:
                        self.syn.data_rxqueue = queue
                    logging.debug("Appended packet cmd=0x%02X to %s queue (len now %d/%d)", cmd,
                                  "sync" if cmd > 0x80 else "data", len(self.syn.sync_rxqueue), len(self.syn.data_rxqueue))

                except socket.timeout:
                    pass
                except (BrokenPipeError, ConnectionResetError) as e:
                    logging.exception("Connection broken during recv/send: %s", e)
                    try:
                        serr = s.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                        logging.debug("SO_ERROR after exception: %s (%d)", errno.errorcode.get(serr, "UNKNOWN"), serr)
                    except Exception as e2:
                        logging.debug("getsockopt failed: %s", e2)
                    dump_conn_state(self.syn.sync_port)
                    break
                except Exception as e:
                    logging.exception("Exception in SocketThread main loop: %s", e)
                    try:
                        serr = s.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                        logging.debug("SO_ERROR after general exception: %s (%d)", errno.errorcode.get(serr, "UNKNOWN"), serr)
                    except Exception:
                        pass
                    dump_conn_state(self.syn.sync_port)
                try:
                    while len(self.syn.txqueue) > 0:
                        packet = self.syn.txqueue.pop(0)
                        data = packet.encode()
                        try:
                            s.sendall(data)
                            logging.debug("Sent packet cmd=0x%02X len=%d", packet.cmd, len(data))
                        except (BrokenPipeError, ConnectionResetError) as e:
                            logging.exception("Broken pipe while sending: %s", e)
                            try:
                                serr = s.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                                logging.debug("SO_ERROR after send failure: %s (%d)", errno.errorcode.get(serr, "UNKNOWN"), serr)
                            except Exception:
                                pass
                            dump_conn_state(self.syn.sync_port)
                            raise
                except Exception as e:
                    logging.debug("tx send loop exception: %s", e)
            logging.info("SocketThread exiting main loop, closing socket")
            try:
                s.close()
            except:
                pass

class Synchronizer:
    def __init__(self, host, sync_port, data_port, firesim_step = 10000, airsim_step = 10, control=None, airsim_ip=AIRSIM_IP, cycle_limit=None, vehicle="drone", initial_y=0.0, terminal_x=None, logname=None):
        self.control   = control
        self.airsim_ip = airsim_ip
        self.sync_host = host
        self.data_host = host
        self.sync_port = sync_port
        self.data_port = data_port
        self.firesim_step = firesim_step
        self.airsim_step  = airsim_step
        self.packet = CoSimPacket()
        self.txqueue = []
        self.txpq = [] 
        self.data_rxqueue = []
        self.sync_rxqueue = []

        self.streaming_queue = collections.OrderedDict()

        self.cycle_limit = cycle_limit
        self.initial_y = initial_y
        self.terminal_x = terminal_x

        self.server_started = False

        print("Starting synchronizer code")
        print("Connecting to server")
        print(f"airsim_ip:{airsim_ip}")
        print(f"AIRSIM_IP:{AIRSIM_IP}")
        self.vehicle = vehicle
        if vehicle == "drone":
            self.client = airsim.MultirotorClient(ip=airsim_ip)
        elif vehicle == "car":
            self.client = airsim.CarClient(ip=airsim_ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.log_csv = None
        self.log_video = None
        if logname is not None:
            self.log_csv = f"{logname}.csv"
            #self.log_video = f"{logname}.mp4"
            self.log_video = f"{logname}.avi"
        self.logger = CoSimLogger(self.client, cycles=self.firesim_step, frames=self.airsim_step, filename=self.log_csv)
        print("Connected to server")
        print("Pausing simulator...")
        # self.client.simPause(True)  # change it back 11/13

        # Set up video logging
        fps = 30
        self.vid_width, self.vid_height = (256, 144)
        #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.vid_stream = cv2.VideoWriter(self.log_video, fourcc, fps, (self.vid_width, self.vid_height))
        if not self.vid_stream.isOpened():
            print("[ERROR] VideoWriter failed to open}")
        else:
            print("[sync]: VideoWriter opened successfully")
        CoSimPacket.firesim_step = firesim_step
    
    def run(self):
        print("\n[sync]: Synchronizer RUN!!")
        socket_thread = SocketThread(self)
        socket_thread.start()
        start_time = time.time()
        self.send_firesim_step()
        print("[sync]: send step")

        # read bw from bw.txt
        try:
            f = open('bw.txt', 'r')
            bw = int(f.readline())
            # self.send_bw(0, round(bw * self.firesim_step / 1e9))
            # self.send_bw(1, round(bw * self.firesim_step / 1e9))
            self.send_bw(1, bw)
            print(f"[sync]: bandwidth:{round(bw * self.firesim_step / 1e9)}")

        except:
            print("[sync]: pass bw.txt")
            pass
        # self.send_bw(0, 2048)
        print("[sync]: stepping beyond bw safely")
        # self.control.launchStabilizer(self.airsim_ip)
        count = 0
        start_frame = 0

        pose = self.client.simGetVehiclePose()
        pose.position.x_val = 0
        pose.position.y_val = self.initial_y
        # self.client.simSetVehiclePose(pose, ignore_collision=True)  # change it back

        # for test only ->
        # self.client.confirmConnection()
        # self.client.enableApiControl(True)
        # self.client.armDisarm(True)
        # self.client.takeoffAsync().join()

        # self.client.simContinueForFrames(1)

        img_row = 0


        while True:
            if os.path.exists('reset.txt'):
                print("[sync]: reset.txt works")
                start_time = time.time()
                count = 0
                pose = self.client.simGetVehiclePose()
                # try:
                #     f = open('angle.txt', 'r')
                #     yaw = float(f.readline()) * np.pi / 180
                #     pose.orientation = airsim.utils.to_quaternion(0,0,yaw)
                # except:
                #     print("[sync]: no angle.txt")
                #     pose.orientation = airsim.utils.to_quaternion(0,0, np.pi)
                # self.client.armDisarm(False)  # change it back
                pose.position.x_val = 0
                pose.position.y_val = self.initial_y
                # if self.vehicle == "drone":
                #     pose.position.z_val = 1
                # self.client.simSetVehiclePose(pose, ignore_collision=True)

                # self.client.simContinueForFrames(1)
                self.client.armDisarm(True)
                # self.control.targets['running'] = True  # change it back 11/13
                self.logger = CoSimLogger(self.client, cycles=self.firesim_step, frames=self.airsim_step, filename=self.log_csv)
                self.logger.start()
                print("[sync]: logger start!")
                os.remove('reset.txt')

            pose = self.client.simGetVehiclePose()
            print(f"[sync]: vehicle pose:{pose}")
            if (self.cycle_limit is not None and count * self.firesim_step >= self.cycle_limit) or (self.logger.started and self.terminal_x is not None and pose.position.x_val < self.terminal_x):
                print("[sync]: END")
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.vid_stream.release()
                # Append-adds at last
                if self.cycle_limit is not None and count * self.firesim_step >= self.cycle_limit:
                    print("Terminated due to exceeding maximum cycles!")
                else:
                    print("Terminated due to completing objective!")
                if self.cycle_limit is not None:
                    writestring = f"{elapsed_time}, {self.cycle_limit/elapsed_time}, {self.cycle_limit}, {self.firesim_step}, {self.airsim_step}"
                    os.system(f"echo {writestring} >> sim_data_test.log")
                    print(f"writestring {writestring}")
                self.logger.log_file()
                print(f"Terminating Simulation at {elapsed_time}")
                socket_thread.kill()
                time.sleep(1)
                exit()
            if self.logger.started:
                print("[sync]: Logging...")
                self.logger.add_row(count)
                self.logger.log_true_kin()
                dat = {}
                dat['target_z'] = self.control.targets['z']   
                dat['target_x_vel'] = self.control.targets['x_vel']  
                dat['target_y_vel'] = self.control.targets['y_vel']  
                dat['target_yawrwate'] = self.control.targets['yawrate'] 
                self.logger.log_targets(dat)
                rawImage = self.client.simGetImage("0", airsim.ImageType.Scene)
                print("[sync]: RAWIMAGE RECEIVED (for vid_stream)!")
                png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_COLOR)
                print(f"[sync]: PNG shape:{png.shape}, PNG type:{png.dtype}")

                self.vid_stream.write(png)
                print("[sync]: video write")
                if start_frame < 500:
                    cv2.imwrite(f'img/img_{start_frame}.png', png)
                elif start_frame == 500:
                    self.vid_stream.release()
                start_frame += 1
            if count % 20 == 0:
                print("[sync]: Stepping airsim")
                if(self.logger.started):
                    self.logger.log_file()
            # self.client.simContinueForFrames(self.airsim_step*10)  # change it back
            # self.client.simContinueForTime(self.airsim_step/100)
            if count % 20 == 0:
                print(f"[sync]: Granting fsim token: {count}")
            count = count + 1
            print(f"[sync]: count num:{count}")
            # IMPORTANT: process the data queue before granting tokens
            
            #process the latency aware queue
            #peek at the top of the queue
            print("[sync]: process the latency aware queue")
            while(len(self.txpq) > 0 and self.txpq[0].latency < 1):
                packet_blob = stable_heap_pop(self.txpq)
                packet = packet_blob.packet
                # print(f"DEBUG: the stable_heap_pop counter of this pop is: {packet_blob.counter}" )
                # print(f"DEBUG: the latency for this packet is: {packet_blob.latency} ")
                # print(f"DEBUG: the latency for this packet is: {packet_blob.packet.latency} ")
                self.txqueue.append(packet)
                #self.sync_conn.sendall(self.packet.encode())
            # Now, iterate through the rest of the queue, decrement latency by 1
            for blobs in self.txpq:
                blobs.latency = blobs.latency - 1
                blobs.packet.latency = blobs.latency
                print("Debug: --")
            print("[sync]: debug check: before granting fsim")
            self.grant_firesim_token()
            print("[sync]: grant firesim token")
            self.process_streaming_queue()
            print("[sync]: process streaming queue")

            # while True:
            #     if (self.client.simIsPause()):
            #         print("[sync]: sim pause")
            #         break
            print(f"[sync]: data_rxqueue len: {len(self.data_rxqueue)}")
            while len(self.data_rxqueue) > 0:
                self.process_fsim_data_packet()
                print("[sync]: process data from firesim")

            if count == 1:
                start_time = time.time()
        socket_thread.join()
    
    def send_firesim_step(self):
        packet = CoSimPacket()
        packet.init(CS_DEFINE_STEP, 4, [self.firesim_step])
        # print(f"Enqueuing step size: {packet}")
        self.txqueue.append(packet)
        #self.sync_conn.sendall(self.packet.encode())
    
    def send_bw(self, dst, bw):
        packet = CoSimPacket()
        packet.init(CS_CFG_BW, 8, [dst, bw])
        self.txqueue.append(packet)
        #self.sync_conn.sendall(self.packet.encode())
    """
    def grant_firesim_token(self, timeout=1.0, check_interval=0.05):
        print("[sync]: func(grant_firesim_token) Enqueuing new token")
        packet = CoSimPacket()
        packet.init(CS_GRANT_TOKEN, 0, None)
        self.txqueue.append(packet)
        #self.sync_conn.sendall(self.packet.encode())
        print(f"[sync]: func(grant_firesim_token) original sync_rxqueue len:{len(self.sync_rxqueue)}")

        start = time.time()
        printed_waiting = False
        while True:
            if len(self.sync_rxqueue) > 0:
                print(f"[sync]: func(grant_firesim_token) sync_rxqueue len:{len(self.sync_rxqueue)}")
                self.sync_rxqueue.pop(0)
                break
            if time.time() - start > timeout:
                print(f"[sync][WARN]: grant_firesim_token timeout {timeout:.2f}s — no response, continuing simulation")
                break
            if time.time() - start > timeout:
                print(f"[sync][WARN]: grant_firesim_token timeout {timeout:.2f}s — no response, continuing simulation")
                break

            if not printed_waiting:
                print("[sync]: func(grant_firesim_token) waiting for firesim response...")
                printed_waiting = True
            else:
                # print(".", end="", flush=True)
                print(".", end="")
            time.sleep(check_interval)
          
            print("[sync]: func(grant_firesim_token) done waiting, continue simulation")
    """
    def grant_firesim_token(self):
        # print("Enqueuing new token")
        packet = CoSimPacket()
        packet.init(CS_GRANT_TOKEN, 0, None)
        self.txqueue.append(packet)
        #self.sync_conn.sendall(self.packet.encode())

        while True:
            if len(self.sync_rxqueue) > 0:
                self.sync_rxqueue.pop(0)
                break


    def get_firesim_cycles(self):
        packet = CoSimPacket()
        packet.init(CS_REQ_CYCLES, 0, None)
        self.txqueue.append(packet)

        while len(self.sync_rxqueue) == 0:
            pass
        response = self.sync_rxqueue.pop(0)
        return response.data[0]
    
    def send_test_firesim_data_packet(self):
        packet = CoSimPacket()
    
    def process_streaming_queue(self):
        for key,enabled in self.streaming_queue.items():
            if enabled:
                if key == CS_RSP_DEPTH_STREAM:
                    depth = self.client.getDistanceSensorData("Distance").distance
                    packet = CoSimPacket()
                    packet.init(CS_RSP_DEPTH_STREAM, 4, [depth])
                    blob = Blob(packet.latency, packet)
                    stable_heap_push(self.txpq, blob)
                else:
                    pass

    def process_fsim_data_packet(self):
        packet = self.data_rxqueue.pop(0)
        print(f"Dequeued data packet: {packet}")
        if packet.cmd == CS_REQ_ARM:
            print("---------------------------------------------------")
            print("Arming...")
            print("---------------------------------------------------")
            # pose = self.client.simGetVehiclePose()
            # # try:
            # #     f = open('angle.txt', 'r')
            # #     yaw = float(f.readline()) * np.pi / 180
            # #     pose.orientation = airsim.utils.to_quaternion(0,0,yaw)
            # # except:
            # #     pose.orientation = airsim.utils.to_quaternion(0,0, np.pi)
            # #     # pose.orientation = airsim.utils.to_quaternion(0,0,0)
            # self.client.armDisarm(False)
            # pose.position.x_val = 0
            # pose.position.y_val = self.initial_y
            # pose.position.z_val = -10
            # self.client.simSetVehiclePose(pose, ignore_collision=True)
            # self.client.simContinueForFrames(5)

            # pose.position.x_val = 0
            # pose.position.y_val = self.initial_y
            # print(f"Setting initial x, y: ({pose.position.x_val}, {pose.position.y_val})")
            # # if self.vehicle == "drone":
            # #     pose.position.z_val = 1
            # self.client.simSetVehiclePose(pose, ignore_collision=True)
            # self.client.simContinueForFrames(1)
            
            # ##################################
            # self.client.enableApiControl(True)
            # state = self.client.getMultirotorState()
            # s = pprint.pformat(state)
            # print("state: %s" % s)
            # imu_data = self.client.getImuData()
            # s = pprint.pformat(imu_data)
            # print("imu_data: %s" % s)
            # barometer_data = self.client.getBarometerData()
            # s = pprint.pformat(barometer_data)
            # print("barometer_data: %s" % s)
            # magnetometer_data = self.client.getMagnetometerData()
            # s = pprint.pformat(magnetometer_data)
            # print("magnetometer_data: %s" % s)
            # gps_data = self.client.getGpsData()
            # s = pprint.pformat(gps_data)
            # print("gps_data: %s" % s)
            # ##################################
            pose = self.client.simGetVehiclePose()
            pose.position.x_val = 0
            pose.position.y_val = 0
            pose.position.z_val = -10
            self.client.simSetVehiclePose(pose, ignore_collision=True)
            self.client.armDisarm(True)
            self.control.targets['running'] = True
            self.logger = CoSimLogger(self.client, cycles=self.firesim_step, frames=self.airsim_step, filename=self.log_csv)
            self.logger.start()
        elif packet.cmd == CS_REQ_TAKEOFF:
            print("---------------------------------------------------")
            print("Taking off...")
            print("---------------------------------------------------")
            if self.vehicle == "drone":

                state = self.client.getMultirotorState()
                print("Landed state:", state.landed_state)

                self.client.enableApiControl(True)
                self.client.armDisarm(True)

                print("API control:", self.client.isApiControlEnabled())
                print(state.kinematics_estimated.position)
                print(state.collision.has_collided)

                self.client.takeoffAsync().join()
                state = self.client.getMultirotorState()
                print("After takeoff: z =", state.kinematics_estimated.position.z_val)
                print("Landed state:", state.landed_state)

                print("Moving up...")
                self.client.moveToZAsync(-4, 2).join()

                state = self.client.getMultirotorState()
                print("Now: z =", state.kinematics_estimated.position.z_val)
                print("Landed state:", state.landed_state)
                # self.client.takeoffAsync().join()

                # # self.client.moveToZAsync(-10, 1)
                # state = self.client.getMultirotorState()  
                # print("Landed state:", state.landed_state)
                # # self.client.moveByVelocityAsync(0, 0, 0, 0)
            self.control.targets['running'] = True

        elif packet.cmd == CS_RSP_WAYPOINT:
            print("---------------------------------------------------")
            print(f"Flying to: {packet.data[0]}, {packet.data[1]}, {packet.data[2]}")
            print("---------------------------------------------------")
            self.client.moveToPositionAsync(packet.data[0],packet.data[1],packet.data[2], packet.data[3]).join()
            # self.client.moveToPositionAsync(10, 10, -5, 1).join()
            # self.client.hoverAsync().join()

        elif packet.cmd == CS_SET_TARGETS:
            print("---------------------------------------------------")
            print(f"z: {packet.data[0]}, x_vel: {packet.data[1]}, y_vel: {packet.data[2]}, yawrate: {packet.data[3]}")
            print("---------------------------------------------------")
            self.logger.log_event("target_req")
            self.control.targets['z']       = packet.data[0]
            self.control.targets['x_vel']   = packet.data[1]
            self.control.targets['y_vel']   = packet.data[2]
            self.control.targets['yawrate'] = packet.data[3]
        elif packet.cmd == CS_REQ_IMG:
            print("---------------------------------------------------")
            print("Got image request...")
            print("---------------------------------------------------")
            self.logger.log_event("img_req")
            rawImage = self.client.simGetImage("0", airsim.ImageType.Scene)
            print("[sync]: RAWIMAGE RECEIVED (for req_img)!")
            # png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_COLOR)
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_COLOR)
            cv2.imwrite('img/img.png', png)
            png = cv2.resize(png, (INPUT_DIM, INPUT_DIM))
            png_arr = png.reshape((INPUT_DIM,INPUT_DIM*3))
            print(png_arr.shape)
            # print(png_arr)
            # png_arr = png.reshape((172_800))
            k = 0
            for row in png_arr:
                png_packet_arr = row.view(np.uint32).tolist()
                if k < 4:
                    print(len(png_packet_arr))
                    print([hex(x) for x in png_packet_arr])
                    print([(i, hex(png_packet_arr[i])) for i in range(len(png_packet_arr))])
                    k += 1
                # print(png_packet_arr)
                packet = CoSimPacket()
                packet.init(CS_RSP_IMG, len(png_packet_arr)*4, png_packet_arr)
                blob = Blob(packet.latency, packet)
                stable_heap_push(self.txpq, blob)
        elif packet.cmd == CS_REQ_IMG_POLL:
            print("---------------------------------------------------")
            print("Got image polling request...")
            print("---------------------------------------------------")
            self.logger.log_event("img_req")
            rawImage = self.client.simGetImage("0", airsim.ImageType.Scene)
            # png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_COLOR)
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_COLOR)
            cv2.imwrite('img/img.png', png)
            png = cv2.resize(png, (INPUT_DIM, INPUT_DIM))
            png_arr = png.reshape((INPUT_DIM,INPUT_DIM*3))
            # print(png_arr.shape)
            # print(png_arr)
            # png_arr = png.reshape((172_800))
            k = 0
            for row in png_arr:
                png_packet_arr = row.view(np.uint32).tolist()
                if k < 4:
                    print(len(png_packet_arr))
                    print([hex(x) for x in png_packet_arr])
                    print([(i, hex(png_packet_arr[i])) for i in range(len(png_packet_arr))])
                    k += 1
                print(png_packet_arr)
                packet = CoSimPacket()
                packet.init(CS_RSP_IMG_POLL, len(png_packet_arr)*4, png_packet_arr)
                blob = Blob(packet.latency, packet)
                stable_heap_push(self.txpq, blob)
        elif packet.cmd == CS_REQ_DEPTH:
            print("---------------------------------------------------")
            print("Got depth request...")
            print("---------------------------------------------------")
            packet = CoSimPacket()
            packet.init(CS_RSP_DEPTH_STREAM, 4, [1])
            blob = Blob(packet.latency, packet) 
            stable_heap_push(self.txpq, blob) 
            depth = self.client.getDistanceSensorData("Distance").distance
            print(depth)
            packet = CoSimPacket()
            packet.init(CS_RSP_DEPTH, 4, [depth])
            blob = Blob(packet.latency, packet)
            stable_heap_push(self.txpq, blob)
            print("[sync/depth] End depth request")
            # CoSimPacket.cmd_latency_dict[CS_RSP_DEPTH] += 0.25
        elif packet.cmd == CS_REQ_DEPTH_STREAM:
            print("---------------------------------------------------")
            print("Got depth streaming request...")
            print("---------------------------------------------------") 
            # self.streaming_queue[CS_RSP_DEPTH_STREAM] = True
            depth = 100 # dummy data
            packet_vanilla = CoSimPacket()
            packet_chocolate = CoSimPacket()
            packet_vanilla.init(CS_RSP_DEPTH_STREAM, 4, [depth])
            packet_chocolate.init(CS_RSP_DEPTH, 4, [depth])
            bloba_vanilla = Blob(packet_vanilla.latency, packet_vanilla)
            bloba_chocolate = Blob(packet_chocolate.latency, packet_chocolate)
            stable_heap_push(self.txpq, bloba_vanilla)
            stable_heap_push(self.txpq, bloba_chocolate)
        elif packet.cmd == CS_REQ_COOR_X:
            print("---------------------------------------------------")
            print("Got X request...")
            print("---------------------------------------------------") 
            pose = self.client.simGetVehiclePose()
            x_val = float(pose.position.x_val)
            rsp = CoSimPacket()
            rsp.init(CS_RSP_COOR_X, 4, [x_val])
            self.txqueue.append(rsp)
        elif packet.cmd == CS_REQ_COOR_Y:
            print("---------------------------------------------------")
            print("Got Y request...")
            print("---------------------------------------------------") 
            pose = self.client.simGetVehiclePose()
            y_val = float(pose.position.y_val)
            rsp = CoSimPacket()
            rsp.init(CS_RSP_COOR_Y, 4, [y_val])
            self.txqueue.append(rsp)
        elif packet.cmd == CS_REQ_PHI:
            print("---------------------------------------------------")
            print("Got PHI request...")
            print("---------------------------------------------------") 
            pose = self.client.simGetVehiclePose()
            _, _, yaw = airsim.utils.to_eularian_angles(pose.orientation)
            rsp = CoSimPacket()
            rsp.init(CS_RSP_PHI, 4, [float(yaw)])
            self.txqueue.append(rsp)
        # elif packet.cmd == AD_IMG_BACK:
        #     print(packet.cmd, packet.num_bytes, packet.data)
        #     row_bytes = b''.join([p.to_bytes(4, 'little') for p in  packet.data])
        #     row_np = np.frombuffer(row_bytes, dtype=np.uint8)[:INPUT_DIM*3]
        #     row_np = row_np.reshape((1, INPUT_DIM, 3))
        #     img_rows.append(row_np)
        #     img_row = img_row + 1
        #     if img_row >= INPUT_DIM:
        #         img_row = 0
        #         img_np = np.concatenate(img_rows, axis=0)
        #         img_rows = []
        #         cv2.imwrite(f"./sync_backrecv/received_img_{img_index:03d}.png", img_np)
        #         img_index = img_index + 1
        #         print(f"Image shape: {img_np.shape}")
        else:
            pass
        
if __name__ == "__main__":
    arg_list = argparse.ArgumentParser()

    # Add arguments to the parser
    arg_list.add_argument("-a", "--Airsim-steps", type=int, default=1, help="airsim steps")
    arg_list.add_argument("-f", "--Firesim-steps", type=int, default=20000000, help="firesim steps")
    arg_list.add_argument("-c", "--Cycle-limit", type=int, default = None)
    args = vars(arg_list.parse_args())
    print(args)

    control = control_drone.IntermediateDroneApi()
    print(f"Firesim_step: {args['Firesim_steps']}")
    sync = Synchronizer(HOST, SYNC_PORT, DATA_PORT, firesim_step=args['Firesim_steps'], airsim_step=args['Airsim_steps'], cycle_limit=args['Cycle_limit'])
    sync.run()
    
    while True:
        rawImage = client.simGetImage("0", airsim.ImageType.Scene)
        if (rawImage == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_COLOR)
            png_arr = png.reshape((172_800))

        
def run_trailnet(ort_sess, img):
    img_BGR = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # im_f = img_BGR/255
    im_f = img_BGR.astype(np.float32)
    im_f = np.expand_dims(im_f,axis=0)
    # print(im_f.shape)
    im_f = im_f.transpose(0, 3,1,2)
