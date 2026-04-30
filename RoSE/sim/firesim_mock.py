#!/usr/bin/env python3
from colorama import init
import traceback
import threading
import socket, threading, struct, random, time, os, subprocess, sys, signal
import numpy as np
import cv2
import math
from test_remote_infer_client import send_infer

# from inference_realesrgan_reorg import sr_image_np


sys.stdout.reconfigure(line_buffering=True)
init()

# ======= basic =======
HOST = "127.0.0.1"
ROSE_PORT = 10001
SYNC_PORT = 10002
BUFFER_SIZE = 4096

# ======= Mock FireSim commands =======
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
CS_REQ_DEPTH    = 0x12
CS_RSP_DEPTH    = 0x13
CS_REQ_DEPTH_STREAM = 0x14
CS_RSP_DEPTH_STREAM = 0x15
CS_REQ_IMG_POLL = 0x16
CS_RSP_IMG_POLL = 0x17

CS_SET_TARGETS  = 0x20

CS_REQ_COOR_X   = 0x30
CS_RSP_COOR_X   = 0x31
CS_REQ_COOR_Y   = 0x32
CS_RSP_COOR_Y   = 0x33
CS_REQ_PHI      = 0x34
CS_RSP_PHI      = 0x35

INPUT_DIM = 56

GOAL_POINT = (83.0, 0.0)
current_x = None
current_y = None
current_phi = None
depth_image = None
depth_obs_image = None
max_depth_meters = 15.0
screen_height = 60
screen_width = 90
last_infer_response = None
last_infer_error = None

def wrap_angle(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi

def try_compute_relative():
    if current_x is None or current_y is None or current_phi is None:
        return None, None
    gx, gy = GOAL_POINT
    dxy = math.hypot(current_x - gx, current_y - gy)
    phi_goal = math.atan2(gy - current_y, gx - current_x)
    dphi = wrap_angle(phi_goal - current_phi)
    return dxy, dphi

def send_infer_and_store(dxy, dphi, obs_image):
    global last_infer_response, last_infer_error
    try:
        last_infer_response = send_infer(dxy, dphi, obs_image)
        last_infer_error = None
        print(f"[Mock] infer response: {last_infer_response}")
    except Exception as e:
        last_infer_error = f"{e.__class__.__name__}: {e}"
        print(f"[Mock] infer error: {last_infer_error}")

# ======= Helper functions =======
def kill_port(port):
    try:
        pids = subprocess.check_output(f"lsof -t -i:{port}", shell=True, text=True).split()
        if pids:
            print(f"[killscreen] Killing old process on port {port} (PID {' '.join(pids)})...")
            os.system("kill -9 " + " ".join(pids))
            time.sleep(1)
            # print("[Debug] in kill_port")
    except subprocess.CalledProcessError:
        pass

def wait_for_port_release(port, timeout=10):
    for _ in range(timeout):
        result = os.system(f"netstat -tuln | grep {port} > /dev/null 2>&1")
        if result != 0:
            return True
        print(f"[wait] Port {port} still in use, waiting...")
        time.sleep(1)
    raise RuntimeError(f"Port {port} still not released after {timeout}s")

def recv_all(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    print(f"[Mock/RECV]: recv_all:{data}")
    return data

def recv_all_f(conn, n):
    data = conn.recv(n)
    print(f"[Mock/RECV]: recv_all_f:{data}")
    return data

def send_packet(conn, cmd, data, is_float=False):
    """
    Encode and send packet: [4B cmd][4B num_bytes][payload...]
    data: list of numbers; if is_float True -> pack as float32, else unsigned int32
    """
    print("[Mock/SEND] prepare to send")
    num_bytes = len(data) * 4 if data else 0
    pkt = cmd.to_bytes(4, 'little') + num_bytes.to_bytes(4, 'little')
    for datum in data:
        pkt += struct.pack("f", float(datum)) if is_float else int(datum).to_bytes(4, "little", signed=False)
    try:
        conn.sendall(pkt)
        print(f"[Mock/SEND] send_packet cmd=0x{cmd:02X}, total={len(pkt)} bytes sent to {conn.getpeername()}")
    except (BrokenPipeError, ConnectionResetError) as e:
        print(f"[Mock/SEND] sendall error : {e.__class__.__name__}: {e}")
        try:
            peer = conn.getpeername()
        except Exception:
            peer = "(unknown)"
        print(f"[Mock/SEND] peer at error: {peer}")
        try:
            conn.close()
        except Exception:
            pass
        raise
    except OSError as e:
        print(f"[Mock/SEND] OSError during sendall: {e}")
        try:
            conn.close()
        except Exception:
            pass
        raise

# ======= Mock FireSim Core State =======
cycle_counter = 0
firesim_step = 10000   # default step; set by CS_DEFINE_STEP
latency_queue = []  # list of (send_time, cmd, data, is_float)
latency_lock = threading.Lock()


def enqueue_response(delay_s, cmd, data=None, is_float=False):
    """Schedule a response to be sent after delay_s seconds."""
    if data is None:
        data = []
    send_time = time.time() + delay_s
    #send_time = time.time()
    with latency_lock:
        latency_queue.append((send_time, cmd, data, is_float))
        print(f"[Mock] latency_queue append cmd=0x{cmd:02X}")

def process_latency_queue(conn):
    """Send all due responses. Must be called frequently in the client loop."""
    now = time.time()
    to_send = []
    with latency_lock:
        remain = []
        for item in latency_queue:
            if item[0] <= now:
                print(f"[Mock] Sending: cmd=0x{item[1]:02X}")
                to_send.append(item)
            else:
                remain.append(item)
        latency_queue[:] = remain
    for (_, cmd, data, is_float) in to_send:
        print(f"[Mock/SEND]: conn={conn}, cmd={cmd}, data_len={(len(data)*4) if data else 0}, is_float={is_float}")
        try:
            send_packet(conn, cmd, data, is_float=is_float)
        except Exception as e:
            print(f"[Mock] process_latency_queue send failed: {e.__class__.__name__}: {e}")
            with latency_lock:
                latency_queue.clear()
            try:
                conn.close()
            except Exception:
                pass
            return

# ======= Packet handling modeled on airsim.cc behavior =======
'''
def handle_packet_logic(cmd, payload):
    """
    Return tuple (resp_cmd, resp_data_list, is_float, delay_seconds).
    If resp_cmd is None -> means no response needed.
    """
    global cycle_counter, firesim_step

    # Default: no response
    resp_cmd = None
    resp_data = []
    is_float = False
    delay = 0.005  # small processing delay

    # CS_DEFINE_STEP: set step size (payload[0] is uint)
    if cmd == CS_DEFINE_STEP:
        if payload:
            firesim_step = int(payload[0])
            print(f"[Mock] CS_DEFINE_STEP received: firesim_step={firesim_step}")
        else:
            print("[Mock] CS_DEFINE_STEP received: no payload")
        # airsim.cc: typically no immediate response for define step
        resp_cmd = None

    # CS_CFG_BW: configure bandwidth (no response required)
    elif cmd == CS_CFG_BW:
        print(f"[Mock] CS_CFG_BW payload={payload} (ignored in mock)")
        resp_cmd = None

    # CS_GRANT_TOKEN: allow firesim to step; after performing step send CS_RSP_STALL (or similar)
    elif cmd == CS_GRANT_TOKEN:
        # In airsim.cc, GRANT_TOKEN causes execution of cycles and then RSP_STALL is used to indicate done/stall
        cycle_counter += firesim_step
        print(f"[Mock] CS_GRANT_TOKEN received. Advance cycles -> cycle_counter={cycle_counter}")
        resp_cmd = CS_RSP_STALL
        resp_data = []    # RSP_STALL typically no payload
        is_float = False
        delay = 0.01      # simulate slight processing time before sending stall

    # CS_REQ_CYCLES: return current cycles
    elif cmd == CS_REQ_CYCLES:
        print(f"[Mock] CS_REQ_CYCLES -> respond cycles={cycle_counter}")
        resp_cmd = CS_RSP_CYCLES
        resp_data = [cycle_counter]
        is_float = False
        delay = 0.001

    # CS_SET_TARGETS: Set control targets; reply with GRANT_TOKEN (ack)
    elif cmd == CS_SET_TARGETS:
        print(f"[Mock] CS_SET_TARGETS payload(len)={len(payload)} (interpreted as floats?)")
        # in your synchronizer, SET_TARGETS uses floats; here we just ack
        resp_cmd = CS_RSP_STALL
        resp_data = []
        is_float = False
        delay = 0.001

    # CS_REQ_WAYPOINT -> reply RSP_WAYPOINT with floats
    elif cmd == CS_REQ_WAYPOINT:
        resp_cmd = CS_RSP_WAYPOINT
        resp_data = [1.0, 2.0, 3.0]  # dummy waypoint floats (x,y,z)
        is_float = True
        print("[Mock] CS_REQ_WAYPOINT -> sending dummy waypoint")
        delay = 0.01

    # ARM/DISARM/TAKEOFF -> ack with GRANT_TOKEN
    elif cmd == CS_REQ_ARM or cmd == CS_REQ_DISARM or cmd == CS_REQ_TAKEOFF:
        print(f"[Mock] control cmd 0x{cmd:02X} -> ack (CS_GRANT_TOKEN)")
        resp_cmd = CS_GRANT_TOKEN
        resp_data = []
        is_float = False
        delay = 0.005

    # Image requests
    elif cmd == CS_REQ_IMG or cmd == CS_REQ_IMG_POLL:
        resp_cmd = CS_RSP_IMG if cmd == CS_REQ_IMG else CS_RSP_IMG_POLL
        # Create random image floats in [0,255)
        # Note: synchronizer expects rows flattened into uint32 view in original code,
        # but it decodes as floats for these commands. We'll send float32 pixels.
        img = (np.random.rand(INPUT_DIM * INPUT_DIM * 3) * 255.0).astype('float32')
        resp_data = img.tolist()
        is_float = True
        print(f"[Mock] CS_REQ_IMG/_POLL -> sending fake image ({len(resp_data)} floats)")
        delay = 0.02  # image generation delay

    # Depth requests
    elif cmd == CS_REQ_DEPTH:
        resp_cmd = CS_RSP_DEPTH
        resp_data = [random.uniform(0.5, 50.0)]
        is_float = True
        print(f"[Mock] CS_REQ_DEPTH -> sending depth {resp_data[0]:.2f}")
        delay = 0.005

    elif cmd == CS_REQ_DEPTH_STREAM:
        resp_cmd = CS_RSP_DEPTH_STREAM
        resp_data = [random.uniform(0.5, 50.0)]
        is_float = True
        print("[Mock] CS_REQ_DEPTH_STREAM -> sending one depth sample")
        delay = 0.002

    else:
        print(f"[Mock] Unknown cmd=0x{cmd:02X}, payload={payload}. Replying with GRANT_TOKEN as fallback")        
        resp_cmd = CS_GRANT_TOKEN
        resp_data = []
        is_float = False
        delay = 0.001

    return resp_cmd, resp_data, is_float, delay
'''

# ======= Client handler (mock server) =======
depth_flag = 0
img_flag = 0
depth_stream_flag = 0
x_flag = 0
y_flag = 0
phi_flag = 0
store_png = []
flag_lock = threading.Lock()

def sr_and_save(img, idx):
    print("[SR] start SR")
    from inference_realesrgan_reorg import sr_image_np
    print("[SR] import done")
    sr = sr_image_np(img)
    print("[SR] img done")
    cv2.imwrite(f"received_img_sr_{idx:03d}.png", sr)

def handle_client(conn):
    global depth_flag, img_flag, depth_stream_flag, x_flag, y_flag, phi_flag
    global store_png
    global current_x, current_y, current_phi
    global depth_image, depth_obs_image

    print("[Mock] connected from", conn.getpeername())
    latency_thread = threading.Thread(target=latency_loop, args=(conn,), daemon=True)
    latency_thread.start()
    
    send_thread = threading.Thread(target=send_mock, args=(), daemon=True)
    send_thread.start()

    img_row = 0
    img_rows = [] 
    img_index = 0
    
    try:
        while True:
            #process_latency_queue(conn)

            # Read header
            header = recv_all(conn, 4)
            if not header:
                print("[Mock] Not Header BREAK")
                break
            cmd = int.from_bytes(header, "little", signed=False)
            
            # # Optional decode 
            # if cmd in [CS_RSP_IMG, CS_RSP_DEPTH, CS_RSP_IMG_POLL, CS_RSP_DEPTH_STREAM]:
            #     latency_data = recv_all_f(conn, 4)

            # Read num bytes
            num_bytes_data = recv_all_f(conn, 4)
            if not num_bytes_data:
                print("[Mock]: Not num_dytes_data BREAK")
                break
            num_bytes = int.from_bytes(num_bytes_data, "little", signed=False)

            payload = []
            if num_bytes > 0:
                raw = recv_all_f(conn, num_bytes)
                if raw is None:
                    print("[Mock]: raw is None BREAK")
                    break
                # Interpret payload: for some cmds (SET_TARGETS, RSP_WAYPOINT) data are float32,
                # but for internal commands and many requests they are ints. We'll attempt both:
                for i in range(0, len(raw), 4):
                    word = raw[i:i+4]
                    # Heuristic: if command is SET_TARGETS or RSP_WAYPOINT or RSP_DEPTH_STREAM -> float
                    if cmd in [CS_SET_TARGETS, CS_RSP_WAYPOINT, CS_RSP_DEPTH_STREAM]:
                        payload.append(struct.unpack("f", word)[0])
                    else:
                        payload.append(int.from_bytes(word, "little", signed=False))

            # print(f"\033[35m[Mock] RX cmd=0x{cmd:02X}, num_bytes={num_bytes}, payload_len={len(payload)}\033[0m")
            print(f"\n[Mock] RX cmd=0x{cmd:02X}, num_bytes={num_bytes}, payload_len={len(payload)}\n")

            if cmd == CS_RSP_COOR_X and payload:
                current_x = float(payload[0])
                dxy, dphi = try_compute_relative()
                if dxy is not None:
                    print(f"[Mock] dxy={dxy:.4f}, dphi={dphi:.4f} (rad)")

            if cmd == CS_RSP_COOR_Y and payload:
                current_y = float(payload[0])
                dxy, dphi = try_compute_relative()
                if dxy is not None:
                    print(f"[Mock] dxy={dxy:.4f}, dphi={dphi:.4f} (rad)")

            if cmd == CS_RSP_PHI and payload:
                current_phi = float(payload[0])
                dxy, dphi = try_compute_relative()
                if dxy is not None:
                    print(f"[Mock] dxy={dxy:.4f}, dphi={dphi:.4f} (rad)")

            if cmd == CS_RSP_DEPTH_STREAM and payload:
                if len(payload) == INPUT_DIM * INPUT_DIM:
                    depth_image = np.array(payload, dtype=np.float32).reshape(INPUT_DIM, INPUT_DIM)
                    image_resize = cv2.resize(depth_image, (screen_width, screen_height))
                    min_distance_to_obstacles = float(image_resize.min())
                    image_scaled = np.clip(image_resize, 0, max_depth_meters) / max_depth_meters * 255.0
                    image_scaled = 255.0 - image_scaled
                    depth_obs_image = image_scaled.astype(np.uint8)
                    print(
                        f"[Mock] depth_obs_image shape={depth_obs_image.shape}, "
                        f"min_dist={min_distance_to_obstacles:.3f}"
                    )
                    dxy, dphi = try_compute_relative()
                    if dxy is not None:
                        threading.Thread(
                            target=send_infer_and_store,
                            args=(float(dxy), float(dphi), depth_obs_image.copy()),
                            daemon=True,
                        ).start()
                else:
                    print(
                        f"[Mock] CS_RSP_DEPTH_STREAM payload size mismatch: "
                        f"got={len(payload)}, expect={INPUT_DIM*INPUT_DIM}"
                    )

            # Process packet logic (returns what to reply and delay)
            # resp_cmd, resp_data, is_float, delay = handle_packet_logic(cmd, payload)
            if cmd in [CS_GRANT_TOKEN]:
            # if cmd == CS_GRANT_TOKEN:
                resp_cmd = CS_RSP_STALL
                resp_data = []
                is_float = False
                delay = 0.001
            # If response expected, enqueue it with delay
            #if resp_cmd is not None:
                enqueue_response(delay, resp_cmd, resp_data, is_float)
                print(f"[Mock] response enqueue with delay: cmd=0x{resp_cmd:02X}, len={len(resp_data)}")
            
            with flag_lock:
                if cmd == 0x13:
                    depth_flag = 1
                    print("detect cmd = 0x13, flip flag")
                if cmd == 0x15:
                    depth_stream_flag = 1
                    print("detect cmd = 0x15, flip flag")
                if cmd == 0x31:
                    x_flag = 1
                    print("detect cmd = 0x31, flip flag")
                if cmd == 0x33:
                    y_flag = 1
                    print("detect cmd = 0x33, flip flag")
                if cmd == 0x35:
                    phi_flag = 1
                    print("detect cmd = 0x35, flip flag")
                if cmd == 0x11:
                    # combine into a RGB image
                    #######################
                    row_bytes = b''.join([p.to_bytes(4, 'little') for p in payload])
                    row_np = np.frombuffer(row_bytes, dtype=np.uint8)[:INPUT_DIM*3]
                    row_np = row_np.reshape((1, INPUT_DIM, 3))
                    img_rows.append(row_np)
                    #######################
                    img_row = img_row + 1
                    if img_row >= INPUT_DIM:
                        img_row = 0
                        img_flag = 1
                        print("detect cmd = 0x11, flip flag")
                        img_np = np.concatenate(img_rows, axis=0)
                        img_rows = []
                        cv2.imwrite(f"received_img_{img_index:03d}.png", img_np)         
                        threading.Thread(
                            target=sr_and_save,
                            args=(img_np.copy(), img_index),
                            daemon=True
                        ).start()
                        img_index = img_index + 1
                        print(f"Image shape: {img_np.shape}")
                        store_png = img_np
            
            # Always attempt to flush any due responses (also in case some earlier responses are ready)
            #process_latency_queue(conn)
            
            # Keep looping, do not close connection unless peer closes
    except Exception as e:
        print("[Mock] Exception in handle_client:", e)
    finally:
        try:
            conn.close()
        except:
            pass
        print("[Mock] disconnected")

def latency_loop(conn):
    while True:
        try:
            process_latency_queue(conn)
        except Exception as e:
            print("[Mock] latency thread error:", e)
        time.sleep(0.001)

def send_mock():
    global depth_flag, img_flag, depth_stream_flag, x_flag, y_flag, phi_flag
    # resp_cmd, resp_data, is_float, delay, enqueue_response()
    # send_arm
    time.sleep(5)
    resp_cmd = CS_REQ_ARM
    resp_data = []
    is_float = False
    delay = 0.001
    enqueue_response(delay, resp_cmd, resp_data, is_float)
    time.sleep(10)

    # send_takeoff
    resp_cmd = CS_REQ_TAKEOFF
    resp_data = []
    is_float = False
    delay = 0.005
    enqueue_response(delay, resp_cmd, resp_data, is_float)
    time.sleep(10)
    
    # cnt_img = 0
    # while cnt_img < 8:
    #     # send_depth_req
    #     print("========== send depth req ==========")
    #     resp_cmd = CS_REQ_DEPTH
    #     resp_data = []
    #     is_float = False
    #     delay = 0.01
    #     enqueue_response(delay, resp_cmd, resp_data, is_float)
    #     # time.sleep(10) 
        
    #     # recv_depth
    #     print("========== wait depth rsp ==========")
    #     while True:
    #         with flag_lock:
    #             if depth_flag == 1:
    #                 print("flag break")
    #                 depth_flag = 0
    #                 break
    #     time.sleep(2) 

    #     # send_img_req
    #     print("========== send img req ==========")
    #     resp_cmd = CS_REQ_IMG
    #     resp_data = []
    #     is_float = False
    #     delay = 0.02
    #     enqueue_response(delay, resp_cmd, resp_data, is_float)
    #     # time.sleep(10)
        
    #     # recv_img
    #     print("========== wait img rsp ==========")
    #     while True:
    #         with flag_lock:
    #             if img_flag == 1:
    #                 img_flag = 0
    #                 break
    #     time.sleep(2) 

    #     # fly to next waypoint
    #     print("========== send waypoint req ==========")
    #     resp_cmd = CS_RSP_WAYPOINT
    #     if cnt_img == 0: y_next = 10
    #     elif cnt_img == 1: y_next = 15
    #     elif cnt_img == 2: y_next = 10
    #     elif cnt_img == 3: y_next = 0
    #     elif cnt_img == 4: y_next = -5
    #     elif cnt_img == 5: y_next = -10
    #     elif cnt_img == 6: y_next = -5
    #     elif cnt_img == 7: y_next = 0
    #     resp_data = [(cnt_img+1)*10, y_next, -4, 1]
    #     is_float = True
    #     delay = 0.01
    #     enqueue_response(delay, resp_cmd, resp_data, is_float)
    #     time.sleep(2) 

    # cnt_img = 0
    # while cnt_img < 8:
    while True:
        # send x y phi req
        print("========== send coor req ==========")
        resp_data = []
        is_float = False
        delay = 0.01
        resp_cmd = CS_REQ_COOR_X
        enqueue_response(delay, resp_cmd, resp_data, is_float)
        while True:
            with flag_lock:
                if x_flag == 1:
                    print("flag break")
                    x_flag = 0
                    break
        resp_cmd = CS_REQ_COOR_Y
        enqueue_response(delay, resp_cmd, resp_data, is_float)
        while True:
            with flag_lock:
                if y_flag == 1:
                    print("flag break")
                    y_flag = 0
                    break
        resp_cmd = CS_REQ_PHI
        enqueue_response(delay, resp_cmd, resp_data, is_float)
        while True:
            with flag_lock:
                if phi_flag == 1:
                    print("flag break")
                    phi_flag = 0
                    break
        time.sleep(1)

        # send_depth stream req
        print("========== send depth stream req ==========")
        resp_cmd = CS_REQ_DEPTH_STREAM
        resp_data = []
        is_float = True
        delay = 0.01
        enqueue_response(delay, resp_cmd, resp_data, is_float)
        
        # recv_depth
        print("========== wait depth rsp ==========")
        while True:
            with flag_lock:
                if depth_stream_flag == 1:
                    print("flag break")
                    depth_stream_flag = 0
                    break
        time.sleep(5)

        # fly to next waypoint
        print("========== send waypoint req ==========")
        resp_cmd = CS_RSP_WAYPOINT
        if cnt_img == 0: y_next = 10
        elif cnt_img == 1: y_next = 15
        elif cnt_img == 2: y_next = 10
        elif cnt_img == 3: y_next = 0
        elif cnt_img == 4: y_next = -5
        elif cnt_img == 5: y_next = -10
        elif cnt_img == 6: y_next = -5
        elif cnt_img == 7: y_next = 0
        resp_data = [(cnt_img+1)*10, y_next, -4, 1]
        is_float = True
        delay = 0.01
        enqueue_response(delay, resp_cmd, resp_data, is_float)
        time.sleep(2) 


        cnt_img = cnt_img + 1
        print(f"[Mock] Complete {cnt_img} IMG")
    # end the test
    # os.kill(os.getpid(), signal.SIGTERM)
    
# ======= Proxy Layer =======
def recv_full_packet(sock):
    header = sock.recv(8)
    if not header:
        return None
    cmd = int.from_bytes(header[:4], 'little')
    num_bytes = int.from_bytes(header[4:], 'little')
    payload = b''
    while len(payload) < num_bytes:
        chunk = sock.recv(num_bytes - len(payload))
        if not chunk:
            break
        payload += chunk
    return header + payload

def pipe(src, dst, tag):
    try:
        while True:
            packet = recv_full_packet(src)
            print(f"[{tag}] packet received")
            if not packet:
                print(f"[{tag}] EOF")
                break
            print(f"[{tag}] prepare to send packet")
            try:
                dst.sendall(packet)
            except Exception as e:
                print(f"[{tag}] sendall error at {time.strftime('%H:%M:%S')} : {e.__class__.__name__}: {e}")
                traceback.print_exc()
                try:
                    dst.close()
                except Exception:
                    pass
                try:
                    src.close()
                except Exception:
                    pass
                print(f"[{tag}] dst peer disconnected, closing both ends")
                return
            try:
                src_name = src.getsockname()
            except OSError:
                src_name = "(disconnected)"
            try:
                dst_name = dst.getpeername()
            except OSError:
                dst_name = "(disconnected)"
            print(f"[{tag}] forwarding packet len={len(packet)} / src={src_name} -> dst={dst_name}")
    except Exception as e:
        print(f"[{tag}] unexpected error: {e}")
        traceback.print_exc()
        try:
            dst.close()
        except Exception:
            pass
        try:
            src.close()
        except Exception:
            pass

def proxy_loop(client_sock):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as mock_sock:
            mock_sock.connect((HOST, SYNC_PORT))
            print("[Proxy] connected mock (10002)")
            #mock_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            #client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            t1 = threading.Thread(target=pipe, args=(client_sock, mock_sock, "RoSE->Mock"), daemon=True)
            t2 = threading.Thread(target=pipe, args=(mock_sock, client_sock, "Mock->RoSE"), daemon=True)
            t1.start()
            t2.start()

            t1.join()
            t2.join()
            print("[Proxy] Connection pair closed cleanly.")
    except Exception as e:
        print(f"[Proxy] exception: {e}")
    finally:
        try:
            client_sock.close()
        except Exception:
            pass
        print("[Proxy] client_sock closed")

# ======= MAIN =======
def main():
    kill_port(ROSE_PORT)
    # print("[Debug] kill_port1")
    kill_port(SYNC_PORT)
    # print("[Debug] kill_port2")
    wait_for_port_release(ROSE_PORT)
    wait_for_port_release(SYNC_PORT)
    print("[Mock] Ports cleared, starting mock server and proxy...")

    # Start Mock Server
    def mock_thread():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, SYNC_PORT))
            s.listen()
            print(f"[Mock] listening on {HOST}:{SYNC_PORT}")
            while True:
                conn, _ = s.accept()
                threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

    threading.Thread(target=mock_thread, daemon=True).start()

    # Start Proxy
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, ROSE_PORT))
        s.listen()
        print(f"[Proxy] listening on {HOST}:{ROSE_PORT} -> forwarding to {SYNC_PORT}")
        while True:
            client_sock, addr = s.accept()
            print("[Proxy] new RoSE client", addr)
            threading.Thread(target=proxy_loop, args=(client_sock,), daemon=True).start()

if __name__ == "__main__":
    main()

