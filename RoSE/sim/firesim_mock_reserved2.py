import socket, threading, struct, random, time, os, subprocess, sys
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

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

INPUT_DIM = 56

# ======= Helper functions =======
def kill_port(port):
    try:
        pid = subprocess.check_output(f"lsof -t -i:{port}", shell=True, text=True).strip()
        if pid:
            print(f"Killing old process on port {port} (PID {pid})...")
            os.system(f"kill -9 {pid}")
            time.sleep(1)
    except subprocess.CalledProcessError:
        pass

def wait_for_port_release(port, timeout=10):
    for _ in range(timeout):
        result = os.system(f"netstat -tuln | grep {port} > /dev/null 2>&1")
        if result != 0:
            return True
        print(f"Port {port} still in use, waiting...")
        time.sleep(1)
    raise RuntimeError(f"Port {port} still not released after {timeout}s")

def recv_all(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_packet(conn, cmd, data, is_float=False):
    num_bytes = len(data) * 4 if data else 0
    conn.sendall(cmd.to_bytes(4, 'little'))
    conn.sendall(num_bytes.to_bytes(4, 'little'))
    for datum in data:
        if is_float:
            conn.sendall(struct.pack("f", float(datum)))
        else:
            conn.sendall(int(datum).to_bytes(4, "little", signed=False))

# ======= Mock FireSim Core Logic =======
cycle_counter = 0
latency_queue = []

def handle_packet(conn, cmd, payload):
    global cycle_counter
    resp_cmd = None
    resp_data = []
    is_float = False

    if cmd == CS_GRANT_TOKEN:
        print("[Mock] CS_GRANT_TOKEN received")
        resp_cmd = CS_GRANT_TOKEN
    elif cmd == CS_REQ_CYCLES:
        cycle_counter += 10000
        resp_cmd = CS_RSP_CYCLES
        resp_data = [cycle_counter]
    elif cmd == CS_REQ_WAYPOINT:
        resp_cmd = CS_RSP_WAYPOINT
        resp_data = [1.0, 2.0, 3.0]
        is_float = True
        print("[Mock] CS_REQ_WAYPOINT -> sending dummy waypoint")
    elif cmd == CS_REQ_ARM:
        resp_cmd = CS_GRANT_TOKEN
        print("[Mock] CS_REQ_ARM -> granted")
    elif cmd == CS_REQ_DISARM:
        resp_cmd = CS_GRANT_TOKEN
        print("[Mock] CS_REQ_DISARM -> granted")
    elif cmd == CS_REQ_TAKEOFF:
        resp_cmd = CS_GRANT_TOKEN
        print("[Mock] CS_REQ_TAKEOFF -> granted")
    elif cmd in [CS_REQ_IMG, CS_REQ_IMG_POLL]:
        resp_cmd = CS_RSP_IMG if cmd == CS_REQ_IMG else CS_RSP_IMG_POLL
        fake_data = np.random.rand(INPUT_DIM*INPUT_DIM*3).astype('float32').tolist()
        resp_data = fake_data
        is_float = True
        print(f"[Mock] Image request received (cmd=0x{cmd:02X})")
    elif cmd == CS_REQ_DEPTH:
        resp_cmd = CS_RSP_DEPTH
        resp_data = [10.0]  # dummy depth
        is_float = True
        print("[Mock] CS_REQ_DEPTH -> sending 10.0")
    elif cmd == CS_REQ_DEPTH_STREAM:
        resp_cmd = CS_RSP_DEPTH_STREAM
        resp_data = [10.0]*INPUT_DIM  # dummy stream
        is_float = True
        print("[Mock] CS_REQ_DEPTH_STREAM -> sending dummy stream")
    elif cmd == CS_SET_TARGETS:
        resp_cmd = CS_GRANT_TOKEN
        print("[Mock] CS_SET_TARGETS -> payload:", payload)
    else:
        print(f"[Mock] Unknown cmd=0x{cmd:02X}, payload={payload}")
        resp_cmd = CS_GRANT_TOKEN

    # 模拟延迟发送
    latency_queue.append((time.time()+0.01, (resp_cmd, resp_data, is_float)))

def process_latency_queue(conn):
    now = time.time()
    to_send = [item for item in latency_queue if item[0] <= now]
    for item in to_send:
        send_packet(conn, item[1][0], item[1][1], is_float=item[1][2])
        latency_queue.remove(item)
        print("[Mock] Sent delayed response for cmd=0x{:02X}".format(item[1][0]))

def handle_client(conn):
    print("[Mock] connected from", conn.getpeername())
    try:
        while True:
            header = recv_all(conn, 4)
            if not header:
                break
            cmd = int.from_bytes(header, "little", signed=False)

            num_bytes_data = recv_all(conn, 4)
            if not num_bytes_data:
                break
            num_bytes = int.from_bytes(num_bytes_data, "little", signed=False)

            payload = []
            if num_bytes > 0:
                raw = recv_all(conn, num_bytes)
                for i in range(0, len(raw), 4):
                    word = raw[i:i+4]
                    if cmd in [CS_SET_TARGETS, CS_RSP_WAYPOINT]:
                        payload.append(struct.unpack("f", word)[0])
                    else:
                        payload.append(int.from_bytes(word, "little", signed=False))

            handle_packet(conn, cmd, payload)
            process_latency_queue(conn)

    except Exception as e:
        print("[Mock] Exception:", e)
    finally:
        conn.close()
        print("[Mock] disconnected")

# ======= Proxy Layer =======
def pipe(src, dst, tag):
    try:
        while True:
            data = src.recv(BUFFER_SIZE)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        src.close()
        dst.close()
        print(f"{tag} closed")

def proxy_loop(client_sock):
    """RoSE <-> Mock"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as mock_sock:
        mock_sock.connect((HOST, SYNC_PORT))
        print("[Proxy] connected mock (10002)")
        threading.Thread(target=pipe, args=(client_sock, mock_sock, "[RoSE->Mock]"), daemon=True).start()
        pipe(mock_sock, client_sock, "[Mock->RoSE]")

# ======= MAIN =======
def main():
    kill_port(ROSE_PORT)
    kill_port(SYNC_PORT)
    wait_for_port_release(ROSE_PORT)
    wait_for_port_release(SYNC_PORT)

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

