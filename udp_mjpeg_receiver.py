import socket
import struct
import io
from PIL import Image

LISTEN_IP = '0.0.0.0'
LISTEN_PORT = 5000
BUFFER_SIZE = 65536

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LISTEN_IP, LISTEN_PORT))
print(f"Listening for MJPEG UDP on {LISTEN_IP}:{LISTEN_PORT}")

# store partial frames: frame_id -> {total:int, chunks:dict}
frames = {}

while True:
    data, addr = sock.recvfrom(BUFFER_SIZE)
    if len(data) < 8:
        continue
    # header: 4-byte frame_id, 2-byte total, 2-byte idx
    fid, total, idx = struct.unpack('>IHH', data[:8])
    chunk = data[8:]
    entry = frames.get(fid)
    if entry is None:
        entry = {'total': total, 'chunks': {}}
        frames[fid] = entry
    entry['chunks'][idx] = chunk
    # check if complete
    if len(entry['chunks']) == entry['total']:
        # reassemble
        parts = [entry['chunks'][i] for i in range(entry['total'])]
        payload = b''.join(parts)
        try:
            img = Image.open(io.BytesIO(payload))
            img.show()
        except Exception as e:
            print('Failed to decode image', e)
        # cleanup
        del frames[fid]
