import pickle
import os.path


def write_big_pickle(obj, file_path, max_bytes=2**31 - 1):
    # write
    bytes_out = pickle.dumps(obj)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def read_big_pickle(file_path, max_bytes=2**31 - 1):
    # read
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)
