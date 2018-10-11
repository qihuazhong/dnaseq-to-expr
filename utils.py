import pickle
import os.path

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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


def onehot_encode(seq):
    """
    one-hot encode a DNA sequence that contains alphabet of ["A", "C", "G", "T", "N"]
    A -> [1, 0, 0, 0]
    C -> [0, 1, 0, 0]
    G -> [0, 0, 1, 0]
    T -> [0, 0, 0, 1]
    N -> [0, 0, 0, 0]
    """

    seq_list = list(seq)
    label_encoder = LabelEncoder()
    label_encoder.fit(np.array(['A', 'C', 'G', 'T', 'N']))
    integer_encoded = label_encoder.transform(seq_list)
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int, categories=[range(5)])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return np.delete(onehot_encoded, -2, 1)