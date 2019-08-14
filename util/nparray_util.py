import numpy as np


def pad_this(expected_len, array_2d_like):
    def process_element(array):
        if len(array) < expected_len:
            padding_len = expected_len - len(array)
            pad = np.pad(array, (0, padding_len), 'constant')
            return pad
        return np.asarray(array)

    return np.asarray([process_element(array) for array in array_2d_like])