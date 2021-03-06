import numpy as np
import matplotlib.pyplot as plt

from config import FEATURES_DATA_PATH
from util.nparray_util import pad_this

if __name__ == "__main__":
    x = np.load(FEATURES_DATA_PATH / '316_Jeth_Stan.2hpss.voice_activation.npy', allow_pickle=True)  # ashes to ashes
    activations = x[1]
    time = x[0]
    max_len = np.max([len(array) for array in activations])
    a = pad_this(max_len, activations)
    a = a.swapaxes(0, 1)

    fig, ax = plt.subplots(1, 1)
    first_sample = 0
    last_sample = 5000
    ax.imshow(a[:, first_sample:last_sample], cmap='jet',
              interpolation='none')  # , extent=[time[first_sample], time[last_sample], max_len, 0])

    number_of_samples = last_sample - first_sample + 1
    x_ticks = np.asarray([first_sample + int((number_of_samples - 1) * tick) for tick in np.arange(0, 1, 1.0/20)])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(np.round(time[first_sample + x_ticks].astype(int), 0))
    plt.show()
    print('info: done')
