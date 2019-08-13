import concurrent.futures
from functools import partial

from config import CPU_WORKERS, N_FFT_HPSS_2, N_HOP_HPSS_2, SR_HPSS, RNN_INPUT_SIZE_VOICE_ACTIVATION
from util.leglaive.audio_processor import process_single_audio
import argparse
import numpy as np

from util.leglaive.config_rnn import *
from librosa.core import frames_to_time


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input', type=str)
    args = parser.parse_args()
    return args


def predict_songs(model_name, list_of_filenames, cache=True):
    import os
    import sys

    input_mels = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_WORKERS) as executor:
        _process_single_audio = partial(process_single_audio, res_list=input_mels)
        executor.map(_process_single_audio, list_of_filenames)

    from keras.models import load_model

    # set gpu number
    from keras import backend

    if len(backend.tensorflow_backend._get_available_gpus()) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # load mode
    loaded_model = load_model(str(WEIGHTS_DIR / 'rnn_{}.h5'.format(model_name)))
    print("loaded model")
    print(loaded_model.summary())

    total_x = []

    x = input_mels
    for i in range(0, x.shape[2] - RNN_INPUT_SIZE_VOICE_ACTIVATION, 1):
        x_segment = x[:, :, i: i + RNN_INPUT_SIZE_VOICE_ACTIVATION]
        total_x.append(x_segment)

    total_x = np.array(total_x).swapaxes(0, 1)  # final_shape: (#_file, #_hops, #_mel_filters, #_window)

    try:
        mean_std = np.load(WEIGHTS_DIR / 'train_mean_std_{}.npy'.format(model_name))
        mean = mean_std[0]
        std = mean_std[1]
    except Exception:
        print("mean, std not found")
        sys.exit()

    total_x_norm = (total_x - mean) / std
    total_x_norm = np.swapaxes(total_x_norm, 2, 3)

    x_test = total_x_norm
    y_pred = loaded_model.predict(x_test, verbose=1)  # Shape=(total_frames,)
    print('info: predicted with shape {}'.format(y_pred.shape))
    print(y_pred)
    return y_pred


def predict_song(model_name, filename, cache=True):
    """
    Predict Voice Activity Regions for a given song.

    :param model_name: name of the trained model
    :param filename:  path to the music file to be predicted
    :param cache: flag to optimize heavy operations with caching in disk
    :return: Prediction: Raw probability for each frame of the MFCC of the input song with overlapping by the RNN settings
    """
    audio_name = filename.parts[-1]
    audio_name_prefix = '.'.join(filename.parts[:-1])
    cache_filename = PREDICTIONS_DIR / '{}.{}.{}.npy'.format(audio_name_prefix, audio_name, model_name)
    try:
        if not cache:
            raise IOError
        y_pred = np.load(cache_filename)
    except IOError:
        import os
        import sys

        input_mel = process_single_audio(filename, cache=True)

        from keras.models import load_model

        # set gpu number
        from keras import backend

        if len(backend.tensorflow_backend._get_available_gpus()) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # load mode
        loaded_model = load_model(str(WEIGHTS_DIR / 'rnn_{}.h5'.format(model_name)))
        print("loaded model")
        print(loaded_model.summary())

        total_x = []

        x = input_mel
        for i in range(0, x.shape[1] - RNN_INPUT_SIZE_VOICE_ACTIVATION, 1):
            x_segment = x[:, i: i + RNN_INPUT_SIZE_VOICE_ACTIVATION]
            total_x.append(x_segment)

        total_x = np.array(total_x)
        try:
            mean_std = np.load(WEIGHTS_DIR / 'train_mean_std_{}.npy'.format(model_name))
            mean = mean_std[0]
            std = mean_std[1]
        except Exception:
            print("mean, std not found")
            sys.exit()

        total_x_norm = (total_x - mean) / std
        total_x_norm = np.swapaxes(total_x_norm, 1, 2)

        x_test = total_x_norm
        y_pred = loaded_model.predict(x_test, verbose=1)  # Shape=(total_frames,)

        print(y_pred)
        np.save(cache_filename, y_pred) if cache else None
    return y_pred


def frame_level_predict(model_name, filename, cache=True, plot=False):
    """
    Predict Voice Activity Regions at a Frame Level for a given song.
    For each frame of the MFCC a Voice Detection Probability is predicted, then the output have shape: (n_frames, 1)

    :param model_name: name of the trained model
    :param filename:  path to the music file to be predicted
    :param cache: flag to optimize heavy operations with caching in disk
    :param plot: flag to plot MFCCs and SVD in an aligned plot if GUI available.
    :return: (Time, Predictions): SVD probabilities at frame level with time markings
    """
    audio_name = filename.parts[-1]
    audio_name_prefix = '.'.join(filename.parts[:-1])
    serialized_filename = PREDICTIONS_DIR / '{}.{}.{}.csv'.format(audio_name_prefix, audio_name, model_name)
    mel = process_single_audio(filename, cache=cache)

    try:
        if not cache:
            raise IOError
        data = np.loadtxt(serialized_filename, delimiter=',')
        time = data[0]
        frame_level_y_pred = data[1]
        print("info: loaded serialized prediction")
    except Exception:

        # transform raw predictions to frame level
        y_pred = predict_song(model_name, filename, cache=cache)
        aligned_y_pred = [[] for _ in range(mel.shape[1])]
        for first_frame_idx, window_prediction in enumerate(y_pred):
            # for each prediction
            for offset, frame_prediction in enumerate(window_prediction):
                # accumulate overlapped predictions in a list
                aligned_y_pred[first_frame_idx + offset].append(frame_prediction[0])

        frame_level_y_pred = []
        for _, predictions in enumerate(aligned_y_pred[:-1]):
            # reduce the overlapped predictions to a single value
            frame_level_y_pred.append(min(predictions))

        time = frames_to_time(range(len(frame_level_y_pred)), sr=SR_HPSS, n_fft=N_FFT_HPSS_2, hop_length=N_HOP_HPSS_2)
        np.savetxt(serialized_filename, np.asarray((time, frame_level_y_pred)), delimiter=",")
        print("info: saved serialized prediction")
    if plot:
        import matplotlib.pyplot as plt
        import librosa.display

        # plot stacked MFCCs
        plt.figure(figsize=(14, 5))
        plt.subplot(211)
        librosa.display.specshow(mel, sr=SR_HPSS, x_axis='time', y_axis='hz', hop_length=N_HOP_HPSS_2)

        # plot frame level predictions
        plt.subplot(313)
        plt.plot(time, frame_level_y_pred)
        plt.xlabel("Time")
        plt.ylabel("Singing Voice Activation")
        plt.show()
        print("info: plotted")
    print('info: done')
    return time, frame_level_y_pred


if __name__ == "__main__":
    args = init()
    x, y = frame_level_predict(args.model_name, args.input, cache=True)
    print("info: plotting frame-wize predictions: ")
    print(y)
    print("info: plotting frame timestamps [seconds]: ")
    print(x)
