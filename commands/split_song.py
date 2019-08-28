import pathlib

import librosa
import numpy as np

from config import SR, FEATURES_DATA_PATH
from features import SVDPonderatedVolumeFeatureExtractor

if __name__ == '__main__':
    """
    Takes a song and splits the song by it's silences, given by the top_db parameter.
    """
    # parameters
    in_path = FEATURES_DATA_PATH / SVDPonderatedVolumeFeatureExtractor.feature_name
    out_path = FEATURES_DATA_PATH / 'test'
    song_name = '112_Fait_Ange.2hpss.voice_activation.mean_svd.svd_ponderated_volume.wav'
    # get song and split
    wav, sr = librosa.core.load(in_path / song_name, sr=SR)
    intervals = librosa.effects.split(
        wav,
        top_db=80
    )
    # export intervals as new songs (wav)
    for idx, interval in enumerate(intervals):
        if interval[1] - interval[0] < SR:
            # if length is lesser that 1 second, discard interval
            continue
        librosa.output.write_wav(out_path / '{}-{}.wav'.format(song_name, idx), wav[interval[0]:interval[1]], sr=SR)
