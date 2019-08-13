''' Preprocessing audio files to mel features '''
import numpy as np
import librosa

from config import N_FFT_HPSS_1, N_HOP_HPSS_1, N_FFT_HPSS_2, N_HOP_HPSS_2, SR_HPSS
from util.leglaive.config_rnn import *
from util.leglaive.audio import ono_hpss, log_melgram


def process_single_audio(audio_file, cache=False, res_list=None):
    ''' Compute double stage HPSS for the given audio file
    Args : 
        audio_file : path to audio file 
    Return :
        mel_D2_total : concatenated melspectrogram of percussive, harmonic components of double stage HPSS. Shape=(2 * n_bins, total_frames) ex. (80, 2004) 
    '''
    audio_name = audio_file.parts[-1]
    audio_name_prefix = '.'.join(audio_file.parts[:-1])
    cache_filename = MEL_CACHE_DIR / '{}.{}.mel.npy'.format(audio_name_prefix, audio_name)
    try:
        if not cache:
            raise IOError
        mel_total = np.load(cache_filename)
    except IOError:
        audio_src, _ = librosa.load(audio_file, sr=SR_HPSS)
        # Normalize audio signal
        audio_src = librosa.util.normalize(audio_src)
        # first HPSS
        D_harmonic, D_percussive = ono_hpss(audio_src, N_FFT_HPSS_1, N_HOP_HPSS_1)
        # second HPSS
        D2_harmonic, D2_percussive = ono_hpss(D_percussive, N_FFT_HPSS_2, N_HOP_HPSS_2)

        assert D2_harmonic.shape == D2_percussive.shape
        print(D2_harmonic.shape, D2_percussive.shape)

        # compute melgram
        mel_harmonic = log_melgram(D2_harmonic, SR_HPSS, N_FFT_HPSS_2, N_HOP_HPSS_2, N_MELS)
        mel_percussive = log_melgram(D2_percussive, SR_HPSS, N_FFT_HPSS_2, N_HOP_HPSS_2, N_MELS)
        # concat
        mel_total = np.vstack((mel_harmonic, mel_percussive))
        
        np.save(cache_filename, mel_total)

    print(mel_total.shape)
    if res_list:
        res_list.append(mel_total)
    return mel_total
