import argparse
import io
import warnings
from contextlib import redirect_stderr
from pathlib import Path

import librosa
import norbert
import numpy as np
import resampy
import scipy.signal
import torch
import tqdm

import util.open_unmix.model as model
import util.open_unmix.utils as utils
from config import OUNMIX_RESIDUAL_MODEL, OUNMIX_MODEL, OUNMIX_ALPHA, OUNMIX_TARGETS, OUNMIX_NITER, \
    OUNMIX_SOFTMAX, OUNMIX_SAMPLE_RATE, MODELS_DATA_PATH


def load_model(target, model_name='umxhq', device='cpu'):
    """
    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)
    """
    from features import SingingVoiceSeparationOpenUnmixFeatureExtractor
    model_path = Path(MODELS_DATA_PATH / SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name / model_name).expanduser()
    if not model_path.exists():
        # model path does not exist, use hubconf model
        try:
            # disable progress bar
            err = io.StringIO()
            with redirect_stderr(err):
                return torch.hub.load(
                    'sigsep/open_unmix',
                    model_name,
                    target=target,
                    device=device,
                    pretrained=True
                )
            print(err.getvalue())
        except AttributeError:
            raise NameError('Model does not exist on torchhub')
            # assume model is a path to a local model_name direcotry
    else:
        target_model_path = next(Path(model_path).glob("%s*.pth" % target))
        state = torch.load(
            target_model_path,
            map_location=device
        )

        max_bin = utils.bandwidth_to_max_bin(
            rate=44100,
            n_fft=4096,
            bandwidth=16000
        )

        # load open unmix model
        unmix = model.OpenUnmix(
            n_fft=4096,
            n_hop=1024,
            nb_channels=2,
            hidden_size=512,
            max_bin=max_bin
        )

        unmix.load_state_dict(state)
        unmix.stft.center = True
        unmix.eval()
        unmix.to(device)
        return unmix


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def separate(
        audio,
        targets,
        model_name='umxhq',
        niter=1, softmask=False, alpha=1.0,
        residual_model=False, device='cpu'
):
    """
    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

    model_name: str
        name of torchhub model or path to model folder, defaults to `umxhq`

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary of all restimates as performed by the separation model.

    """
    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)
    print('debug: separating audio tensor with shape {}'.format(audio_torch.size()))

    source_names = []
    V = []
    torch.cuda.empty_cache()
    for j, target in enumerate(tqdm.tqdm(targets)):
        with torch.no_grad():
            unmix_target = load_model(
                target=target,
                model_name=model_name,
                device=device
            )
            Vj = unmix_target(audio_torch).cpu().detach().numpy()
        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj ** alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, ...])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    X = unmix_target.stft(audio_torch).detach().cpu().numpy()
    # convert to complex numpy type
    X = X[..., 0] + X[..., 1] * 1j
    X = X[0].transpose(2, 1, 0)

    if residual_model or len(targets) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += (['residual'] if len(targets) > 1
                         else ['accompaniment'])

    Y = norbert.wiener(V, X.astype(np.complex128), niter,
                       use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            n_fft=unmix_target.stft.n_fft,
            n_hopsize=unmix_target.stft.n_hop
        )
        estimates[name] = audio_hat.T

    return estimates


def inference_args(parser, remaining_args):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    inf_parser.add_argument(
        '--softmask',
        dest='softmask',
        action='store_true',
        help=('if enabled, will initialize separation with softmask.'
              'otherwise, will use mixture phase with spectrogram')
    )

    inf_parser.add_argument(
        '--niter',
        type=int,
        default=1,
        help='number of iterations for refining results.'
    )

    inf_parser.add_argument(
        '--alpha',
        type=int,
        default=1,
        help='exponent in case of softmask separation'
    )

    inf_parser.add_argument(
        '--samplerate',
        type=int,
        default=44100,
        help='model samplerate'
    )

    inf_parser.add_argument(
        '--residual-model',
        action='store_true',
        help='create a model for the residual'
    )
    return inf_parser.parse_args()


def separate_music_file(
        input_file,
        device,
        targets=OUNMIX_TARGETS,
        model=OUNMIX_MODEL,
        residual_model=OUNMIX_RESIDUAL_MODEL,
        alpha=OUNMIX_ALPHA,
        niter=OUNMIX_NITER,
        softmask=OUNMIX_SOFTMAX,
        sample_rate=OUNMIX_SAMPLE_RATE

):
    """

    :param input_file:
    :param device:
    :param targets:
    :param model:
    :param residual_model:
    :param alpha:
    :param niter:
    :param softmask:
    :return: `dict` [`str`, `np.ndarray`]
        dictionary of all restimates as performed by the separation model.



    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

    model_name: str
        name of torchhub model or path to model folder, defaults to `umxhq`

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates:
    """
    # handling an input audio path
    audio, rate = librosa.core.load(input_file, mono=False, sr=sample_rate)
    if len(audio.shape) == 1:
        audio = audio.reshape(1, -1)
    audio = np.swapaxes(audio, 0, 1)

    if audio.shape[1] > 2:
        warnings.warn(
            'Channel count > 2! '
            'Only the first two channels will be processed!')
        audio = audio[:, :2]

    if rate != sample_rate:
        # resample to model samplerate if needed
        audio = resampy.resample(audio, rate, sample_rate, axis=0)

    if audio.shape[1] == 1:
        # if we have mono, let's duplicate it
        # as the input of OpenUnmix is always stereo
        audio = np.repeat(audio, 2, axis=1)

    estimates = separate(
        audio,
        targets=targets,
        model_name=model,
        niter=niter,
        alpha=alpha,
        softmask=softmask,
        residual_model=residual_model,
        device=device
    )
    return estimates


def separate_wav(
        audio,
        rate,
        device,
        targets=OUNMIX_TARGETS,
        model=OUNMIX_MODEL,
        residual_model=OUNMIX_RESIDUAL_MODEL,
        alpha=OUNMIX_ALPHA,
        niter=OUNMIX_NITER,
        softmask=OUNMIX_SOFTMAX,
        sample_rate=OUNMIX_SAMPLE_RATE

):
    """

    :param input_file:
    :param device:
    :param targets:
    :param model:
    :param residual_model:
    :param alpha:
    :param niter:
    :param softmask:
    :return: `dict` [`str`, `np.ndarray`]
        dictionary of all restimates as performed by the separation model.



    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

    model_name: str
        name of torchhub model or path to model folder, defaults to `umxhq`

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates:
    """
    # handling an input audio path
    if len(audio.shape) == 1:
        audio = audio.reshape(1, -1)
    audio = np.swapaxes(audio, 0, 1)

    if audio.shape[1] > 2:
        warnings.warn(
            'Channel count > 2! '
            'Only the first two channels will be processed!')
        audio = audio[:, :2]

    if rate != sample_rate:
        # resample to model samplerate if needed
        audio = resampy.resample(audio, rate, sample_rate, axis=0)

    if audio.shape[1] == 1:
        # if we have mono, let's duplicate it
        # as the input of OpenUnmix is always stereo
        audio = np.repeat(audio, 2, axis=1)

    estimates = separate(
        audio,
        targets=targets,
        model_name=model,
        niter=niter,
        alpha=alpha,
        softmask=softmask,
        residual_model=residual_model,
        device=device
    )
    return estimates


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='OSU Inference',
        add_help=False
    )

    parser.add_argument(
        'input',
        type=str,
        nargs='+',
        help='List of paths to wav/flac files.'
    )

    parser.add_argument(
        '--targets',
        nargs='+',
        default=['vocals', 'drums', 'bass', 'other'],
        type=str,
        help='provide targets to be processed. \
              If none, all available targets will be computed'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        help='Results path where audio evaluation results are stored'
    )

    parser.add_argument(
        '--model',
        default='umxhq',
        type=str,
        help='path to mode base directory of pretrained models'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    for input_file in args.input:
        # handling an input audio path
        audio, rate = librosa.core.load(input_file, mono=False, sr=args.samplerate)
        audio = np.swapaxes(audio, 0, 1)

        if audio.shape[1] > 2:
            warnings.warn(
                'Channel count > 2! '
                'Only the first two channels will be processed!')
            audio = audio[:, :2]

        if rate != args.samplerate:
            # resample to model samplerate if needed
            audio = resampy.resample(audio, rate, args.samplerate, axis=0)

        if audio.shape[1] == 1:
            # if we have mono, let's duplicate it
            # as the input of OpenUnmix is always stereo
            audio = np.repeat(audio, 2, axis=1)

        estimates = separate(
            audio,
            targets=args.targets,
            model_name=args.model,
            niter=args.niter,
            alpha=args.alpha,
            softmask=args.softmask,
            residual_model=args.residual_model,
            device=device
        )
        if not args.outdir:
            model_path = Path(args.model)
            if not model_path.exists():
                outdir = Path(Path(input_file).stem + '_' + args.model)
            else:
                outdir = Path(Path(input_file).stem + '_' + model_path.stem)
        else:
            outdir = Path(args.outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        # # write out estimates
        # for target, estimate in estimates.items():
        #     sf.write(
        #         outdir / Path(target).with_suffix('.wav'),
        #         estimate,
        #         args.samplerate
        #     )
