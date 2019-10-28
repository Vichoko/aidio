# Aidio

Audio Artificial Intelligence Suite (Aidio) provides:
* Feature Extraction
    * Mel Frequency Spectral Coefficients (MFSC)
    * Mel Frequency Cesptral Coefficients (MFCC)
    * Double Harmonic-Percussive Signal Separation (HPSS)
    * Magnitude & Phase separation of STFT (MagPhase)
    * Singing Voice Detection (SVD) [1]
    * Singing Voice Separation (SVS) [2] [3]
* Classification Models
    * Deep Learning
        * ResNet
        * WaveNet
        * 
    * Classic ML
        * GMM
        

## Instalation
1. Copy source code
2. Install requirements.cpu.txt or requirements.gpu.txt depending on your system.

## Usage

Aidio can be used as a simple feature extractor, or as a whole classification pipeline.
The workflow can be a little bit tricky at the beggining but we try to mantain a standard and match other
libraris like PyTorch

### Feature Extraction

There are 2 types of feature extractors, depending of the source input data.

Here are the available feature extractors explained in detail:

* ```MelSpectralCoefficientsFeatureExtractor```: Audio file to single Numpy array file with MFSC.
* ```WindowedMelSpectralCoefficientsFeatureExtractor```: Audio file to many Numpy array file with MFSC. 
The input song is splitted in non-silent intervals, which are then windowed and finally transformed to MFSC.
* ```DoubleHPSSFeatureExtractor```:  Audio file to single Numpy Array file with 2xHPSS [1].
* ```VoiceActivationFeatureExtractor```: 2xHPSS to SVD. Uses LSTM as described by Leglaive et. al [1] to frame-level overlapped predictions of voice activations. 
The model included is pre-trained on [Jamendo](https://www.audiocontentanalysis.org/data-sets/). Each frame have ~218 predictions.
* ```MeanSVDFeatureExtractor```: SVD to SVD. Reduced the number of prediction for each frame from ~218 to 1 by mean pooling.
* ```SVDPonderatedVolumeFeatureExtractor```: SVD + Audio to Audio. Ponderate the 'volume' of each frame by the singing voice
probability to make a listenable SVD feature (Oyanedel et. al 2019)

#### Audio Data to Feature

When having audio data (as .ogg, .wav or .mp3, for example), is the easiest way to extract features.

You can use the following feature extractors directly:
* ```MelSpectralCoefficientsFeatureExtractor``` :  Mel Frequency Spectral Coefficients as Numpy Arrays (```.npy``` files)
* ```DoubleHPSSFeatureExtractor``` : Double Harmonic-Percussive Signal Separation as Numpy Arrays.
* ```WindowedMelSpectralCoefficientsFeatureExtractor```: MFSC over windows as Numpy Arrays.

You need to have the following directory scheme:

```
folder/
    song1.mp3
    song2.ogg
    ...
    labels.csv
```

In ```labels.csv``` you should map each filename to a set of meta-data, for example, the label you'd like to classify later.
With this scheme we can forget of more complex directory schemes where the label is encoded in folder names, for insttance.

For now on, all the Features extracted with Aidio will follow the same scheme, as is, you can easily pass from one feature extractor
to another, but we will explain this later in detail.

#### Feature to Feature

There are some data extractor that ingest audio files, but others ingest more complex sets of data, for example, the
```SVDPonderatedVolume``` Feature Extractor requires the ```Singing Voice Detection``` mapping aditionally to the audio 
files to work.

The pipelines are defined below in detail as they are needed to be followed step by step manually for now (in the future
we want to rework this logic to be more user friendly)

##### Singing Voice Detection Pipeline

The Singing Voice Detection Pipeline is defined by the following sequential feature extractions included in this suite:

1. ```DoubleHPSSFeatureExtractor```
2. ```VoiceActivationFeatureExtractor```
3. ```MeanSVDFeatureExtractor```
4. ```SVDPonderatedVolumeFeatureExtractor```

##### Singing Voice Separation Pipeline

###### Open-Unmix

Use ```SingingVoiceSeparationOpenUnmixFeatureExtractor``` directly.

##### U-Net

Deprecated, but functioning as OpenUnmix does the job faster and better.

## Dependencies

### Python

#### PIP

Stated in requirements.txt

#### Conda

Stated in comments inside requirements.txt

#### Other

##### MP3 Encoding

LAME is needed to Feature Extractors that exports to MP3.
lame (or lame.exe) should be set in PATH environment variable to work properly.

## Reference

* [1] Simon Leglaive, Romain Hennequin, and Roland Badeau. "Singing voice detection with deep recurrent neural network." [pdf](https://hal.archives-ouvertes.fr/hal-01110035/document)
* [2] Stöter et al. "Open-Unmix - A Reference Implementation for MusicSource Separation" [pdf](https://joss.theoj.org/papers/10.21105/joss.01667)
* [3] A. Jansson, et al. "SINGING VOICE SEPARATION WITH DEEP U-NET CONVOLUTIONAL NETWORKS." [pdf](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf)