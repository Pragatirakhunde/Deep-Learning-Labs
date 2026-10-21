# Assignment 10 - Audio Classification with STT and TTS

## Overview
This app demonstrates:
- Speech-to-Text (STT): captures microphone audio and transcribes speech using SpeechRecognition's Google Web Speech service (internet required).
- Text-to-Speech (TTS): reads the recognized text aloud using pyttsx3 (uses installed system voices; typically works offline).
- Audio event classification: uses the pretrained YAMNet model to predict common sound classes from an audio file, such as speech, music, dog bark, or siren.

This is a practical integration of pretrained deep-learning audio classification with speech interfaces. STT/TTS are separate components; the audio event classifier is YAMNet.

## Environment
Use Python 3.10 or 3.11 in VS Code. TensorFlow wheels and hardware support vary by OS/Python version; Python 3.10 is a conservative option for compatibility.

### Windows PowerShell
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run STT + TTS
```powershell
python main.py
```
Speak after the prompt, then pause. The recognized transcript is printed and spoken aloud. The STT service requires an internet connection and microphone access.

For Indian English, try:
```powershell
python main.py --mode speech --language en-IN
```
## Run audio classification
Use a local audio file (WAV is recommended):
```powershell
python main.py --mode classify --audio "dog_bark.wav"
```
The app does not download an audio dataset. It classifies only the audio file you provide. The first run downloads the pretrained YAMNet model weights from TensorFlow Hub and needs internet access. Audio is converted to mono 16 kHz. Top five predicted event classes and scores are printed. Runtime records are saved to `outputs/runtime_metrics.csv`.

## Evaluation metrics
- **Top-1 predicted class / score:** highest-scoring YAMNet class and its model score.
- **Top-5 predictions:** useful for inspecting ambiguity among sound classes.
- **Runtime (seconds):** measures end-to-end classification duration (includes model download/loading on the first run, so use later runs for timing).
- **STT word error rate (WER):** if you have a known reference transcript, calculate WER externally or extend the app to compare reference and recognized text. This starter records transcript/runtime but does not claim WER without reference text.
- **Classification accuracy / precision / recall / F1:** requires labelled evaluation audio. This project does not invent these metrics; record a test set's true labels and compare predictions if your instructor requires dataset-level metrics.

## Troubleshooting
- Microphone errors: confirm Windows microphone permission and select a working input device.
- STT service errors: check internet connectivity.
- TTS voice error: confirm Windows speech voices are installed; try `pyttsx3` from the activated environment.
- TensorFlow installation errors: use Python 3.10, upgrade pip, and install TensorFlow according to official compatibility guidance.
- Audio decoding errors: convert the file to a standard PCM WAV (mono/stereo is accepted; resampling is handled).
