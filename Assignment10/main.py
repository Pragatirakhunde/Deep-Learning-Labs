"""
Deep Learning Practical 02: Audio classification with STT and TTS.
Uses SpeechRecognition (Google Web Speech API) for speech-to-text when online,
a pretrained YAMNet audio event classifier via TensorFlow Hub, and pyttsx3 for
offline text-to-speech. Includes a WAV-file classification mode.
"""
import argparse
import csv
import os
import time
import wave

import numpy as np

def load_audio_mono_16k(path):
    import soundfile as sf
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    peak = np.max(np.abs(audio)) if audio.size else 0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32), 16000

def classify_audio_file(path):
    import tensorflow as tf
    import tensorflow_hub as hub
    audio, sr = load_audio_mono_16k(path)
    if audio.size == 0:
        raise ValueError("The audio file is empty.")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    scores, embeddings, spectrogram = model(tf.convert_to_tensor(audio, dtype=tf.float32))
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    class_map_path = hub.resolve("https://tfhub.dev/google/yamnet/1") + "/assets/yamnet_class_map.csv"
    labels = []
    with open(class_map_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(row["display_name"])
    top_indices = np.argsort(mean_scores)[::-1][:5]
    results = [(labels[i] if i < len(labels) else f"class_{i}", float(mean_scores[i])) for i in top_indices]
    return results

def speech_to_text(language='en-US'):
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for background noise... please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Speak now (up to around 10 seconds), then pause.")
        audio = recognizer.listen(source, timeout=8, phrase_time_limit=12)
    try:
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        return "Speech could not be understood. Please try again."
    except sr.RequestError as exc:
        return f"STT service error (check internet connection): {exc}"

def speak(text):
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def save_metrics(mode, elapsed, result_text, output_path):
    exists = os.path.exists(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "mode", "elapsed_seconds", "result"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), mode, round(elapsed, 3), result_text])

def main():
    parser = argparse.ArgumentParser(description="Audio classification with STT and TTS.")
    parser.add_argument("--mode", choices=["all", "speech", "classify"], default="all",
                        help="all: microphone STT + TTS; speech: STT + TTS; classify: classify a WAV/audio file")
    parser.add_argument("--audio", default="", help="Path to audio file for classification (WAV recommended)")
    parser.add_argument("--language", default="en-US", help="STT language, e.g. en-US or en-IN")
    args = parser.parse_args()
    os.makedirs("outputs", exist_ok=True)

    if args.mode in ("all", "speech"):
        start = time.perf_counter()
        try:
            recognized = speech_to_text(args.language)
        except Exception as exc:
            recognized = f"Microphone/STT error: {exc}"
        elapsed = time.perf_counter() - start
        print("\\nRecognized text:", recognized)
        save_metrics("STT", elapsed, recognized, "outputs/runtime_metrics.csv")
        try:
            speak(recognized)
            print("TTS playback completed.")
        except Exception as exc:
            print("TTS error:", exc)
        if args.mode == "speech":
            return

    if args.mode == "all":
        print("\\nOptional audio classification: run `python main.py --mode classify --audio path_to_file.wav`.")
        print("YAMNet is a pretrained audio event classifier (e.g., speech, music, dog bark, siren).")
        return

    if args.mode == "classify":
        if not args.audio:
            parser.error("--audio is required when --mode classify is selected")
        start = time.perf_counter()
        try:
            results = classify_audio_file(args.audio)
            elapsed = time.perf_counter() - start
            print("\\nTop audio event predictions:")
            for label, score in results:
                print(f"{label}: {score:.4f}")
            result_text = "; ".join(f"{label}:{score:.4f}" for label, score in results)
            save_metrics("YAMNet classification", elapsed, result_text, "outputs/runtime_metrics.csv")
            print(f"Classification runtime: {elapsed:.2f} seconds")
            print("Saved runtime record to outputs/runtime_metrics.csv")
        except Exception as exc:
            print("Audio classification failed:", exc)
            print("Check that the audio file exists, dependencies are installed, and internet is available for first-time model download.")

if __name__ == "__main__":
    main()
