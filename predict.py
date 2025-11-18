import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras

# ==== Параметри мел-спектру (як у тренуванні) ====
SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 256
DUR = 5.0   # сек
CLASSES = ["speech", "environment", "music"]

def load_mel(path, target_frames=None):
    """Завантажуємо аудіо і конвертуємо в mel-спектр з вирівнюванням по ширині"""
    y, _ = librosa.load(path, sr=SR, mono=True)

    # фіксована довжина в samples
    target_len = int(SR * DUR)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    mel = mel[..., None]  # (128, W, 1)

    # підганяємо по ширині моделі
    if target_frames:
        W = mel.shape[1]
        if W < target_frames:
            mel = np.pad(mel, ((0,0),(0,target_frames - W),(0,0)))
        else:
            mel = mel[:, :target_frames, :]

    return mel

def predict(audio_path, model_path="best.keras"):
    model = keras.models.load_model(model_path)
    target_frames = model.input_shape[2]  # автоматично підбираємо ширину

    mel = load_mel(audio_path, target_frames=target_frames)
    x = mel[np.newaxis, ...]  # (1,128,W,1)

    probs = model.predict(x, verbose=0)[0]
    label = CLASSES[int(np.argmax(probs))]
    return label, {c: float(probs[i]) for i, c in enumerate(CLASSES)}

# ----------------------------
if __name__ == "__main__":
    audio = input("Audio path: ")
    label, probs = predict(audio)
    print("\n--- RESULT ---")
    print("Label:", label)
    for k, v in probs.items():
        print(f"{k}: {v:.4f}")
