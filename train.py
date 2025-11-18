import os, glob, csv
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras

# ==== Параметри мел-спектрів ====
SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 256
DUR = 5.0   # сек
TARGET_FRAMES = 313


def compute_mel(path):
    # Завантаження і підгін по довжині на рівні сигналу (узгоджено з predict.py)
    y, _ = librosa.load(path, sr=SR, mono=True)
    target_len = int(SR * DUR)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP
    )

    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # Z-score нормалізація (як у predict.py)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    # Фіксуємо ширину (підстраховка)
    if mel.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - mel.shape[1]
        mel = np.pad(mel, ((0,0),(0,pad)), mode='constant')
    else:
        mel = mel[:, :TARGET_FRAMES]

    mel = mel[..., np.newaxis]  # (128, 313, 1)
    return mel

def load_esc50(root, limit=None):
    """Побутові звуки."""
    data, labels = [], []
    csv_path = os.path.join(root, "meta", "esc50.csv")

    with open(csv_path, encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        if limit:
            reader = reader[:limit]

        for row in reader:
            p = os.path.join(root, "audio", row["filename"])
            data.append(compute_mel(p))
            labels.append(1)    # env
    return data, labels

def load_music():
    folder = r"music_wav"
    files = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.au"):
        files += glob.glob(os.path.join(folder, ext))

    X, y = [], []
    for p in files:
        try:
            X.append(compute_mel(p))
            y.append(2)   # label: 2 = music
        except:
            pass
    return X, y


def load_speech(limit=None):
    folder = r"cv-corpus-23.0-2025-09-05\uk\clips"
    files = glob.glob(os.path.join(folder, "*.wav"))
    if limit:
        files = files[:limit]

    X, y = [], []
    for p in files:
        try:
            X.append(compute_mel(p))
            y.append(0)   # label: 0 = speech
        except:
            pass
    return X, y

def build_model(width, num_classes):
    inp = keras.Input(shape=(N_MELS, width, 1))
    x = inp
    for c in (32, 64, 128):
        x = keras.layers.Conv2D(c, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D(2)(x)
        x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)

# ==== Основний пайплайн ====
def main():
    # Шляхи
    ESC = "ESC-50-master"
    SPEECH = "speech"
    MUSIC = "music"

    print("Loading datasets...")
    env_X, env_y = load_esc50(ESC, limit=800)
    sp_X, sp_y = load_speech(limit=1500)
    mu_X, mu_y = load_music()

    # Об’єднання
    X = env_X + sp_X + mu_X
    Y = env_y + sp_y + mu_y
    X, Y = np.array(X), np.array(Y)

    # Ширина мел-спектру
    W = X[0].shape[1]

    # Трен/Вал split
    idx = np.random.permutation(len(X))
    split = int(0.85 * len(X))
    X_train, X_val = X[idx[:split]], X[idx[split:]]
    y_train, y_val = Y[idx[:split]], Y[idx[split:]]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(2000).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

    # ==== Модель ====
    model = build_model(W, num_classes=3)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="val_acc"), "accuracy"]
    )

    # Ваги класів для балансування (0: speech, 1: env, 2: music)
    counts = np.bincount(Y, minlength=3)
    total = len(Y)
    class_weight = {i: float(total / (3 * counts[i])) if counts[i] > 0 else 1.0 for i in range(3)}
    print("Class counts:", {i:int(counts[i]) for i in range(3)})
    print("Class weights:", class_weight)

    ckpt = keras.callbacks.ModelCheckpoint("best.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
    es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    rlrop = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)

    print("Training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight=class_weight,
        callbacks=[ckpt, es, rlrop]
    )

    print("Done. Saved to best.keras")

if __name__ == "__main__":
    main()
