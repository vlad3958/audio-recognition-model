import argparse, os
import numpy as np, pandas as pd, librosa, tensorflow as tf
from tensorflow import keras


SR, DUR, N_MELS, N_FFT, HOP = 44100, 5, 128, 2205, 441
W = 1 + (SR * DUR - N_FFT) // HOP  # 496 for ESC-50 defaults


def build_dataset(paths, labels, batch_size=32, shuffle=False,
                  num_parallel_calls=tf.data.AUTOTUNE, deterministic=False):
    # Функція для завантаження та обробки аудіо файлу
    def _load(path_b):
        if isinstance(path_b, (bytes, bytearray)):
            path = path_b.decode("utf-8")
        else:
            path = path_b.numpy().decode("utf-8")
        y, _ = librosa.load(path, sr=SR, mono=True)
        y = np.pad(y, (0, SR * DUR - len(y)), mode="constant")[: SR * DUR]
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
        mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel = np.expand_dims(mel, -1)
        if mel.shape[1] < W:
            mel = np.pad(mel, ((0, 0), (0, W - mel.shape[1]), (0, 0)))
        else:
            mel = mel[:, :W, :]
        return mel
    # Мапер для tf.data
    def mapper(p, y):
        x = tf.py_function(_load, [p], tf.float32)
        x.set_shape((N_MELS, W, 1))
        return x, tf.cast(y, tf.int64)
    # Побудова датасету
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths))
    ds = ds.map(mapper, num_parallel_calls=num_parallel_calls)
    # Дозволяємо недетермінізм для більшої паралельності (швидше завантаження)
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    ds = ds.with_options(options)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Побудова моделі
def build_model():
    inputs = keras.Input(shape=(N_MELS, W, 1))
    x = inputs
    for c in (32, 64, 128):
        x = keras.layers.Conv2D(c, 3, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(c, 3, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(50, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# Головна функція
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=".")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    # Налаштування багатопоточності/паралелізму
    parser.add_argument("--intra_threads", type=int, default=0, help="Потоки всередині опів (0=за замовчуванням TF)")
    parser.add_argument("--inter_threads", type=int, default=0, help="Потоки між опами (0=за замовчуванням TF)")
    parser.add_argument("--parallel_calls", type=int, default=0, help="Паралельні виклики map для tf.data (0=AUTOTUNE)")
    parser.add_argument("--deterministic", action="store_true", help="Детермінізм у tf.data (за замовчуванням вимкнено для швидкості)")
    args = parser.parse_args()

    # Налаштовуємо тредінг TF (CPU/GPU) для кращої паралельності
    if args.intra_threads and args.intra_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.intra_threads)
    if args.inter_threads and args.inter_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(args.inter_threads)

    meta_path = os.path.join(args.data_root, "meta", "esc50.csv")
    audio_dir = os.path.join(args.data_root, "audio")
    df = pd.read_csv(meta_path)
    assert 1 <= args.fold <= 5, "fold must be in 1..5"

# Розбиття на train/val
    train_df = df[df["fold"] != args.fold].reset_index(drop=True)
    val_df = df[df["fold"] == args.fold].reset_index(drop=True)

# Формування списків шляхів та міток
    x_train = [os.path.join(audio_dir, f) for f in train_df["filename"].tolist()]
    y_train = train_df["target"].astype(int).to_numpy()
    
    x_val = [os.path.join(audio_dir, f) for f in val_df["filename"].tolist()]
    y_val = val_df["target"].astype(int).to_numpy()
    # Створення датасетів з налаштуванням паралелізму
    num_calls = tf.data.AUTOTUNE if args.parallel_calls <= 0 else args.parallel_calls
    train_ds = build_dataset(
        x_train, y_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_parallel_calls=num_calls,
        deterministic=args.deterministic,
    )
    val_ds = build_dataset(
        x_val, y_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_parallel_calls=num_calls,
        deterministic=args.deterministic,
    )

    model = build_model()
# Компіляція моделі
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
# Контрольні точки 
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(
        os.path.join("checkpoints", f"esc50_tf_fold{args.fold}_best.keras"),
        monitor="val_acc", mode="max", save_best_only=True)

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[ckpt])
    print(model.evaluate(val_ds, return_dict=True))


if __name__ == "__main__":
    main()
