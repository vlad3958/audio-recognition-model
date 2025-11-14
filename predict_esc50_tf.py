import argparse, os, numpy as np, pandas as pd, librosa, tensorflow as tf
from tensorflow import keras

# Ті самі параметри, що й при тренуванні
SR, DUR, N_MELS, N_FFT, HOP = 44100, 5, 128, 2205, 441
W = 1 + (SR * DUR - N_FFT) // HOP  # 496


def load_and_preprocess(audio_path: str) -> np.ndarray:
    """Завантаження користувацького аудіо і перетворення у мел-спектрограму
    повертає тензор форми (1, N_MELS, W, 1) готовий для моделі.
    """
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    # Обрізаємо/падимо до фіксованої довжини
    y = np.pad(y, (0, SR * DUR - len(y)), mode="constant")[: SR * DUR]
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    mel = np.expand_dims(mel, -1)  # (n_mels, time, 1)
    # Узгоджуємо ширину по часу
    if mel.shape[1] < W:
        mel = np.pad(mel, ((0, 0), (0, W - mel.shape[1]), (0, 0)))
    else:
        mel = mel[:, :W, :]
    mel = np.expand_dims(mel, 0)  # batch dimension
    return mel


def load_label_mapping(meta_csv: str) -> dict:
    """Створює словник {target_id: category_name}."""
    df = pd.read_csv(meta_csv)
    return {int(r.target): r.category for r in df.itertuples()}


def predict(model_path: str, audio_path: str, meta_csv: str, top_k: int = 5):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Не знайдено модель: {model_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Не знайдено аудіо файл: {audio_path}")
    if not os.path.isfile(meta_csv):
        raise FileNotFoundError(f"Не знайдено meta csv: {meta_csv}")

    print("[INFO] Завантаження моделі...")
    model = keras.models.load_model(model_path)

    print("[INFO] Препроцесинг аудіо...")
    x = load_and_preprocess(audio_path)

    print("[INFO] Інференс...")
    preds = model.predict(x, verbose=0)[0]  # (50,)
    label_map = load_label_mapping(meta_csv)

    top_k = min(top_k, len(preds))
    indices = np.argsort(preds)[-top_k:][::-1]

    print("\nTop", top_k, "класи:")
    for rank, idx in enumerate(indices, 1):
        cls_name = label_map.get(int(idx), f"class_{idx}")
        print(f"{rank:2d}. {cls_name:<20} prob={preds[idx]:.4f}")

    best_idx = int(indices[0])
    print("\nНайімовірніший клас:", label_map.get(best_idx, f"class_{best_idx}"))


def main():
    parser = argparse.ArgumentParser(description="Предикт ESC-50 моделі на користувацькому аудіо")
    parser.add_argument("--model", required=True, help="Шлях до збереженої моделі (.keras)")
    parser.add_argument("--audio_path", required=True, help="Шлях до .wav / .ogg / .mp3 файлу")
    parser.add_argument("--meta_csv", default="ESC-50-master/meta/esc50.csv", help="Шлях до esc50.csv для мапи міток")
    parser.add_argument("--top_k", type=int, default=5, help="Скільки топ результатів показати")
    args = parser.parse_args()
    predict(args.model, args.audio_path, args.meta_csv, top_k=args.top_k)


if __name__ == "__main__":
    main()
