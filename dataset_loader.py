import importlib


def get_common_voice_uk_stream(split="train", sampling_rate=16000, source="auto", auth_token=None):
    """Повертає streaming Dataset для Ukrainian Common Voice.

    source варіанти:
      - 'auto' : спробувати v17_0 → v16_0 → v15_0
      - 'cv17' : тільки common_voice_17_0
      - 'cv16' : тільки common_voice_16_0
      - 'cv15' : тільки common_voice_15_0

    Якщо недоступно: дає розширені поради, включно з оновленням пакетів та можливим використанням auth token.
    """
    datasets = importlib.import_module("datasets")
    if source == "auto":
        attempts = [
            ("mozilla-foundation/common_voice_17_0", "uk"),
            ("mozilla-foundation/common_voice_16_0", "uk"),
            ("mozilla-foundation/common_voice_15_0", "uk"),
        ]
    elif source == "cv17":
        attempts = [("mozilla-foundation/common_voice_17_0", "uk")]
    elif source == "cv16":
        attempts = [("mozilla-foundation/common_voice_16_0", "uk")]
    elif source == "cv15":
        attempts = [("mozilla-foundation/common_voice_15_0", "uk")]
    else:
        raise ValueError(f"Невідомий source: {source}")

    last_err = None
    for name, conf in attempts:
        try:
            ds = datasets.load_dataset(name, conf, streaming=True, split=split, use_auth_token=auth_token)
            ds = ds.cast_column("audio", datasets.Audio(sampling_rate=sampling_rate))
            return ds
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "Не вдалося завантажити жодну версію Common Voice Ukrainian.\n" +
        f"Остання помилка: {last_err}\n" +
        "Перевірки/поради:\n" +
        "  1. Оновіть пакети: pip install -U datasets huggingface_hub pyarrow\n" +
        "  2. Якщо за проксі / VPN – перевірте доступ до https://huggingface.co\n" +
        "  3. Спробуйте інший split: --split validation або --split test\n" +
        "  4. Зменшіть ліміт мовлення (--speech_limit) у тренувальному скрипті\n" +
        "  5. Використайте токен HF: export HF_TOKEN=... і передайте auth_token\n" +
        "  6. Якщо нічого не працює – перейдіть тимчасово на ESC-50 тільки або локальний набір."
    )


if __name__ == "__main__":
    import argparse, sys, itertools

    ap = argparse.ArgumentParser(description="Потоковий перегляд Common Voice Ukrainian (auto v17→v16→v15 fallback)")
    ap.add_argument("--split", default="train", help="Спліт: train/validation/test")
    ap.add_argument("--sr", type=int, default=16000, help="Target sampling rate")
    ap.add_argument("--show", type=int, default=5, help="Скільки прикладів показати")
    ap.add_argument("--source", default="auto", choices=["auto","cv17","cv16","cv15"], help="Версія датасету")
    ap.add_argument("--auth_token", default=None, help="HF auth token (при потребі приватного доступу)")
    args = ap.parse_args()

    try:
        ds = get_common_voice_uk_stream(args.split, sampling_rate=args.sr, source=args.source, auth_token=args.auth_token)
    except Exception as e:
        print("[ERROR] Не вдалося завантажити Common Voice: ", e, file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Streaming dataset готовий: split={args.split}, sr={args.sr}, source={args.source}")
    for i, ex in enumerate(itertools.islice(ds, args.show)):
        arr = ex["audio"]["array"]
        dur = len(arr) / args.sr
        print(f"  #{i+1} client_id={ex.get('client_id')} gender={ex.get('gender')} dur={dur:.2f}s text={ex.get('sentence')[:40] if ex.get('sentence') else ''}")
    print("[INFO] Готово. Використовуйте get_common_voice_uk_stream() у тренуванні.")
 