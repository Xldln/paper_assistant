from sentence_transformers import SentenceTransformer
import os




def download_model(model_name):
    print(f"Downloading {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"Downloaded {model_name} successfully!")
    return model


def verify_dl_model(model_cache_dir):
    print("\nVerifying downloaded models...")
    models = [
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-4B",
        "Qwen/Qwen3-Embedding-8B"
    ]
    for model_name in models:
        model_path = os.path.join(model_cache_dir, f"models--{model_name.replace('/', '--')}")
        if os.path.exists(model_path):
            print(f"✓ {model_name} - Downloaded at {model_path}")
        else:
            print(f"✗ {model_name} - Not found")


def main():

    # 设置模型下载目录
    MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
    os.environ['HF_HOME'] = MODEL_CACHE_DIR
    os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR
    os.environ['HF_HUB_CACHE'] = MODEL_CACHE_DIR
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    download_model("Qwen/Qwen3-Embedding-0.6B")
    download_model("Qwen/Qwen3-Embedding-4B")
    download_model("Qwen/Qwen3-Embedding-8B")

    verify_dl_model(MODEL_CACHE_DIR)


if __name__ == "__main__":
    main()