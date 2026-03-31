from sentence_transformers import SentenceTransformer
import os

# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'


def download_model(model_name):

    SentenceTransformer(model_name)


def verify_dl_model():
    


def main():

    download_model("Qwen/Qwen3-Embedding-0.6B")
    download_model("Qwen/Qwen3-Embedding-0.6B")
    download_model("Qwen/Qwen3-Embedding-0.6B")

    verify_dl_model()



if __name__ == "__main__":
    main()