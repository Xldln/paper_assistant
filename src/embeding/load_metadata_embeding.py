
import os

import torch
import torch.nn.functional as F
import json
from typing import List
import numpy as np
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'




def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def embed_chunks(chunks: List[Document], model_path: str = "Qwen/Qwen3-Embedding-0.6B", batch_size: int = 8):

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    task = 'Represent the document for retrieval: '
    texts = [chunk.page_content for chunk in chunks]
    input_texts = [get_detailed_instruct(task, text) for text in texts]
    all_embeddings = []
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i+batch_size]
        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        
        batch_dict.to(model.device)
        
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            # 归一化嵌入
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    return all_embeddings


def load_chunks_from_json(input_file: str) -> List[Document]:
    """从JSON文件加载文档块"""

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = []
    for item in data:
        doc = Document(
            page_content=item["content"],
            metadata=item["metadata"]
        )
        chunks.append(doc)

    return chunks



def main():

    chunks_json_path = "test_processed_chunks.json"
    chunks = load_chunks_from_json(chunks_json_path)
    print(f"Loaded {len(chunks)} chunks")
    embeddings = embed_chunks(chunks)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    emb_file_name = os.path.basename(chunks_json_path).replace("_processed_chunks.json", "_embeddings.npy")
    np.save(emb_file_name, embeddings)
    
    print(f"Embeddings saved to {emb_file_name}")




if __name__ == "__main__":
    main()