from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
import re
from langchain_core.documents import Document
import json
from typing import List
import os
def process_pdf_with_langchain(pdf_path: str):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"PDF总页数: {len(pages)}")
    print(f"第一页预览: {pages[0].page_content[:200]}...")

    return pages

def split_documents(documents, chunk_size=1000, chunk_overlap=200):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        # 每个块的大小（字符数）
        chunk_overlap=chunk_overlap,   # 块之间的重叠部分
        length_function=len,
        separators=[
            "\n\n",     # 段落分隔
            "\n",       # 行分隔
            ".",        # 句子分隔
            " ",        # 单词分隔
            ""          # 字符分隔（最后选择）
        ]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"原始文档数: {len(documents)}")
    print(f"分割后块数: {len(chunks)}")
    print(f"平均块大小: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks):.0f}")

    return chunks

def enhance_document_metadata(chunks):
    enhanced_chunks = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content

        # 创建新的文档对象
        enhanced_doc = Document(
            page_content=text,
            metadata={
                **chunk.metadata,  # 保留原始元数据
                "chunk_index": i,
                "char_count": len(text),
                "word_count": len(text.split()),
                "sentence_count": len(re.split(r'[.!?]+', text)),
                "has_citation": bool(re.search(r'\[\d+\]', text)),  # 是否包含引用
                "has_formula": bool(re.search(r'[∑∫√∂≈≠≤≥]', text)),  # 是否包含公式
            }
        )

        enhanced_chunks.append(enhanced_doc)

    return enhanced_chunks

def clean_text(chunks):

    cleaned_chunks = []

    for chunk in chunks:
        text = chunk.page_content

        text = re.sub(r'\s+', ' ', text)  # 多个空格替换为一个
        text = re.sub(r'\n\s*\n', '\n\n', text)  # 清理空行
        text = text.strip()
        cleaned_doc = Document(
            page_content=text,
            metadata=chunk.metadata
        )
        cleaned_chunks.append(cleaned_doc)

    return cleaned_chunks



def save_chunks_to_json(chunks: List[Document], output_file: str):

    data = []
    for i, chunk in enumerate(chunks):
        data.append({
            "chunk_id": i,
            "content": chunk.page_content,
            "metadata": chunk.metadata
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"保存了 {len(chunks)} 个文本块到 {output_file}")

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

    file_path = "test.pdf"

    pdf_pages = process_pdf_with_langchain(file_path)
    # document_chunks = split_documents(pdf_pages)
    clean_chunks = clean_text(pdf_pages)
    enhanced_chunks = enhance_document_metadata(clean_chunks)

    print(f"处理完成，共 {len(enhanced_chunks)} 个增强文本块")

    file_name = os.path.basename(file_path)
    output_file = f"{os.path.splitext(file_name)[0]}_processed_chunks.json"

    save_chunks_to_json(enhanced_chunks, output_file)

    print(f'Enhanced chunks saved to {output_file}')





if __name__ == "__main__":
    main()