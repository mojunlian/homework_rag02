from pymilvus import MilvusClient, model
from dotenv import load_dotenv
import logging
import torch
import os

load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def query_financial_terms(query_text: str, limit: int = 5):
    """
    查询金融术语
    """
    try:
        # 1. 初始化 Embedding 函数 (BGE-M3)
        logging.info("Initializing embedding model...")
        embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
            model_name='BAAI/bge-m3',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True
        )

        # 2. 连接到 Milvus
        logging.info("Connecting to Milvus...")
        client = MilvusClient(uri="http://localhost:19530")
        collection_name = "financial_terms"

        if not client.has_collection(collection_name):
            logging.error(f"Collection {collection_name} does not exist. Please run import_financial_data.py first.")
            return

        # 3. 生成查询向量
        logging.info(f"Generating embedding for query: '{query_text}'")
        query_embeddings = embedding_function([query_text])

        # 4. 执行搜索
        logging.info("Searching in Milvus...")
        search_result = client.search(
            collection_name=collection_name,
            data=[query_embeddings[0].tolist()],
            limit=limit,
            output_fields=["term", "category"]
        )

        # 5. 打印结果
        logging.info(f"Search results for '{query_text}':")
        for hits in search_result:
            for hit in hits:
                term = hit['entity'].get('term')
                category = hit['entity'].get('category')
                distance = hit['distance']
                print(f"- Term: {term}, Category: {category}, Score: {distance:.4f}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # 示例查询
    import argparse
    parser = argparse.ArgumentParser(description="Query financial terms in Milvus.")
    parser.add_argument("query", type=str, nargs="?", default="ABA", help="The query text to search for.")
    args = parser.parse_args()

    query_financial_terms(args.query)
