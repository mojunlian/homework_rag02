from pymilvus import model
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
import pandas as pd
from tqdm import tqdm
import logging
from dotenv import load_dotenv
import torch
import os

load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 OpenAI 嵌入函数 (这里使用 BGE-M3)
embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
            model_name='BAAI/bge-m3',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True
        )

# 文件路径
# 自动定位到项目根目录下的 backend/data/万条金融标准术语.csv
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设脚本在 backend/tools 目录下，我们需要向上两级找到项目根目录，然后进入 backend/data
# d:\course\1\homework_rag02\backend\tools\import_financial_data.py -> d:\course\1\homework_rag02
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) 
# 或者更简单点，既然我们知道结构是 backend/tools 和 backend/data
# 从 current_dir (backend/tools) -> 上一级 (backend) -> data
data_dir = os.path.join(os.path.dirname(current_dir), "data")
file_path = os.path.join(data_dir, "万条金融标准术语.csv")

if not os.path.exists(file_path):
    # Fallback: 尝试直接使用相对路径，或者报错
    logging.warning(f"File not found at {file_path}, trying relative path...")
    file_path = "backend/data/万条金融标准术语.csv"

logging.info(f"CSV file path: {file_path}")

# 连接到 Milvus (本地部署，端口 19530)
client = MilvusClient(uri="http://localhost:19530")

collection_name = "financial_terms"

# 加载数据
logging.info("Loading data from CSV")
# CSV 只有两列数据，没有表头：Term, Category (FINTERM)
df = pd.read_csv(file_path, 
                 header=None,
                 names=['term', 'category'],
                 dtype=str, 
                 low_memory=False,
                 ).fillna("NA")

# 获取向量维度（使用一个样本文档）
sample_doc = "Sample Financial Term"
sample_embedding = embedding_function([sample_doc])[0]
vector_dim = len(sample_embedding)

# 构造Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    FieldSchema(name="term", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
]

schema = CollectionSchema(fields, 
                          "Financial Terms Collection", 
                          enable_dynamic_field=True)

# 如果集合不存在，创建集合
if not client.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )
    logging.info(f"Created new collection: {collection_name}")

    # 在创建集合后添加索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",  # 指定要为哪个字段创建索引，这里是向量字段
        index_type="AUTOINDEX",  # 使用自动索引类型
        metric_type="COSINE",  # 使用余弦相似度
        params={} 
    )

    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
else:
    logging.info(f"Collection {collection_name} already exists. Appending data...")

# 批量处理
batch_size = 1000

for start_idx in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx]

    # 准备文档 - 只需要 term 即可
    docs = batch_df['term'].tolist()

    # 生成嵌入
    try:
        embeddings = embedding_function(docs)
    except Exception as e:
        logging.error(f"Error generating embeddings for batch {start_idx // batch_size + 1}: {e}")
        continue

    # 准备数据
    data = []
    for idx, (_, row) in enumerate(batch_df.iterrows()):
        data.append({
            "vector": embeddings[idx],
            "term": str(row['term']),
            "category": str(row['category'])
        })

    # 插入数据
    try:
        res = client.insert(
            collection_name=collection_name,
            data=data
        )
        client.flush(collection_name)
        # logging.info(f"Inserted batch {start_idx // batch_size + 1}, result: {res}")
    except Exception as e:
        logging.error(f"Error inserting batch {start_idx // batch_size + 1}: {e}")

logging.info("Insert process completed.")
