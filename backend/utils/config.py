from enum import Enum
from typing import Dict, Any

class VectorDBProvider(str, Enum):
    MILVUS = "milvus"
    CHROMA = "chroma"

# Milvus 配置（保持您原有的）
MILVUS_CONFIG = {
    "uri": "tcp://10.250.221.18:19530",
    "index_types": {
        "flat": "FLAT",
        "ivf_flat": "IVF_FLAT",
        "ivf_sq8": "IVF_SQ8",
        "hnsw": "HNSW"
    },
    "index_params": {
        "flat": {},
        "ivf_flat": {"nlist": 1024},
        "ivf_sq8": {"nlist": 1024},
        "hnsw": {
            "M": 16,
            "efConstruction": 500
        }
    }
}

# # 新增 Chroma 配置
# CHROMA_CONFIG = {
#     # 运行模式选择: 'persistent' (本地持久化) 或 'http' (远程服务)
#     "mode": "persistent", 
    
#     # 连接配置
#     "settings": {
#         "persist_directory": "./chroma_storage", # 对应 PersistentClient
#         "host": "localhost",                     # 对应 HttpClient
#         "port": 8000,                            # 对应 HttpClient
#         "anonymized_telemetry": False            # 关闭匿名数据收集
#     },
#     "index_types": {
#         "hnsw": "HNSW",      # Chroma 默认且核心的索引方式
#         "cosine": "cosine",  # 余弦相似度
#         "l2": "l2",          # 欧氏距离
#         "ip": "ip"           # 内积
#     },
    
#     # Chroma 默认使用 HNSW 索引，其参数通过 collection_metadata 传递
#     # 常见的距离度量（Space）: 'l2', 'ip', 'cosine'
#     "index_params": {
#         "hnsw": {
#             "hnsw:space": "cosine",
#             "hnsw:construction_ef": 128,
#             "hnsw:M": 16,
#             "hnsw:search_ef": 100
#         }
#     }
# }
