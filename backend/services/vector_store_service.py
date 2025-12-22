import os
from datetime import datetime
import json
from typing import List, Dict, Any
import logging
from pathlib import Path
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from utils.config import VectorDBProvider, MILVUS_CONFIG  # Updated import
from pypinyin import lazy_pinyin, Style
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings  # 添加嵌入函数导入

logger = logging.getLogger(__name__)

class VectorDBConfig:
    """
    向量数据库配置类，用于存储和管理向量数据库的配置信息
    """
    def __init__(self, provider: str, index_mode: str):
        """
        初始化向量数据库配置
        
        参数:
            provider: 向量数据库提供商名称
            index_mode: 索引模式
        """
        self.provider = provider
        self.index_mode = index_mode
        self.milvus_uri = MILVUS_CONFIG["uri"]
        # Chroma配置
        self.chroma_persist_dir = "03-vector-store/chroma"

    def _get_milvus_index_type(self, index_mode: str) -> str:
        """
        根据索引模式获取Milvus索引类型
        
        参数:
            index_mode: 索引模式
            
        返回:
            对应的Milvus索引类型
        """
        return MILVUS_CONFIG["index_types"].get(index_mode, "FLAT")
    
    def _get_milvus_index_params(self, index_mode: str) -> Dict[str, Any]:
        """
        根据索引模式获取Milvus索引参数
        
        参数:
            index_mode: 索引模式
            
        返回:
            对应的Milvus索引参数字典
        """
        return MILVUS_CONFIG["index_params"].get(index_mode, {})

class VectorStoreService:
    """
    向量存储服务类，提供向量数据的索引、查询和管理功能
    """
    def __init__(self):
        """
        初始化向量存储服务
        """
        self.initialized_dbs = {}
        # 确保存储目录存在
        os.makedirs("03-vector-store", exist_ok=True)
    
    def _get_milvus_index_type(self, config: VectorDBConfig) -> str:
        """
        从配置对象获取Milvus索引类型
        
        参数:
            config: 向量数据库配置对象
            
        返回:
            Milvus索引类型
        """
        return config._get_milvus_index_type(config.index_mode)
    
    def _get_milvus_index_params(self, config: VectorDBConfig) -> Dict[str, Any]:
        """
        从配置对象获取Milvus索引参数
        
        参数:
            config: 向量数据库配置对象
            
        返回:
            Milvus索引参数字典
        """
        return config._get_milvus_index_params(config.index_mode)
    
    def index_embeddings(self, embedding_file: str, config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到向量数据库
        
        参数:
            embedding_file: 嵌入向量文件路径
            config: 向量数据库配置对象
            
        返回:
            索引结果信息字典
        """
        start_time = datetime.now()
        
        # 读取embedding文件
        embeddings_data = self._load_embeddings(embedding_file)
        
        # 根据不同的数据库进行索引
        if config.provider == VectorDBProvider.MILVUS:
            result = self._index_to_milvus(embeddings_data, config)
        elif config.provider == VectorDBProvider.CHROMA:
            result = self._index_to_chroma(embeddings_data, config)
        else:
            raise ValueError(f"Unsupported vector database provider: {config.provider}")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            "database": config.provider,
            "index_mode": config.index_mode,
            "total_vectors": len(embeddings_data["embeddings"]),
            "index_size": result.get("index_size", "N/A"),
            "processing_time": processing_time,
            "collection_name": result.get("collection_name", "N/A")
        }
    
    def _load_embeddings(self, file_path: str) -> Dict[str, Any]:
        """
        加载embedding文件，返回配置信息和embeddings
        
        参数:
            file_path: 嵌入向量文件路径
            
        返回:
            包含嵌入向量和元数据的字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loading embeddings from {file_path}")
                
                if not isinstance(data, dict) or "embeddings" not in data:
                    raise ValueError("Invalid embedding file format: missing 'embeddings' key")
                    
                # 返回完整的数据，包括顶层配置
                logger.info(f"Found {len(data['embeddings'])} embeddings")
                return data
                
        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {str(e)}")
            raise
    
    def _index_to_milvus(self, embeddings_data: Dict[str, Any], config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到Milvus数据库
        
        参数:
            embeddings_data: 嵌入向量数据
            config: 向量数据库配置对象
            
        返回:
            索引结果信息字典
        """
        try:
            # 使用 filename 作为 collection 名称前缀
            filename = embeddings_data.get("filename", "")
            # 如果有 .pdf 后缀，移除它
            base_name = filename.replace('.pdf', '') if filename else "doc"
            
            # Convert Chinese characters to pinyin
            base_name = ''.join(lazy_pinyin(base_name, style=Style.NORMAL))
            
            # Replace hyphens with underscores in the base name
            base_name = base_name.replace('-', '_')
            
            # Ensure the collection name starts with a letter or number (Chroma requirement)
            # and only contains alphanumeric, underscores, or hyphens
            if not base_name[0].isalnum():
                # If first character is not alphanumeric, add a letter prefix
                base_name = f"doc_{base_name}"
            
            # Remove any invalid characters for Chroma
            base_name = ''.join(c for c in base_name if c.isalnum() or c in ['_', '-'])
            
            # Truncate if necessary (Chroma requires 3-63 characters)
            if len(base_name) < 3:
                base_name = f"{base_name}___"[:63]
            elif len(base_name) > 63:
                base_name = base_name[:63]
            
            # Ensure it doesn't end with non-alphanumeric
            while base_name and not base_name[-1].isalnum():
                base_name = base_name[:-1]
            
            # Ensure it doesn't start with non-alphanumeric (should already be handled)
            while base_name and not base_name[0].isalnum():
                base_name = base_name[1:]
            
            # Get embedding provider
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = f"{base_name}_{embedding_provider}_{timestamp}"
            
            # 连接到Milvus
            connections.connect(
                alias="default", 
                uri=config.milvus_uri
            )
            
            # 从顶层配置获取向量维度
            vector_dim = int(embeddings_data.get("vector_dimension"))
            if not vector_dim:
                raise ValueError("Missing vector_dimension in embedding file")
            
            logger.info(f"Creating collection with dimension: {vector_dim}")
            
            # 定义字段
            fields = [
                {"name": "id", "dtype": "INT64", "is_primary": True, "auto_id": True},
                {"name": "content", "dtype": "VARCHAR", "max_length": 5000},
                {"name": "document_name", "dtype": "VARCHAR", "max_length": 255},
                {"name": "chunk_id", "dtype": "INT64"},
                {"name": "total_chunks", "dtype": "INT64"},
                {"name": "word_count", "dtype": "INT64"},
                {"name": "page_number", "dtype": "VARCHAR", "max_length": 10},
                {"name": "page_range", "dtype": "VARCHAR", "max_length": 10},
                # {"name": "chunking_method", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_provider", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_model", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_timestamp", "dtype": "VARCHAR", "max_length": 50},
                {
                    "name": "vector",
                    "dtype": "FLOAT_VECTOR",
                    "dim": vector_dim,
                    "params": self._get_milvus_index_params(config)
                }
            ]
            
            # 准备数据为列表格式
            entities = []
            for emb in embeddings_data["embeddings"]:
                entity = {
                    "content": str(emb["metadata"].get("content", "")),
                    "document_name": embeddings_data.get("filename", ""),  # 使用 filename 而不是 document_name
                    "chunk_id": int(emb["metadata"].get("chunk_id", 0)),
                    "total_chunks": int(emb["metadata"].get("total_chunks", 0)),
                    "word_count": int(emb["metadata"].get("word_count", 0)),
                    "page_number": str(emb["metadata"].get("page_number", 0)),
                    "page_range": str(emb["metadata"].get("page_range", "")),
                    # "chunking_method": str(emb["metadata"].get("chunking_method", "")),
                    "embedding_provider": embeddings_data.get("embedding_provider", ""),  # 从顶层配置获取
                    "embedding_model": embeddings_data.get("embedding_model", ""),  # 从顶层配置获取
                    "embedding_timestamp": str(emb["metadata"].get("embedding_timestamp", "")),
                    "vector": [float(x) for x in emb.get("embedding", [])]
                }
                entities.append(entity)
            
            logger.info(f"Creating Milvus collection: {collection_name}")
            
            # 创建collection
            # field_schemas = [
            #     FieldSchema(name=field["name"], 
            #                dtype=getattr(DataType, field["dtype"]),
            #                is_primary="is_primary" in field and field["is_primary"],
            #                auto_id="auto_id" in field and field["auto_id"],
            #                max_length=field.get("max_length"),
            #                dim=field.get("dim"),
            #                params=field.get("params"))
            #     for field in fields
            # ]

            field_schemas = []
            for field in fields:
                extra_params = {}
                if field.get('max_length') is not None:
                    extra_params['max_length'] = field['max_length']
                if field.get('dim') is not None:
                    extra_params['dim'] = field['dim']
                if field.get('params') is not None:
                    extra_params['params'] = field['params']
                field_schema = FieldSchema(
                    name=field["name"], 
                    dtype=getattr(DataType, field["dtype"]),
                    is_primary=field.get("is_primary", False),
                    auto_id=field.get("auto_id", False),
                    **extra_params
                )
                field_schemas.append(field_schema)

            schema = CollectionSchema(fields=field_schemas, description=f"Collection for {collection_name}")
            collection = Collection(name=collection_name, schema=schema)
            
            # 插入数据
            logger.info(f"Inserting {len(entities)} vectors")
            insert_result = collection.insert(entities)
            collection.flush()
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": self._get_milvus_index_type(config),
                "params": self._get_milvus_index_params(config)
            }
            collection.create_index(field_name="vector", index_params=index_params)
            collection.load()
            
            return {
                "index_size": len(insert_result.primary_keys),
                "collection_name": collection_name
            }
            
        except Exception as e:
            logger.error(f"Error indexing to Milvus: {str(e)}")
            raise
        
        finally:
            connections.disconnect("default")
            
    def _index_to_chroma(self, embeddings_data: Dict[str, Any], config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到Chroma数据库
        
        参数:
            embeddings_data: 嵌入向量数据
            config: 向量数据库配置对象
            
        返回:
            索引结果信息字典
        """
        try:
            
            # 使用 filename 作为 collection 名称前缀
            filename = embeddings_data.get("filename", "")
            # 如果有 .pdf 后缀，移除它
            base_name = filename.replace('.pdf', '') if filename else "doc"
            
            # Convert Chinese characters to pinyin
            base_name = ''.join(lazy_pinyin(base_name, style=Style.NORMAL))
            
            # Replace hyphens with underscores in the base name
            base_name = base_name.replace('-', '_')
            
            # Ensure the collection name starts with a letter or number (Chroma requirement)
            # and only contains alphanumeric, underscores, or hyphens
            if not base_name[0].isalnum():
                # If first character is not alphanumeric, add a letter prefix
                base_name = f"doc_{base_name}"
            
            # Remove any invalid characters for Chroma
            base_name = ''.join(c for c in base_name if c.isalnum() or c in ['_', '-'])
            
            # Truncate if necessary (Chroma requires 3-63 characters)
            if len(base_name) < 3:
                base_name = f"{base_name}___"[:63]
            elif len(base_name) > 63:
                base_name = base_name[:63]
            
            # Ensure it doesn't end with non-alphanumeric
            while base_name and not base_name[-1].isalnum():
                base_name = base_name[:-1]
            
            # Ensure it doesn't start with non-alphanumeric (should already be handled)
            while base_name and not base_name[0].isalnum():
                base_name = base_name[1:]
            
            # Get embedding provider
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = f"{base_name}_{embedding_provider}_{timestamp}"
            
            # 准备数据为LangChain Document格式
            documents = []
            embeddings = []
            metadatas = []
            
            for emb in embeddings_data["embeddings"]:
                content = emb["metadata"].get("content", "")
                metadata = {
                    "document_name": embeddings_data.get("filename", ""),
                    "chunk_id": emb["metadata"].get("chunk_id", 0),
                    "total_chunks": emb["metadata"].get("total_chunks", 0),
                    "word_count": emb["metadata"].get("word_count", 0),
                    "page_number": emb["metadata"].get("page_number", 0),
                    "page_range": emb["metadata"].get("page_range", ""),
                    "embedding_provider": embeddings_data.get("embedding_provider", ""),
                    "embedding_model": embeddings_data.get("embedding_model", ""),
                    "embedding_timestamp": emb["metadata"].get("embedding_timestamp", "")
                }
                
                documents.append(content)
                embeddings.append(emb.get("embedding", []))
                metadatas.append(metadata)
            
            logger.info(f"Creating Chroma collection: {collection_name}")
            
            # 确保Chroma持久化目录存在
            os.makedirs(config.chroma_persist_dir, exist_ok=True)
            
            # 获取向量维度
            vector_dim = len(embeddings[0]) if embeddings else 0
            if not vector_dim:
                raise ValueError("No embeddings found or embeddings have zero dimension")
            
            # 创建自定义嵌入函数实例
            embedding_function = DummyEmbeddingFunction(vector_dim)
            
            # 使用预先计算好的embeddings创建Chroma实例
            chroma_db = Chroma(
                collection_name=collection_name,
                persist_directory=config.chroma_persist_dir,
                embedding_function=embedding_function  # 添加嵌入函数
            )
            
            # 向Chroma添加文档和向量
            logger.info(f"Inserting {len(embeddings)} vectors into Chroma")
            chroma_db.add_texts(
                texts=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            # 持久化到磁盘
            chroma_db.persist()
            
            return {
                "index_size": len(embeddings),
                "collection_name": collection_name
            }
            
        except Exception as e:
            logger.error(f"Error indexing to Chroma: {str(e)}")
            raise

    def list_collections(self, provider: str) -> List[str]:
        """
        列出指定提供商的所有集合
        
        参数:
            provider: 向量数据库提供商
            
        返回:
            集合名称列表
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                collections = utility.list_collections()
                return collections
            finally:
                connections.disconnect("default")
        elif provider == VectorDBProvider.CHROMA:
            try:
                # 确保Chroma持久化目录存在
                chroma_persist_dir = "03-vector-store/chroma"
                if os.path.exists(chroma_persist_dir):
                    # 使用Chroma客户端列出集合
                    from chromadb import PersistentClient
                    client = PersistentClient(path=chroma_persist_dir)
                    collections = [col.name for col in client.list_collections()]
                    return collections
                return []
            except Exception as e:
                logger.error(f"Error listing Chroma collections: {str(e)}")
                return []
        return []

    def delete_collection(self, provider: str, collection_name: str) -> bool:
        """
        删除指定的集合
        
        参数:
            provider: 向量数据库提供商
            collection_name: 集合名称
            
        返回:
            是否删除成功
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                utility.drop_collection(collection_name)
                return True
            finally:
                connections.disconnect("default")
        elif provider == VectorDBProvider.CHROMA:
            try:
                # 确保Chroma持久化目录存在
                chroma_persist_dir = "03-vector-store/chroma"
                if os.path.exists(chroma_persist_dir):
                    # 使用Chroma客户端删除集合
                    from chromadb import PersistentClient
                    client = PersistentClient(path=chroma_persist_dir)
                    client.delete_collection(collection_name)
                    return True
                return False
            except Exception as e:
                logger.error(f"Error deleting Chroma collection: {str(e)}")
                return False
        return False

    def get_collection_info(self, provider: str, collection_name: str) -> Dict[str, Any]:
        """
        获取指定集合的信息
        
        参数:
            provider: 向量数据库提供商
            collection_name: 集合名称
            
        返回:
            集合信息字典
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                collection = Collection(collection_name)
                return {
                    "name": collection_name,
                    "num_entities": collection.num_entities,
                    "schema": collection.schema.to_dict()
                }
            finally:
                connections.disconnect("default")
        elif provider == VectorDBProvider.CHROMA:
            try:
                # 确保Chroma持久化目录存在
                chroma_persist_dir = "03-vector-store/chroma"
                if os.path.exists(chroma_persist_dir):
                    # 使用Chroma客户端获取集合信息
                    from chromadb import PersistentClient
                    client = PersistentClient(path=chroma_persist_dir)
                    
                    # 获取集合
                    collection = client.get_collection(collection_name)
                    if collection:
                        # 获取集合大小
                        num_entities = collection.count()
                        
                        # 获取集合的schema信息（Chroma的schema结构与Milvus不同）
                        # Chroma使用metadata和embedding向量，没有传统的schema
                        schema_info = {
                            "fields": [
                                {
                                    "name": "id",
                                    "type": "string",
                                    "is_primary": True
                                },
                                {
                                    "name": "embedding",
                                    "type": "embedding"
                                },
                                {
                                    "name": "metadata",
                                    "type": "dict"
                                },
                                {
                                    "name": "document",
                                    "type": "string"
                                }
                            ]
                        }
                        
                        return {
                            "name": collection_name,
                            "num_entities": num_entities,
                            "schema": schema_info
                        }
                return {}
            except Exception as e:
                logger.error(f"Error getting Chroma collection info: {str(e)}")
                return {}
        return {}

# 自定义嵌入函数类，用于预计算的嵌入向量
class DummyEmbeddingFunction(Embeddings):
    def __init__(self, embedding_dimension: int):
        self.embedding_dimension = embedding_dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 因为我们使用预计算的嵌入，所以这个方法不会被实际调用
        # 但为了满足接口要求，我们返回一些虚拟的嵌入
        return [[0.0] * self.embedding_dimension for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        # 同样，这个方法也不会被实际调用
        return [0.0] * self.embedding_dimension