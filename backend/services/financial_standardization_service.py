import os
import time
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
from openai import OpenAI
from pymilvus import MilvusClient, model
import torch

logger = logging.getLogger(__name__)

class FinancialStandardizationService:
    """
    金融术语标准化服务类：负责金融实体的识别与标准化
    使用 LLM 进行命名实体识别 (NER)
    结合 Milvus 向量检索 + LLM 进行术语标准化 (RAG)
    """
    def __init__(self, milvus_uri: str = "http://localhost:19530", collection_name: str = "financial_terms"):
        self.default_entity_types = [
            "金融机构",
            "金融产品",
            "货币单位",
            "交易类型",
            "财务指标",
            "法律法规",
            "其他"
        ]
        
        # 初始化 Milvus 客户端
        try:
            self.client = MilvusClient(uri=milvus_uri)
            self.collection_name = collection_name
            if not self.client.has_collection(self.collection_name):
                logger.warning(f"Collection {self.collection_name} not found in Milvus. Standardization may rely solely on LLM.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.client = None

        # 初始化 Embedding 函数 (保持与 import_financial_data.py 一致)
        try:
            self.embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
                model_name='BAAI/bge-m3',
                device='cuda:0' if torch.cuda.is_available() else 'cpu',
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_function = None

    def get_entity_types(self) -> List[str]:
        """获取支持的实体类型"""
        return self.default_entity_types

    def search_similar_terms(self, query: str, limit: int = 5) -> List[Dict]:
        """
        在 Milvus 中搜索相似的标准术语
        """
        if not self.client or not self.embedding_function:
            logger.warning("Milvus client or embedding function not initialized. Skipping vector search.")
            return []

        try:
            start_time = time.time()
            query_embeddings = self.embedding_function([query])
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                data=[query_embeddings[0].tolist()],
                limit=limit,
                output_fields=["term", "category"]
            )
            end_time = time.time()
            logger.info(f"Vector search for '{query}' took {end_time - start_time:.4f} seconds")
            
            results = []
            for hit in search_result[0]:
                results.append({
                    "term": hit['entity'].get('term'),
                    "category": hit['entity'].get('category'),
                    "distance": float(hit['distance'])
                })
            return results
        except Exception as e:
            logger.error(f"Error in search_similar_terms: {e}")
            return []

    def search_and_explain_stream(self, text: str, api_key: Optional[str] = None):
        """
        流式搜索并解释金融术语
        """
        try:
            if not api_key:
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DeepSeek API key not provided")

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

            # 1. 向量检索 (非流式，因为速度很快)
            candidates = self.search_similar_terms(text, limit=10)
            candidates_str = ""
            if candidates:
                candidates_str = "参考标准术语库中的相关术语：\n"
                for c in candidates:
                    candidates_str += f"- {c['term']} (类别: {c['category']}, 相似度: {c['distance']:.4f})\n"
            else:
                candidates_str = "未在标准术语库中找到相关术语。"

            # 先发送候选词数据（以特殊格式，方便前端解析）
            yield f"DATA: {json.dumps({'candidates': candidates})}\n\n"

            # 2. LLM 流式解释
            explain_prompt = f"""你是一个金融领域的专家。用户输入了一个查询："{text}"。
            
            {candidates_str}
            
            请根据用户的查询和上述参考信息（如果有），用中文详细解释该查询的含义。
            如果查询是一个英文术语，请给出中文翻译和定义。
            如果查询是一个中文术语，请给出详细的定义和背景。
            如果参考信息中有相关的标准术语，请在解释中引用并对比。
            
            请直接返回解释内容，不需要包含 JSON 格式，直接输出 Markdown 格式的文本。
            """

            messages = [
                {"role": "system", "content": "你是一个专业的金融知识助手。"},
                {"role": "user", "content": explain_prompt}
            ]

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error in search_and_explain_stream: {str(e)}")
            yield f"ERROR: {str(e)}"

    def search_and_explain(self, text: str, provider: str = "deepseek", model_name: str = "deepseek-v3", api_key: Optional[str] = None) -> Dict:
        """
        直接在 Milvus 中搜索相似术语，并调用 LLM 解释中文含义
        """
        try:
            if not api_key:
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DeepSeek API key not provided")

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

            # 1. 向量检索
            candidates = self.search_similar_terms(text, limit=10)
            candidates_str = ""
            if candidates:
                candidates_str = "参考标准术语库中的相关术语：\n"
                for c in candidates:
                    candidates_str += f"- {c['term']} (类别: {c['category']}, 相似度: {c['distance']:.4f})\n"
            else:
                candidates_str = "未在标准术语库中找到相关术语。"

            # 2. LLM 解释
            explain_prompt = f"""你是一个金融领域的专家。用户输入了一个查询："{text}"。
            
            {candidates_str}
            
            请根据用户的查询和上述参考信息（如果有），用中文详细解释该查询的含义。
            如果查询是一个英文术语，请给出中文翻译和定义。
            如果查询是一个中文术语，请给出详细的定义和背景。
            如果参考信息中有相关的标准术语，请在解释中引用并对比。
            
            请直接返回解释内容，不需要包含 JSON 格式，直接输出 Markdown 格式的文本。
            """

            messages = [
                {"role": "system", "content": "你是一个专业的金融知识助手。"},
                {"role": "user", "content": explain_prompt}
            ]

            llm_start_time = time.time()
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3,
                max_tokens=1024
            )
            llm_end_time = time.time()
            logger.info(f"DeepSeek API call for '{text}' took {llm_end_time - llm_start_time:.4f} seconds")

            explanation = response.choices[0].message.content.strip()
            logger.info(f"LLM Response: {explanation}")
            
            return {
                "query": text,
                "candidates": candidates,
                "explanation": explanation
            }

        except Exception as e:
            logger.error(f"Error in search_and_explain: {str(e)}")
            raise
