import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

class FinancialStandardizationService:
    """
    金融术语标准化服务类：负责金融实体的识别与标准化
    使用 LLM 进行命名实体识别 (NER) 和术语标准化
    """
    def __init__(self):
        self.default_entity_types = [
            "金融机构",
            "金融产品",
            "货币单位",
            "交易类型",
            "财务指标",
            "法律法规"
        ]
        
    def get_entity_types(self) -> List[str]:
        """获取支持的实体类型"""
        return self.default_entity_types

    def recognize_entities(self, text: str, entity_types: List[str] = None, provider: str = "deepseek", model_name: str = "deepseek-v3", api_key: Optional[str] = None) -> List[Dict]:
        """
        识别文本中的金融实体
        """
        try:
            if not entity_types:
                entity_types = self.default_entity_types

            if not api_key:
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DeepSeek API key not provided")

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

            types_str = "、".join(entity_types)
            prompt = f"""你是一个金融领域的专家。请从以下文本中识别出属于以下类型的金融实体：{types_str}。
            
            文本：
            {text}
            
            请直接以 JSON 数组格式返回结果，每个对象包含 'entity' (实体文本) 和 'type' (实体类型) 两个字段。不要包含任何其他解释。
            示例格式：
            [
                {{"entity": "工商银行", "type": "金融机构"}},
                {{"entity": "余额宝", "type": "金融产品"}}
            ]
            """

            messages = [
                {"role": "system", "content": "你是一个专业的金融文本分析助手。"},
                {"role": "user", "content": prompt}
            ]

            response = client.chat.completions.create(
                model="deepseek-chat", # 使用 deepseek-v3 的 API 名称
                messages=messages,
                temperature=0.1, # 降低随机性
                max_tokens=1024,
                response_format={"type": "json_object"} if provider == "openai" else None # DeepSeek 暂时不一定完全支持这个参数，先手动解析
            )

            content = response.choices[0].message.content.strip()
            
            # 清理可能的 Markdown 格式
            if content.startswith("```json"):
                content = content.replace("```json", "", 1).rsplit("```", 1)[0].strip()
            elif content.startswith("```"):
                content = content.replace("```", "", 1).rsplit("```", 1)[0].strip()

            entities = json.loads(content)
            if isinstance(entities, dict) and "entities" in entities:
                entities = entities["entities"]
            
            return entities

        except Exception as e:
            logger.error(f"Error in recognize_entities: {str(e)}")
            raise

    def standardize_entity(self, entity: str, entity_type: str, provider: str = "deepseek", model_name: str = "deepseek-v3", api_key: Optional[str] = None) -> Dict:
        """
        标准化单个金融实体
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

            prompt = f"""你是一个金融术语标准化专家。请将以下识别到的实体进行标准化。
            
            实体：{entity}
            类型：{entity_type}
            
            请返回一个 JSON 对象，包含以下字段：
            - 'original': 原始实体文本
            - 'standardized': 标准化后的正式术语（如：缩写转全称，别名转官方名称）
            - 'explanation': 简短的标准化原因说明
            
            示例：
            {{
                "original": "工行",
                "standardized": "中国工商银行",
                "explanation": "将常用简称转换为官方全称"
            }}
            """

            messages = [
                {"role": "system", "content": "你是一个专业的金融术语标准化助手。"},
                {"role": "user", "content": prompt}
            ]

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.1,
                max_tokens=512
            )

            content = response.choices[0].message.content.strip()
            
            # 清理可能的 Markdown 格式
            if content.startswith("```json"):
                content = content.replace("```json", "", 1).rsplit("```", 1)[0].strip()
            elif content.startswith("```"):
                content = content.replace("```", "", 1).rsplit("```", 1)[0].strip()

            result = json.loads(content)
            return result

        except Exception as e:
            logger.error(f"Error in standardize_entity: {str(e)}")
            raise
