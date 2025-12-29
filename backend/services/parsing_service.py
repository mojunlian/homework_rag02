import logging
from typing import Dict, List
import fitz  # PyMuPDF
import pandas as pd
from datetime import datetime
import os
import pdfplumber
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.md import partition_md

logger = logging.getLogger(__name__)

class ParsingService:
    """
    文档解析服务类
    
    该类提供多种解析策略来提取和构建文档内容，包括：
    - 全文提取
    - 逐页解析
    - 基于标题的分段
    - 文本、表格和图像混合解析 (PDF/MD)
    """

    def parse_pdf(self, text: str, method: str, metadata: dict, page_map: list = None, file_path: str = None) -> dict:
        """保持向后兼容的别名"""
        return self.parse_file(text, method, metadata, page_map, file_path)

    def parse_file(self, text: str, method: str, metadata: dict, page_map: list = None, file_path: str = None) -> dict:
        """
        使用指定方法解析文档

        参数:
            text (str): 文档文本内容
            method (str): 解析方法 ('all_text', 'by_pages', 'by_titles', 'text_and_tables', 'hi_res')
            metadata (dict): 文档元数据
            page_map (list): 页面映射列表
            file_path (str): 文件物理路径，用于 hi_res 解析

        返回:
            dict: 解析后的文档数据
        """
        try:
            parsed_content = []
            total_pages = len(page_map) if page_map else 0
            
            if method == "all_text" and page_map:
                parsed_content = self._parse_all_text(page_map)
            elif method == "by_pages" and page_map:
                parsed_content = self._parse_by_pages(page_map)
            elif method == "by_titles" and page_map:
                parsed_content = self._parse_by_titles(page_map)
            elif method == "text_and_tables" and file_path:
                parsed_content = self._parse_with_pdfplumber(file_path)
            elif method == "hi_res" and file_path:
                parsed_content = self._parse_with_unstructured(file_path)
            else:
                # Fallback to simple page map if nothing else works
                if page_map:
                    parsed_content = self._parse_by_pages(page_map)
                else:
                    raise ValueError(f"Unsupported parsing method: {method} or missing required data.")
                
            # Create document-level metadata
            document_data = {
                "metadata": {
                    "filename": metadata.get("filename", ""),
                    "total_pages": total_pages,
                    "parsing_method": method,
                    "timestamp": datetime.now().isoformat()
                },
                "content": parsed_content
            }
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error in parse_file: {str(e)}")
            raise

    def _parse_all_text(self, page_map: list) -> list:
        return [{
            "type": "Text",
            "content": page["text"],
            "page": page["page"],
            "metadata": page.get("metadata", {})
        } for page in page_map]

    def _parse_by_pages(self, page_map: list) -> list:
        parsed_content = []
        for page in page_map:
            parsed_content.append({
                "type": "Page",
                "page": page["page"],
                "content": page["text"],
                "metadata": page.get("metadata", {})
            })
        return parsed_content

    def _parse_by_titles(self, page_map: list) -> list:
        parsed_content = []
        current_title = "Introduction"
        current_content = []

        for page in page_map:
            lines = page["text"].split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # Heuristic for title: short, not ending with punctuation, maybe upper
                if len(line) < 100 and (line.isupper() or line[0].isdigit()):
                    if current_content:
                        parsed_content.append({
                            "type": "section",
                            "title": current_title,
                            "content": '\n'.join(current_content),
                            "page": page["page"]
                        })
                    current_title = line
                    current_content = []
                else:
                    current_content.append(line)

        if current_content:
            parsed_content.append({
                "type": "section",
                "title": current_title,
                "content": '\n'.join(current_content),
                "page": page_map[-1]["page"] if page_map else 0
            })

        return parsed_content

    def _parse_with_pdfplumber(self, file_path: str) -> list:
        """使用 pdfplumber 提取文本和表格"""
        parsed_content = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    # 1. 提取表格
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            parsed_content.append({
                                "type": "table",
                                "content": df.to_markdown(),
                                "page": i,
                                "metadata": {"rows": len(df), "cols": len(df.columns)}
                            })
                    
                    # 2. 提取文本
                    text = page.extract_text()
                    if text:
                        parsed_content.append({
                            "type": "text",
                            "content": text,
                            "page": i
                        })
            return parsed_content
        except Exception as e:
            logger.error(f"pdfplumber parsing error: {str(e)}")
            return []

    def _parse_with_unstructured(self, file_path: str) -> list:
        """使用 unstructured hi_res 提取各种元素（含图像占位）"""
        parsed_content = []
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            elements = []
            
            if file_ext == ".pdf":
                try:
                    logger.info("Attempting hi_res parsing with unstructured...")
                    elements = partition_pdf(
                        filename=file_path,
                        strategy="hi_res",
                        infer_table_structure=True,
                        chunking_strategy="by_title"
                    )
                except Exception as e:
                    if "locate the file on the Hub" in str(e) or "Internet connection" in str(e):
                        logger.warning(f"Hi-res parsing failed due to network issues (model download failed). Falling back to 'fast' strategy. Error: {str(e)}")
                        elements = partition_pdf(
                            filename=file_path,
                            strategy="fast",
                            chunking_strategy="by_title"
                        )
                    else:
                        raise e
            elif file_ext == ".md":
                elements = partition_md(filename=file_path)
            else:
                from unstructured.partition.auto import partition
                elements = partition(filename=file_path)

            for elem in elements:
                elem_type = elem.category.lower() if hasattr(elem, 'category') else "text"
                page_number = elem.metadata.page_number if hasattr(elem.metadata, 'page_number') else 1
                
                content = str(elem)
                if elem_type == "table":
                    if hasattr(elem, 'metadata') and hasattr(elem.metadata, 'text_as_html'):
                        # Optionally convert HTML table to markdown
                        content = elem.metadata.text_as_html
                
                parsed_content.append({
                    "type": elem_type,
                    "content": content,
                    "page": page_number,
                    "metadata": elem.metadata.__dict__ if hasattr(elem, 'metadata') else {}
                })
            return parsed_content
        except Exception as e:
            logger.error(f"Unstructured parsing error: {str(e)}")
            return []