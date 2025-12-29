from datetime import datetime
import logging
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

logger = logging.getLogger(__name__)

class ChunkingService:
    """
    文本分块服务，提供多种文本分块策略
    
    该服务支持以下分块方法：
    - by_pages: 按页面分块，每页作为一个块
    - fixed_size: 按固定大小分块
    - by_paragraphs: 按段落分块
    - by_sentences: 按句子分块
    - recursive: 递归字符分块 (LangChain)
    - markdown: 按 Markdown 标题分块
    - token: 按 Token 数量分块
    """
    
    def chunk_text(self, text: str, method: str, metadata: dict, page_map: list = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> dict:
        """
        将文本按指定方法分块
        
        Args:
            text: 原始文本内容
            method: 分块方法
            metadata: 文档元数据
            page_map: 页面映射列表
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            包含分块结果的文档数据结构
        """
        try:
            if not page_map and method != "markdown":
                raise ValueError("Page map is required for non-markdown chunking.")
            
            chunks = []
            total_pages = len(page_map) if page_map else 0
            
            if method == "by_pages":
                chunks = self._chunk_by_pages(page_map)
            
            elif method == "fixed_size":
                chunks = self._chunk_fixed_size(page_map, chunk_size, chunk_overlap)
            
            elif method == "by_paragraphs":
                chunks = self._chunk_by_separator(page_map, "\n\n")
            
            elif method == "by_sentences":
                chunks = self._chunk_recursive(page_map, chunk_size, chunk_overlap, separators=[". ", "! ", "? ", "\n"])
                
            elif method == "recursive":
                chunks = self._chunk_recursive(page_map, chunk_size, chunk_overlap)
                
            elif method == "markdown":
                chunks = self._chunk_markdown(text, metadata)
                
            elif method == "token":
                chunks = self._chunk_token(page_map, chunk_size, chunk_overlap)
                
            else:
                raise ValueError(f"Unsupported chunking method: {method}")

            # 创建标准化的文档数据结构
            document_data = {
                "filename": metadata.get("filename", ""),
                "total_chunks": len(chunks),
                "total_pages": total_pages,
                "loading_method": metadata.get("loading_method", ""),
                "chunking_method": method,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "timestamp": datetime.now().isoformat(),
                "chunks": chunks
            }
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error in chunk_text: {str(e)}")
            raise

    def _chunk_by_pages(self, page_map: list) -> list:
        chunks = []
        for page_data in page_map:
            chunks.append({
                "content": page_data['text'],
                "metadata": {
                    "chunk_id": len(chunks) + 1,
                    "page_number": page_data['page'],
                    "page_range": str(page_data['page']),
                    "word_count": len(page_data['text'].split())
                }
            })
        return chunks

    def _chunk_fixed_size(self, page_map: list, chunk_size: int, chunk_overlap: int) -> list:
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return self._apply_splitter(page_map, splitter)

    def _chunk_by_separator(self, page_map: list, separator: str) -> list:
        chunks = []
        for page_data in page_map:
            paras = [p.strip() for p in page_data['text'].split(separator) if p.strip()]
            for para in paras:
                chunks.append({
                    "content": para,
                    "metadata": {
                        "chunk_id": len(chunks) + 1,
                        "page_number": page_data['page'],
                        "page_range": str(page_data['page']),
                        "word_count": len(para.split())
                    }
                })
        return chunks

    def _chunk_recursive(self, page_map: list, chunk_size: int, chunk_overlap: int, separators: list = None) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""]
        )
        return self._apply_splitter(page_map, splitter)

    def _chunk_token(self, page_map: list, chunk_size: int, chunk_overlap: int) -> list:
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return self._apply_splitter(page_map, splitter)

    def _chunk_markdown(self, text: str, metadata: dict) -> list:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = splitter.split_text(text)
        
        chunks = []
        for i, split in enumerate(md_header_splits, 1):
            chunks.append({
                "content": split.page_content,
                "metadata": {
                    "chunk_id": i,
                    **split.metadata,
                    "word_count": len(split.page_content.split())
                }
            })
        return chunks

    def _apply_splitter(self, page_map: list, splitter) -> list:
        chunks = []
        for page_data in page_map:
            split_texts = splitter.split_text(page_data['text'])
            for text in split_texts:
                if text.strip():
                    chunks.append({
                        "content": text.strip(),
                        "metadata": {
                            "chunk_id": len(chunks) + 1,
                            "page_number": page_data['page'],
                            "page_range": str(page_data['page']),
                            "word_count": len(text.split())
                        }
                    })
        return chunks
