import math
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.services.knowledge_base_store import knowledge_base_store
from app.services.knowledge_global_config_store import knowledge_global_config_store
from app.services.openai_compat import normalize_openai_base_url

try:
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter

    LLAMAINDEX_AVAILABLE = True
except Exception:
    Document = Any
    VectorStoreIndex = Any
    SentenceSplitter = Any
    LLAMAINDEX_AVAILABLE = False


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", (text or "").lower())


def _normalize_embedding_api_base(api_base: str) -> str:
    return normalize_openai_base_url(api_base)


@dataclass
class SearchHit:
    doc_id: str
    title: str
    chunk: str
    score: float
    metadata: Dict[str, Any]


class KnowledgeIndexService:
    """知识库索引服务，负责构建和管理知识库的向量索引。"""

    def __init__(self) -> None:
        """初始化知识库索引服务。"""
        # 可重入锁，确保多线程环境下的并发安全
        self._lock = threading.RLock()
        # 索引缓存：key 为知识库ID，value 为三元组 (signature, index, fallback_chunks)
        # - signature: 知识库签名，用于检测内容变化
        # - index: VectorStoreIndex 对象（llama_index 向量索引）
        # - fallback_chunks: 备用文本块列表，用于回退搜索
        self._cache: Dict[str, Tuple[str, Any, List[Dict[str, Any]]]] = {}

    @staticmethod
    def _signature(kb: Dict[str, Any]) -> str:
        doc_parts = []
        for doc in kb.get("documents", []):
            doc_parts.append(f"{doc.get('id')}:{doc.get('updated_at')}:{len(doc.get('content', ''))}")
        return "|".join(
            [
                str(kb.get("updated_at")),
                str(kb.get("chunk_size")),
                str(kb.get("chunk_overlap")),
                *doc_parts,
            ]
        )

    @staticmethod
    def _fallback_chunks(kb: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将知识库文档分割成小块文本（chunks），用于回退搜索。
        
        该方法实现滑动窗口切分算法：
        - 使用指定的 chunk_size 和 chunk_overlap 参数
        - 文本块之间有重叠，以保持上下文连续性
        - 短文本直接作为单个块，长文本按滑动窗口切分
        
        Args:
            kb: 知识库字典，包含 documents、chunk_size、chunk_overlap 等字段
            
        Returns:
            List[Dict[str, Any]]: 切分后的文本块列表，每个块包含：
                - doc_id: 原始文档ID
                - title: 文档标题
                - chunk: 文本块内容
                - metadata: 文档元数据
        """
        chunks: List[Dict[str, Any]] = []
        
        # 获取切分参数，使用默认值
        chunk_size = int(kb.get("chunk_size") or 512)  # 每个块的大小（字符数）
        overlap = int(kb.get("chunk_overlap") or 50)   # 块之间的重叠大小
        step = max(1, chunk_size - overlap)            # 滑动步长
        
        # 遍历所有文档
        for doc in kb.get("documents", []):
            text = doc.get("content") or ""
            if not text:
                continue
            
            # 如果文本长度小于等于块大小，直接作为一个块
            if len(text) <= chunk_size:
                chunks.append(
                    {
                        "doc_id": doc.get("id", ""),
                        "title": doc.get("title", ""),
                        "chunk": text,
                        "metadata": doc.get("metadata") or {},
                    }
                )
                continue
            
            # 滑动窗口切分长文本
            for start in range(0, len(text), step):
                piece = text[start : start + chunk_size]
                if not piece:
                    continue
                chunks.append(
                    {
                        "doc_id": doc.get("id", ""),
                        "title": doc.get("title", ""),
                        "chunk": piece,
                        "metadata": doc.get("metadata") or {},
                    }
                )
        
        return chunks

    def _build_index(self, kb: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        构建知识库的向量索引。
        
        该方法负责：
        1. 生成备用文本块（fallback_chunks）用于回退搜索
        2. 如果 llama_index 不可用，直接返回回退数据
        3. 使用 SentenceSplitter 进行智能文本切分（基于句子边界）
        4. 将文档转换为 llama_index 的 Document 对象
        5. 使用指定的嵌入模型构建向量索引
        
        Args:
            kb: 知识库字典，包含 documents、chunk_size、chunk_overlap、embedding_model 等字段
            
        Returns:
            Tuple[Any, List[Dict[str, Any]]]: 元组，包含：
                - index: VectorStoreIndex 对象（如果 llama_index 不可用则为 None）
                - fallback_chunks: 备用文本块列表，用于回退搜索
        """
        # 先生成备用文本块（无论是否使用 llama_index 都需要）
        fallback_chunks = self._fallback_chunks(kb)
        
        # 如果 llama_index 不可用，直接返回回退数据
        if not LLAMAINDEX_AVAILABLE:
            return None, fallback_chunks
        
        # 获取切分参数
        chunk_size = int(kb.get("chunk_size") or 512)
        overlap = int(kb.get("chunk_overlap") or 50)
        
        # 创建智能句子切分器（基于句子边界进行切分，比简单的字符切分更智能）
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        
        # 将文档转换为 llama_index 的 Document 对象
        docs = [
            Document(
                text=(doc.get("content") or ""),
                metadata={
                    "doc_id": doc.get("id", ""),
                    "title": doc.get("title", ""),
                    **(doc.get("metadata") or {}),
                },
            )
            for doc in kb.get("documents", [])
            if (doc.get("content") or "").strip()  # 过滤空内容的文档
        ]
        
        # 如果没有有效文档，返回回退数据
        if not docs:
            return None, fallback_chunks
        
        # 构建嵌入模型
        embed_model = self._build_embed_model(kb)
        
        # 使用 VectorStoreIndex 构建向量索引
        if embed_model is not None:
            index = VectorStoreIndex.from_documents(
                docs,
                transformations=[splitter],
                embed_model=embed_model,
            )
        else:
            # 如果没有指定嵌入模型，使用默认配置
            index = VectorStoreIndex.from_documents(docs, transformations=[splitter])
        
        return index, fallback_chunks

    @staticmethod
    def _build_embed_model(kb: Dict[str, Any]) -> Any:
        from app.services.embedding_model_store import embedding_model_store
        models = embedding_model_store.list_models()
        if not models:
            return None
        
        target_model = None
        kb_model_val = kb.get("embedding_model")
        if kb_model_val:
            # Try matching by ID first, then by model name
            target_model = next((m for m in models if m.get("id") == kb_model_val), None)
            if not target_model:
                target_model = next((m for m in models if m.get("model") == kb_model_val), None)
        
        if not target_model:
            # Fallback to the first model
            target_model = models[0]
            
        api_base = target_model.get("api_base")
        api_key = target_model.get("api_key")
        model_name = target_model.get("model")
        
        if not api_base or not api_key or not model_name:
            return None
        api_base = _normalize_embedding_api_base(api_base)
        try:
            from llama_index.embeddings.openai_like import OpenAILikeEmbedding

            return OpenAILikeEmbedding(
                model_name=model_name,
                api_base=api_base,
                api_key=api_key,
                embed_batch_size=10,
            )
        except Exception:
            try:
                from llama_index.embeddings.openai import OpenAIEmbedding

                return OpenAIEmbedding(
                    model_name=model_name,
                    api_base=api_base,
                    api_key=api_key,
                    embed_batch_size=10,
                )
            except Exception:
                return None

    def reindex(self, kb_id: str) -> Dict[str, Any]:
        kb = knowledge_base_store.get(kb_id)
        if not kb:
            raise ValueError("Knowledge base not found")
        with self._lock:
            signature = self._signature(kb)
            index, fallback_chunks = self._build_index(kb)
            self._cache[kb_id] = (signature, index, fallback_chunks)
        return {
            "kb_id": kb_id,
            "status": "ok",
            "documents": len(kb.get("documents", [])),
            "engine": "llamaindex" if LLAMAINDEX_AVAILABLE and index is not None else "fallback",
        }

    @staticmethod
    def _fallback_search(query: str, chunks: List[Dict[str, Any]], top_k: int) -> List[SearchHit]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        scored: List[SearchHit] = []
        for chunk_item in chunks:
            c_tokens = _tokenize(chunk_item.get("chunk", ""))
            if not c_tokens:
                continue
            overlap = sum(1 for t in c_tokens if t in q_set)
            if overlap == 0:
                continue
            score = overlap / math.sqrt(len(c_tokens))
            scored.append(
                SearchHit(
                    doc_id=chunk_item.get("doc_id", ""),
                    title=chunk_item.get("title", ""),
                    chunk=chunk_item.get("chunk", ""),
                    score=float(score),
                    metadata=chunk_item.get("metadata") or {},
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def search(self, kb_id: str, query: str, top_k: int | None = None) -> Dict[str, Any]:
        kb = knowledge_base_store.get(kb_id)
        if not kb:
            raise ValueError("Knowledge base not found")
        if not kb.get("documents"):
            return {"answer": "", "hits": []}
        effective_top_k = int(top_k or kb.get("top_k") or 3)
        with self._lock:
            signature = self._signature(kb)
            cached = self._cache.get(kb_id)
            if not cached or cached[0] != signature:
                index, fallback_chunks = self._build_index(kb)
                cached = (signature, index, fallback_chunks)
                self._cache[kb_id] = cached
            _, index, fallback_chunks = cached
        if index is None:
            hits = self._fallback_search(query=query, chunks=fallback_chunks, top_k=effective_top_k)
            answer = "\n\n".join(hit.chunk for hit in hits)
            return {
                "answer": answer,
                "hits": [hit.__dict__ for hit in hits],
            }
        retriever = index.as_retriever(similarity_top_k=effective_top_k)
        response_nodes = retriever.retrieve(query)
        hits: List[Dict[str, Any]] = []
        for node_with_score in response_nodes:
            node = getattr(node_with_score, "node", None)
            metadata = getattr(node, "metadata", {}) if node is not None else {}
            chunk_text = ""
            if node is not None and hasattr(node, "get_content"):
                chunk_text = node.get_content()
            elif node is not None:
                chunk_text = str(getattr(node, "text", ""))
            hits.append(
                {
                    "doc_id": metadata.get("doc_id", ""),
                    "title": metadata.get("title", ""),
                    "chunk": chunk_text,
                    "score": float(getattr(node_with_score, "score", 0.0) or 0.0),
                    "metadata": metadata,
                }
            )
        if not hits:
            fallback_hits = self._fallback_search(query=query, chunks=fallback_chunks, top_k=effective_top_k)
            return {
                "answer": "\n\n".join(hit.chunk for hit in fallback_hits),
                "hits": [hit.__dict__ for hit in fallback_hits],
            }
        answer = "\n\n".join(item.get("chunk", "") for item in hits if item.get("chunk"))
        return {"answer": answer, "hits": hits}


knowledge_index_service = KnowledgeIndexService()