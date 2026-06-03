from langchain_chroma import Chroma
import config_data as config
from langchain_core.documents import Document


class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def _keyword_overlap_score(self, query: str, text: str) -> float:
        """简单关键词重叠打分，避免纯向量检索在业务词上的偏差。"""
        query_tokens = {token for token in query.lower().split() if token}
        if not query_tokens:
            return 0.0
        text_tokens = set(text.lower().split())
        return len(query_tokens.intersection(text_tokens)) / len(query_tokens)

    def retrieve_documents(self, query: str) -> list[Document]:
        """
        混合检索：
        1) similarity 检索
        2) mmr 检索（提升多样性）
        3) 去重 + 轻量关键词重排
        """
        similarity_docs = self.vector_store.similarity_search(query, k=config.top_k)
        mmr_docs = self.vector_store.max_marginal_relevance_search(query, k=config.mmr_k)
        merged_docs = similarity_docs + mmr_docs

        unique_docs = {}
        for doc in merged_docs:
            key = f"{doc.page_content}|{doc.metadata}"
            unique_docs[key] = doc

        reranked = sorted(
            unique_docs.values(),
            key=lambda d: self._keyword_overlap_score(query, d.page_content),
            reverse=True,
        )
        return reranked[: config.hybrid_top_k]


if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    service = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4"))
    res = service.retrieve_documents("我的体重180斤，尺码推荐")
    print(res)
