from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

import config_data as config
from business_tools import BusinessTools
from file_history_store import get_history
from vector_stores import VectorStoreService


class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.business_tools = BusinessTools()
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是服装电商智能客服，请优先基于知识库与业务规则作答。"
                    "答案要求：专业、简洁、可执行。"
                    "若用户问题不完整，先给可落地建议，再提示其补充关键信息。",
                ),
                ("system", "知识库上下文:\n{retrieved_context}"),
                ("system", "业务工具上下文:\n{business_context}"),
                ("system", "可引用来源:\n{source_hints}"),
                MessagesPlaceholder("history"),
                ("user", "{input}"),
            ]
        )
        self.chain = self._build_chain()

    @staticmethod
    def _format_documents(docs: list[Document]) -> str:
        if not docs:
            return "无相关参考资料"

        lines = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content.strip().replace("\n", " ")
            lines.append(f"[{idx}] 来源:{source} | 内容:{content}")
        return "\n".join(lines)

    @staticmethod
    def _format_source_hints(docs: list[Document]) -> str:
        if not docs:
            return "无"
        unique_sources = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if source not in unique_sources:
                unique_sources.append(source)
        return "\n".join(f"- {s}" for s in unique_sources)

    def _retrieve_bundle(self, query: str) -> dict:
        docs = self.vector_service.retrieve_documents(query)
        return {
            "retrieved_context": self._format_documents(docs),
            "source_hints": self._format_source_hints(docs),
        }

    @staticmethod
    def _merge_inputs(payload: dict) -> dict:
        retrieval_bundle = payload["retrieval_bundle"]
        original_input = payload["input"]
        return {
            "input": original_input["input"],
            "history": original_input["history"],
            "retrieved_context": retrieval_bundle["retrieved_context"],
            "source_hints": retrieval_bundle["source_hints"],
            "business_context": payload["business_context"],
        }

    def _build_chain(self):
        chain = (
            {
                "input": RunnablePassthrough(),
                "retrieval_bundle": RunnableLambda(lambda x: self._retrieve_bundle(x["input"])),
                "business_context": RunnableLambda(
                    lambda x: self.business_tools.build_business_context(x["input"])
                ),
            }
            | RunnableLambda(self._merge_inputs)
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        return RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def invoke(self, user_input: str, session_config: dict):
        return self.chain.invoke({"input": user_input}, session_config)

    def stream(self, user_input: str, session_config: dict):
        return self.chain.stream({"input": user_input}, session_config)


if __name__ == "__main__":
    service = RagService()
    res = service.invoke("我身高170cm体重65kg，想要通勤风，预算300元，怎么选？", config.session_config)
    print(res)
