import os

md5_path = os.getenv("MD5_PATH", "./md5.text")

# Chroma
collection_name = os.getenv("COLLECTION_NAME", "rag")
persist_directory = os.getenv("PERSIST_DIRECTORY", "./chroma_db")

# splitter
chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_split_char_number = int(os.getenv("MAX_SPLIT_CHAR_NUMBER", "1000"))

# retrieval
top_k = int(os.getenv("TOP_K", "4"))
mmr_k = int(os.getenv("MMR_K", "4"))
hybrid_top_k = int(os.getenv("HYBRID_TOP_K", "5"))

embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
chat_model_name = os.getenv("CHAT_MODEL_NAME", "qwen-plus")

default_session_id = os.getenv("DEFAULT_SESSION_ID", "user_001")
session_config = {
    "configurable": {
        "session_id": default_session_id,
    }
}
