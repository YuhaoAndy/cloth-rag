import time # 导入time模块，用于模拟上传时间

import streamlit as st 
from knowledge_base import KnowledgeBaseService 

st.set_page_config(page_title="知识库更新", page_icon="📚", layout="centered")
st.title("知识库更新服务")
st.caption("上传文档后自动切片并写入向量库")

# file_uploader
uploader_file = st.file_uploader(
    "请上传知识文档（txt / md）",
    type=["txt", "md"],
    accept_multiple_files=False,    # False表示仅接受一个文件的上传
)

# session_state就是一个字典
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


if uploader_file is not None:
    # 提取文件的信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024    # KB

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")

    # get_value -> bytes -> decode('utf-8')
    text = uploader_file.getvalue().decode("utf-8", errors="ignore")

    with st.spinner("载入知识库中。。。"):       # 在spinner内的代码执行过程中，会有一个转圈动画
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)



