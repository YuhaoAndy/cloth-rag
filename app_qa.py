import streamlit as st

import config_data as config
from file_history_store import get_history
from rag import RagService

st.set_page_config(page_title="智能服装客服", page_icon="🧥", layout="wide")
st.title("智能服装客服（RAG + 业务工具）")
st.caption("支持：商品推荐、尺码建议、洗护与售后问答、知识库检索")
st.divider()

with st.sidebar:
    st.subheader("会话设置")
    session_id = st.text_input("Session ID", value=config.default_session_id)
    if st.button("清空该会话历史"):
        get_history(session_id).clear()
        st.success(f"会话 {session_id} 历史已清空")
    st.markdown("---")
    st.subheader("示例提问")
    st.write("1. 我170cm/65kg，通勤风预算300元怎么选？")
    st.write("2. 针织毛衣怎么洗？会起球吗？")
    st.write("3. 这件衣服不合适可以退换吗？")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好，我可以帮你做尺码建议、搭配推荐和售后问答。"}
    ]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("请输入你的问题")
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    session_config = {"configurable": {"session_id": session_id}}
    ai_res_list = []

    with st.spinner("AI思考中..."):
        response_stream = st.session_state["rag"].stream(prompt, session_config)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(response_stream, ai_res_list))

    st.session_state["messages"].append({"role": "assistant", "content": "".join(ai_res_list)})
