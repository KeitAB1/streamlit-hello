import streamlit as st
import requests  # 假设通过API调用模型

# 设置API的URL
MODEL_API_URL = "http://your-model-api-url.com/predict"  # 替换为实际模型API

# 初始化对话记录
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# 标题和说明
st.title("钢板知识问答系统")
st.write("输入你的问题，获取与钢板相关的专业解答。")

# 展示聊天记录
for message in st.session_state['messages']:
    st.write(message)

# 用户输入框
user_input = st.text_input("请输入你的问题：")

# 当用户输入后，调用模型并展示回复
if user_input:
    # 将用户的输入发送到模型API
    response = requests.post(MODEL_API_URL, json={"question": user_input})

    # 获取模型的回复
    if response.status_code == 200:
        model_reply = response.json().get("answer", "抱歉，我无法回答这个问题。")
    else:
        model_reply = "模型服务暂时不可用，请稍后再试。"

    # 将对话记录保存到 session_state 中
    st.session_state['messages'].append(f"用户: {user_input}")
    st.session_state['messages'].append(f"模型: {model_reply}")

    # 刷新界面以显示新的对话记录
    st.experimental_rerun()

# 添加一些钢板相关的知识说明
st.sidebar.title("钢板相关知识")
st.sidebar.write("""
- 钢板种类：冷轧钢板、热轧钢板等。
- 常见用途：建筑、造船、汽车制造等。
- 钢板的性能指标：抗拉强度、屈服强度、延展性等。
""")
