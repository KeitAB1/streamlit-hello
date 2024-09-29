import streamlit as st

# 从 secrets.toml 文件中获取 OpenWeboi 链接
openweboi_url = ""

# 设置页面标题
st.title("Steel Plate Intelligent Q&A System")

# 显示英文提示信息
st.write("The Steel Plate Intelligent Q&A feature is currently being updated. However, you can access it through OpenWeboi via internal network tunneling.")

# 添加一个按钮，点击后跳转到 OpenWeboi 的链接
if st.button("Enter via OpenWeboi"):
    if openweboi_url:
        st.write("Redirecting to OpenWeboi...")
        st.markdown(f"[Click here to access OpenWeboi]({openweboi_url})")
    else:
        st.error("OpenWeboi URL not found. Please check your secrets.toml file.")
