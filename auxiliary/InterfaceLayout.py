import streamlit as st
import base64
import os


# 将图片转换为 base64 格式
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 自定义一个函数来创建带有图标和标题的组合,并自定义标题大小(默认24px)
def display_icon_with_header(icon_path, header_text, font_size="24px"):
    if os.path.exists(icon_path):
        # 将图片转换为 Base64 格式
        img_base64 = image_to_base64(icon_path)
        # 使用 HTML 和 CSS 实现图标和标题的对齐，同时支持自定义字体大小
        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_base64}" width="40" style="margin-right: 10px;">
                <h3 style="margin: 0; font-size: {font_size};">{header_text}</h3>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(f"<h3 style='font-size: {font_size};'>{header_text}</h3>", unsafe_allow_html=True)