import os
import pandas as pd

def clear_csv(file_path):
    """清空CSV文件内容"""
    if os.path.exists(file_path):
        open(file_path, 'w').close()  # 打开并清空文件内容


def is_csv_empty(file_path):
    """检查CSV文件是否为空"""
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0  # 文件存在且大小为0时返回True


def clear_folder(folder_path):
    """清空图片文件夹"""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)  # 删除文件夹中的所有图片


# 创建文件夹（如果不存在）
def ensure_directory_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def append_to_csv(data, file_path):
    """将识别结果追加到CSV文件中"""
    df = pd.DataFrame(data)  # 将数据转换为DataFrame格式
    # 如果文件存在，则追加数据；否则创建新文件并写入表头
    if os.path.exists(file_path):
        if is_csv_empty(file_path):
            df.to_csv(file_path, mode='a', header=True, index=False)  # 追加模式，不写入表头
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)  # 追加模式，不写入表头
    else:
        df.to_csv(file_path, index=False)  # 如果文件不存在，创建文件并写入表头
        df.to_csv(file_path, mode='a', header=False, index=False)  # 再次追加数据以防止错误
