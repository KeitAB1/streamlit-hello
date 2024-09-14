import pandas as pd
from sqlalchemy import create_engine

# MySQL 连接配置
user = 'root'
password = 'root'
host = 'localhost'
port = '3306'
database = 'steel_data_db'

# 创建数据库连接引擎
engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}')

# 读取 CSV 文件
file_path = 'data/steel_data06.csv'  # 修改为你的文件路径
df = pd.read_csv(file_path)

# 清理列名
df.columns = df.columns.str.strip().str.replace(' ', '_')

# 插入数据到 MySQL 数据库
df.to_sql('steel_data06', con=engine, if_exists='append', index=False)

print("数据插入成功!")
