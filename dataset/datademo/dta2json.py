# 用于将 .dta 文件转换为 JSON 格式
import pandas as pd

# 读取 .dta 文件
file_path = '结构化数据.dta'
df = pd.read_stata(file_path)

# 将 DataFrame 转换为 JSON 格式
json_data = df.to_json(orient='records', force_ascii=False)

# 保存 JSON 文件
with open('output.json', 'w', encoding='utf-8') as f:
    f.write(json_data)

print("转换完成！")