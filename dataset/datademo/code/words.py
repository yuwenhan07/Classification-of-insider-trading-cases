# 为每条记录添加"words"字段
import json

# 加载JSON文件
with open('xz.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 为每条记录添加"词数"字段
for record in data:
    text = record.get('text', '')
    
    # 计算text字段中的字数，使用len统计字符数
    word_count = len(text)
    
    # 添加"词数"字段
    record['words'] = word_count

# 保存修改后的JSON文件
with open('xz_with_word_count.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("已成功为每条记录添加词数字段！")