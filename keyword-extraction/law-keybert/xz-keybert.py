import json
from keybert import KeyBERT
import jieba

# 加载 JSON 文件
input_file = '../../dataset/datademo/code/xz_with_word_count.json'
output_file = 'xz_with_keywords.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化 KeyBERT 模型
kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

# 创建保存结果的列表
results = []

# 处理每一个文本
for item in data:
    label = item.get('label')
    text = item.get('text')
    words = item.get('words')
    # 使用 jieba 分词
    doc = " ".join(jieba.cut(text))
    
    # 提取关键字
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), top_n=20, stop_words=None)
    
    # 将原文、标签和提取的关键字一起保存
    result = {
        'label': label,
        'text': text,
        'words':words,
        'keywords': keywords
    }
    results.append(result)

# 将结果保存到新的 JSON 文件中
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"结果已保存到 {output_file}")