# keybert
## 原理和一些参数说明
KeyBERT(Sharma, P., & Li, Y. (2019). Self-Supervised Contextual Keyword and Keyphrase Retrieval with Self-Labelling)，提出了一个利用bert快速提取关键词的方法。
### 原理
原理十分简单：首先使用 BERT 提取文档嵌入以获得文档级向量表示。随后，为N-gram 词/短语提取词向量，然后，我们使用余弦相似度来找到与文档最相似的单词/短语。最后可以将最相似的词识别为最能描述整个文档的词。
### 参数
其中，有几个参数：
「**keyphrase_ngram_range：**」 默认(1, 1)，表示单个词， 如"抗美援朝", "纪念日"是孤立的两个词；(2, 2)表示考虑词组， 如出现有意义的词组 "抗美援朝 纪念日；(1, 2)表示同时考虑以上两者情况；
**top_n:**显示前n个关键词，默认5；
「**use_maxsum:**」 默认False;是否使用Max Sum Similarity作为关键词提取标准；
「**use_mmr:**」 默认False;是否使用Maximal Marginal Relevance (MMR) 作为关键词提取标准；
「**diversity:**」 如果use_mmr=True，可以设置该参数。参数取值范围从0到1；

## 优缺点
### 优点
Keybert基于一种假设，关键词与文档在语义表示上是一致的，利用bert的编码能力，能够得到较好的结果。
**主要是在语义表示上，会有较好的效果**
### 缺点
但缺点很明显：
首先，不同的语义编码模型会产生不同的结果，这个比较重要。
此外，由于bert只能接受限定长度的文本，例如512个字，这个使得我们在处理长文本时，需要进一步加入摘要提取等预处理措施，这无疑会带来精度损失。