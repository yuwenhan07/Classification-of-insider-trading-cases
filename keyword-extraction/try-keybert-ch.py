from keybert import KeyBERT
import jieba

# 中文测试
doc = "刚刚，理论计算机科学家、UT Austin 教授、量子计算先驱 Scott Aaronson 因其「对量子计算的开创性贡献」被授予 2020 年度 ACM 计算奖。在获奖公告中，ACM 表示：「量子计算的意义在于利用量子物理学定律解决传统计算机无法解决或无法在合理时间内解决的难题。Aaronson 的研究展示了计算复杂性理论为量子物理学带来的新视角，并清晰地界定了量子计算机能做什么以及不能做什么。他在推动量子优越性概念发展的过程起到了重要作用，奠定了许多量子优越性实验的理论基础。这些实验最终证明量子计算机可以提供指数级的加速，而无需事先构建完整的容错量子计算机。」 ACM 主席 Gabriele Kotsis 表示：「几乎没有什么技术拥有和量子计算一样的潜力。尽管处于职业生涯的早期，但 Scott Aaronson 因其贡献的广度和深度备受同事推崇。他的研究指导了这一新领域的发展，阐明了它作为领先教育者和卓越传播者的可能性。值得关注的是，他的贡献不仅限于量子计算，同时也在诸如计算复杂性理论和物理学等领域产生了重大影响。」"
doc = " ".join(jieba.cut(doc))
kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

print("naive ...")
keywords = kw_model.extract_keywords(doc)
print(keywords)

print("\nkeyphrase_ngram_range ...")
keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=None)
print(keywords)

print("\nhighlight ...")
keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), highlight=True)
print(keywords)

# 为了使结果多样化，我们将 2 x top_n 与文档最相似的词/短语。
# 然后，我们从 2 x top_n 单词中取出所有 top_n 组合，并通过余弦相似度提取彼此最不相似的组合。
print("\nuse_maxsum ...")
keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english',
                              use_maxsum=True, nr_candidates=20, top_n=5)
print(keywords)

# 为了使结果多样化，我们可以使用最大边界相关算法(MMR)
# 来创建同样基于余弦相似度的关键字/关键短语。 具有高度多样性的结果：
print("\nhight diversity ...")
keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english',
                        use_mmr=True, diversity=0.7)
print(keywords)

# 低多样性的结果
print("\nlow diversity ...")
keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english',
                        use_mmr=True, diversity=0.2)
print(keywords)

'''
参数：
docs：要提取关键字/关键短语的文档
candidates：要使用的候选关键字/关键短语，而不是从文档中提取它们
keyphrase_ngram_range：提取的关键字/关键短语的长度（以字为单位）
stop_words：要从文档中删除的停用词
top_n：返回前 n 个关键字/关键短语
min_df：如果需要提取多个文档的关键字，则一个单词在所有文档中的最小文档频率
use_maxsum: 是否使用 Max Sum Similarity 来选择keywords/keyphrases
use_mmr：是否使用最大边际相关性（MMR）进行关键字/关键短语的选择
diversity：如果 use_mmr 设置为 True，结果的多样性在 0 和 1 之间
nr_candidates：如果 use_maxsum 设置为 True，要考虑的候选数
vectorizer：从 scikit-learn 传入你自己的 CountVectorizer
highlight：是否打印文档并突出显示其关键字/关键短语。注意：如果传递了多个文档，这将不起作用。
'''