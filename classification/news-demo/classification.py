# 加深理解，注释都是非常简单的，非常详细，一是希望自己能够彻底理解处理流程，
# 二是希望基础薄弱的读着看完能有所收获
# 来自：https://blog.csdn.net/weixin_45552370/article/details/119788420
import torch # pytor库，必用
import pandas as pd # pandas库是一个处理各种数据文件的库，类似于wps，可以打开，保存各种word，ppt等格式的文件
import torch.nn as nn #导入nn，这个库里面是一些全连接网络，池化层，这种神经网络中非常基本的一些模块，这里我们主要是用nn.linear是一个全连接层
from transformers import BertModel, BertTokenizer# transformer库是一个把各种预训练模型集成在一起的库，导入之后，你就可以选择性的使用自己想用的模型，这里使用的BERT模型。所以导入了bert模型，和bert的分词器，这里是对bert的使用，而不是bert自身的源码，如果有时间或者兴趣的话我会在另一篇文章写bert的源码实现
from sklearn.model_selection import train_test_split #sklearn是一个非常基础的机器学习库，里面都是一切基础工具，类似于聚类算法啊，逻辑回归算法啊，各种对数据处理的方法啊，这里我们使用的train_test_split方法，是把数据，一部分用作训练，一部分用作测试的划分数据的方法

# 第一部分
# 加载训练集,第一个参数文件位置，默认会以空格划分每行的内容，
# delimier参数设置的备选用制表符划分，
# 第三个参数 是不满足要求的行舍弃掉，正常情况下，每行都是一个序号，一个标签，一段文本，
# 如果不是这样的，我们就舍弃这一行

#train_set = pd.read_csv("./data/train.tsv",delimiter="\t",error_bad_lines=False)
train_set = pd.read_csv("./data/train.tsv",sep="\t",on_bad_lines='skip')
model_name = "hfl/chinese-bert-wwm" #我们是要使用bert模型，但是bert也有很多模型，比如处理英文的，处理法语的，我们这里使用处理中文，且全mask的方法，感兴趣可以看这里了解https://github.com/ymcui/Chinese-BERT-wwm，另外，如果手码代码出错了，可能是因为字符串打错了，fhl而不是hf1，是L而不是数字1
# 下面我们就获得了bert模型中的hfl/chinese-bert-wwm模型的模型以及模型分词方法，这里是原始模型，我们要使用微调，所以下面自写类
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 第二部分
# 采用bert微调策略，在反向传播时一同调整BERT和线性层的参数，使bert更加适合分类任务
# bert微调，在这里 就是bert（类的前四行）+一个全连接层（第五行 self.fc = nn.Linear(768,15)）组成一个全新模型，
class BertClassfication(nn.Module):#括号里面是继承什么，该类继承自nn.module，因为我们的模型在根本上是神经网络的，所以继承nn.module，继承它的基本属性是自然的了
    def __init__(self):
        # 前四行就是bert，第五行是全连接层，即bert微调，就是本文要使用的模型
        super(BertClassfication,self).__init__() # 调用父类 nn.Module 的构造函数，确保继承过来的功能能够正常初始化，比如神经网络的参数管理等。
        self.model_name = 'hfl/chinese-bert-wwm'
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(768,15)     #768取决于BERT结构，2-layer, 768-hidden, 12-heads, 110M parameters
        # 线性层  定义一个全连接层  一共15个分类，所以输出是15，输入是768，这个768是bert模型的输出，也就是最后一层的隐藏层的输出，这个768是固定的，因为bert模型的最后一层的隐藏层的输出是768，所以这里是768


# 前向传播过程，描述了数据从输入到输出的流程，这里是一个简单的文本分类任务，所以输入是文本，输出是15个分类的概率
    def forward(self,x):#这里的输入x是一个list,也就是输入文本x：RNG全队梦游失误频频不敌FW，后续淘汰赛成绩引人担忧，我这里是用一句话举例子，实际上的数据是很多很多句话（哈哈，好不专业，很多很多）
        # 这句话是对文本1.进行分词，2.进行编码可以这里理解一下https://www.cnblogs.com/zjuhaohaoxuexi/p/15135466.html
        # 第一个参数x自然也就是文本了
        # 第二个参数add_special_token，就是说要不要加入一些特殊字符，比如标记开头的cls，等等
        # 第三个参数就是最长长度，
        # 第四个参数是pad_to_max_len：就是说如果不够max_len的长度，就补长到最大长度
        # 还有一个参数这里没用到，就是比最长的长就减掉，第四第五个参数也就是长的减，短的补
        
        # 使用BERT的分词器对文本进行分词，编码，加入特殊字符，填充到最大长度  batch_encode_plus是一个批量编码的方法，可以一次性处理多个文本，返回的是一个字典，里面有input_ids，attention_mask等等，这里我们只用到了input_ids和attention_mask
        batch_tokenized = self.tokenizer.batch_encode_plus(x, add_special_tokens=True,
                                max_length=148, pad_to_max_length=True)      #tokenize、add special token、pad


        # 可以看到上一步的结果是好几个维度，（建议一遍写代码，一遍在jupyter里面调试，看看每一步的结果或者是什么形式）
        # 取出input_ids：这是对汉语的编码
        # attention_mask:这是对每个字是否mask的一个标记，原本的词的位置是1，如果由于词长不够max_len，用pad或者其他的填充了，该位置就是0，意味着在模型中不注意他，因为该位置是填充上的没有意义的字符，我们为什么要管他呢？
        
        # 转化为pytorch的tensor
        input_ids = torch.tensor(batch_tokenized['input_ids']) # 句子的token编码，表示文本被Bert模型所识别的输入序列
        attention_mask = torch.tensor(batch_tokenized['attention_mask']) # 掩码标志，1表示有效token，0表示填充token（pad部分）


        # 把上边两个输入bert模型，得到bert最后一层的隐藏层的输出
        hiden_outputs = self.model(input_ids,attention_mask=attention_mask)
        # bert的输出结果有四个维度：
        # last_hidden_state：shape是(batch_size, sequence_length, hidden_size)，hidden_size=768,它是模型最后一层输出的隐藏状态，bert模型对每个token对应的隐藏状态的输出。
        # pooler_output：shape是(batch_size, hidden_size)，这是序列的第一个token(classification token)的最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的。（通常用于句子分类，至于是使用这个表示，还是使用整个输入序列的隐藏状态序列的平均化或池化，视情况而定）  第一个token是[CLS]，这个token是用来做分类的，所以这个输出是用来做分类的
        # hidden_states：这是输出的一个可选项，如果输出，需要指定config.output_hidden_states=True,它也是一个元组，它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)
        # attentions：这也是输出的一个可选项，如果输出，需要指定config.output_attentions=True,它也是一个元组，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值。
        # 我们是微调模式，需要获取bert最后一个隐藏层的输出输入到下一个全连接层，所以取第一个维度，也就是hiden_outputs[0]
        # 此时shape是(batch_size, sequence_length, hidden_size)，[:,0,:]的意思是取出第一个也就是cls对应的结果，至于为什么这样操作，我也不知道，有人告诉我因为它最具有代表性，但为什么我还是不知道，有大神能给我讲一下吗
        outputs = hiden_outputs[0][:,0,:]     #[0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果 [CLS]是句子的第一个token，用于分类任务，通常被认为可以总结整个句子的语义
        # 把bert最后一层的结果输入到全连接层中，全连接层是{768,15},会输出15分类的一个结果， 将768维的向量映射到15维的向量 【全连接层为分类头】
        output = self.fc(outputs)
        # 这里就是返回最终地分类结果了
        return output

# 第三部分，整理数据集
# 可以看一下tsv文件的格式，就知道下面这两行什么意思了，一个存储文本，一个存储改文本对应的标签
sentences = train_set['text_a'].values
targets = train_set['label'].values
train_features,test_features,train_targets,test_targets = train_test_split(sentences,targets)# 这里是把数据分为训练集和测试集，开头导入这个库的方法时说了

batch_size = 64 #把64句话当成一个块来处理，相当于一段有64句话
batch_count = int(len(train_features) / batch_size) #这里就是所有的数据一共可以分为多少块（段）
batch_train_inputs, batch_train_targets = [], []# 一个列表存储分段的文本，一个列表存储分段的标签
# 分段
for i in range(batch_count):
    batch_train_inputs.append(train_features[i*batch_size : (i+1)*batch_size])
    batch_train_targets.append(train_targets[i*batch_size : (i+1)*batch_size])

# 第四部分，训练
bertclassfication = BertClassfication() #实例化
lossfuction = nn.CrossEntropyLoss() #定义损失函数，交叉熵损失函数
optimizer = torch.optim.Adam(bertclassfication.parameters(),lr=2e-5)#torch.optim里面都是一些优化器，就是一些反向传播调整参数的方法，比如梯度下降，随机梯度下降等，这里使用ADAM，一种随机梯度下降的改进优化方法
epoch = 5 # 训练轮数，5轮就是所有数据跑五遍
for _ in range(epoch):
    los = 0  # 损失，写在这里，因为每一轮训练的损失都好应该重新开始计数
    for i in range(batch_count):#刚才说了batch_count的意思有多少块（段），每段有64句话
        inputs = batch_train_inputs[i]
        targets = torch.tensor(batch_train_targets[i])
        optimizer.zero_grad()#1.梯度置零
        outputs= bertclassfication(inputs)#2.模型获得结果
        loss = lossfuction(outputs,targets)#3.计算损失
        loss.backward()#4.反向传播
        optimizer.step()# 5.修改参数，w，b

        los += loss.item() #item()返回loss的值
        # 下面每处理五个段，我们看一下当前损失是多少
        if i%5==0:
            print("Batch:%d,Loss %.4f" % ((i),los/5))
            los = 5


# 第四部分 验证
hit = 0 #用来计数，看看预测对了多少个
total = len(test_features) # 看看一共多少例子
for i in range(total):
    outputs = model([test_features[i]])
    _,predict = torch.max(outputs,1)# 这里你需要了解一下torch.max函数，详见https://www.jianshu.com/p/3ed11362b54f
    if predict==test_targets[i]:# 预测对
        hit+=1
print('准确率为%.4f'%(hit/len(test_features)))


# 小实验，效果呈现
# 模型训练完事之后，由于原数据集的label只有数字，没有中文注释，于是我自己人工注释了0-14这15个数字对应的类别
transform_dict = {0:'文学',1:'娱乐资讯',2:'体育',3:'财经',4:'房产与住宅',5:'汽车',6:'教育',
                  7:'科技与互联网',8:'军事',9:'旅游',10:'国际新闻与时政',11:'股票',12:'三农',
                  13:'电子竞技',14:'小说、故事与轶闻趣事' }
result = bertclassfication(["2022年是政治之年"])
_,result = torch.max(result,1)
print(transform_dict[result])


