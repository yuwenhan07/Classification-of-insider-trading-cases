import torch
import pandas as pd
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 导入tqdm库，用于进度条

# 检查是否有可用的GPU，并打印相关信息
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_set = pd.read_csv("./data/train.tsv", sep="\t", on_bad_lines='skip')
model_name = "hfl/chinese-bert-wwm"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

class BertClassfication(nn.Module):
    def __init__(self):
        super(BertClassfication, self).__init__()
        self.model_name = 'hfl/chinese-bert-wwm'
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(768, 15)

    def forward(self, x):
        batch_tokenized = self.tokenizer.batch_encode_plus(x, add_special_tokens=True,
                                max_length=148, truncation=True, padding='max_length')  # 修改为 truncation=True 和 padding='max_length'
        input_ids = torch.tensor(batch_tokenized['input_ids']).to(device)
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(device)
        hiden_outputs = self.model(input_ids, attention_mask=attention_mask)
        outputs = hiden_outputs[0][:, 0, :]
        output = self.fc(outputs)
        return output

# 整理数据集
sentences = train_set['text_a'].values
targets = train_set['label'].values
train_features, test_features, train_targets, test_targets = train_test_split(sentences, targets)

batch_size = 64
batch_count = int(len(train_features) / batch_size)
batch_train_inputs, batch_train_targets = [], []
for i in range(batch_count):
    batch_train_inputs.append(train_features[i * batch_size: (i + 1) * batch_size])
    batch_train_targets.append(train_targets[i * batch_size: (i + 1) * batch_size])

# 实例化模型并迁移到GPU
bertclassfication = BertClassfication().to(device)

# 定义损失函数和优化器
lossfuction = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(bertclassfication.parameters(), lr=2e-5)

# 训练轮数
epoch = 5
for ep in range(epoch):
    los = 0
    print(f"Epoch {ep+1}/{epoch}")
    # 使用tqdm进度条，设置总数为batch_count
    for i in tqdm(range(batch_count), desc="Training Progress", unit="batch"):
        inputs = batch_train_inputs[i]
        targets = torch.tensor(batch_train_targets[i]).to(device)
        optimizer.zero_grad()  # 梯度置零
        outputs = bertclassfication(inputs)  # 模型获得结果

        loss = lossfuction(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 修改参数

        los += loss.item()
        if i % 5 == 0:
            print(f"Batch {i}, Loss: {los/5:.4f}")
            los = 0

# 验证
hit = 0
total = len(test_features)
print("Running validation...")
for i in tqdm(range(total), desc="Validation Progress", unit="sample"):
    outputs = bertclassfication([test_features[i]])
    _, predict = torch.max(outputs, 1)
    if predict == test_targets[i]:
        hit += 1
print('准确率为%.4f' % (hit / len(test_features)))

# 小实验，效果呈现
transform_dict = {0: '文学', 1: '娱乐资讯', 2: '体育', 3: '财经', 4: '房产与住宅', 5: '汽车', 6: '教育',
                  7: '科技与互联网', 8: '军事', 9: '旅游', 10: '国际新闻与时政', 11: '股票', 12: '三农',
                  13: '电子竞技', 14: '小说、故事与轶闻趣事'}
result = bertclassfication(["2022年是政治之年"])
_, result = torch.max(result, 1)
print(transform_dict[result])