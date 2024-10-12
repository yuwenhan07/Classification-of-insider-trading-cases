import torch
import pandas as pd
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split

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
                                max_length=148, pad_to_max_length=True)
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        hiden_outputs = self.model(input_ids, attention_mask=attention_mask)
        outputs = hiden_outputs[0][:, 0, :]
        output = self.fc(outputs)
        return output

sentences = train_set['text_a'].values
targets = train_set['label'].values
train_features, test_features, train_targets, test_targets = train_test_split(sentences, targets)

batch_size = 64
batch_count = int(len(train_features) / batch_size)
batch_train_inputs, batch_train_targets = [], []
for i in range(batch_count):
    batch_train_inputs.append(train_features[i * batch_size: (i + 1) * batch_size])
    batch_train_targets.append(train_targets[i * batch_size: (i + 1) * batch_size])

bertclassfication = BertClassfication()
lossfuction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(bertclassfication.parameters(), lr=2e-5)

epoch = 5
for _ in range(epoch):
    los = 0
    for i in range(batch_count):
        inputs = batch_train_inputs[i]
        targets = torch.tensor(batch_train_targets[i])
        optimizer.zero_grad()
        outputs = bertclassfication(inputs)
        loss = lossfuction(outputs, targets)
        loss.backward()
        optimizer.step()
        los += loss.item()
        if i % 5 == 0:
            print("Batch:%d,Loss %.4f" % (i, los / 5))
            los = 5

hit = 0
total = len(test_features)
for i in range(total):
    outputs = model([test_features[i]])
    _, predict = torch.max(outputs, 1)
    if predict == test_targets[i]:
        hit += 1
print('准确率为%.4f' % (hit / len(test_features)))

transform_dict = {0: '文学', 1: '娱乐资讯', 2: '体育', 3: '财经', 4: '房产与住宅', 5: '汽车', 6: '教育',
                  7: '科技与互联网', 8: '军事', 9: '旅游', 10: '国际新闻与时政', 11: '股票', 12: '三农',
                  13: '电子竞技', 14: '小说、故事与轶闻趣事'}
result = bertclassfication(["2022年是政治之年"])
_, result = torch.max(result, 1)
print(transform_dict[result])