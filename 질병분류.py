#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np

disease = pd.read_excel('질병사전.xlsx')

disease = disease[disease['질병명'].isin(disease[disease['항목'] == '증상']['질병명'].tolist())]
disease.reset_index(inplace=True, drop=True)
disease['요약'] = disease['요약'].str.replace('요약\n', '')
disease

#머리
disease_head = disease[(disease['신체부위'] == "머리") | (disease['신체부위']== "귀") | (disease['신체부위']== "눈") | (disease['신체부위']== "입") | (disease['신체부위']== "코")  | (disease['신체부위']== "목")]
##머리-나이대
disease_head_t = disease_head[(disease_head['나이대']== "10대-")]
disease_head_c = disease_head[(disease_head['나이대']== "공통")]


#가슴어깨
disease_chest = disease[(disease['신체부위'] == "가슴") | (disease['신체부위']== "어깨")]
##가슴어깨-성별
##가슴어깨-성별-나이대
disease_chest_m = disease_chest[(disease_chest['성별']== "공통")]
disease_chest_m_t = disease_chest_m[(disease_chest_m['나이대']== "10대-")]
disease_chest_m_c = disease_chest_m[(disease_chest_m['나이대']== "공통")]

disease_chest_w = disease_chest[(disease_chest['성별']== "여성") | (disease_chest['성별']== "공통")]
disease_chest_w_t = disease_chest_w[(disease_chest_w['나이대']== "10대-")]
disease_chest_w_c = disease_chest_w[(disease_chest_w['나이대']== "공통")]


#팔다리
disease_arms  = disease[(disease['신체부위'] == "팔") | (disease['신체부위']== "다리")]


#배
disease_stomach = disease[(disease['신체부위'] == "배")]


#허리등엉덩이
disease_waist = disease[(disease['신체부위'] == "허리") | (disease['신체부위']== "등") | (disease['신체부위']== "엉덩이")]
##허리등엉덩이-나이대
disease_waist_t = disease_waist[(disease_waist['나이대']== "10대-")]
disease_waist_c = disease_waist[(disease_waist['나이대']== "공통")]


#피부
disease_skin = disease[(disease['신체부위'] == "피부")]
##피부-나이대
disease_skin_t = disease_skin[(disease_skin['나이대']== "10대-")]
disease_skin_c = disease_skin[(disease_skin['나이대']== "공통")]

#손발
disease_hand = disease[(disease['신체부위'] == "손") | (disease['신체부위'] == "발")]


#생식기
disease_organs = disease[(disease['신체부위'] == "생식기")]
##생식기-성별
disease_organs_m = disease_organs[(disease_organs['성별']== "남성") | (disease_organs['성별']== "공통")]
##생식기-성별-나이대
disease_organs_m_t = disease_organs_m[(disease_organs_m['나이대']== "10대-")]
disease_organs_m_c = disease_organs_m[(disease_organs_m['나이대']== "공통")]

disease_organs_w = disease_organs[(disease_organs['성별']== "여성") | (disease_organs['성별']== "공통")]
disease_organs_w_t = disease_organs_w[(disease_organs_w['나이대']== "10대-")]
disease_organs_w_c = disease_organs_w[(disease_organs_w['나이대']== "공통")]


#기타
disease_etc = disease[(disease['신체부위'] == "기타")]
#기타-나이대
disease_etc_t = disease_etc[(disease_etc['나이대']== "10대-")]
disease_etc_t = disease_etc[(disease_etc['나이대']== "공통")]


#전체
disease
#전체-성별
disease_m = disease[(disease['성별']== "남성") | (disease['성별']== "공통")]
#전체-성별-나이대
disease_m_t = disease_m[(disease_m['나이대']== "10대-")]
disease_m_c = disease_m[(disease_m['나이대']== "공통")]

disease_w = disease[(disease['성별']== "여성") | (disease['성별']== "공통")]
disease_m_t = disease_w[(disease_w['나이대']== "10대-")]
disease_m_c = disease_w[(disease_w['나이대']== "공통")]


# In[5]:


# 이는 구글 코랩으로 돌린 버전입니다. 그리고 기본 코드는 SKT Brain의 KoBERT를 그대로 가져왔고, 학습 및 테스트 데이터셋만 따로 준비한 것입니다.
# SKT Brain github 주소는 다음과 같습니다. https://github.com/SKTBrain/KoBERT



import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

##GPU 사용 시
#device = torch.device("cuda:0")
device = torch.device("cpu")

bertmodel, vocab = get_pytorch_kobert_model()


# 학습용 데이터셋 불러오기
import pandas as pd
disease = pd.read_excel('질병사전.xlsx')

disease = disease[disease['질병명'].isin(disease[disease['항목'] == '증상']['질병명'].tolist())]
disease.reset_index(inplace=True, drop=True)
disease['요약'] = disease['요약'].str.replace('요약\n', '')
disease_1 = disease

disease = disease[['항목내용', '질병명']]
new_data = disease

# 질병명 라벨링
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(new_data['질병명'])
new_data['질병명'] = encoder.transform(new_data['질병명'])
new_data.head()

mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))
print(mapping)

# Train / Test set 분리
from sklearn.model_selection import train_test_split
train, test = train_test_split(new_data, test_size=0.2, random_state=42)
print("train shape is:", len(train))
print("test shape is:", len(test))

train = new_data

# 기본 Bert tokenizer 사용
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTDataset(Dataset):
  def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
               pad, pair):
      transform = nlp.data.BERTSentenceTransform(
          bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair) 

      self.sentences = [transform([i]) for i in dataset.iloc[:, 0]]
      self.labels = [np.int32(i) for i in dataset.iloc[:, 1]]

  def __getitem__(self, i):
      return (self.sentences[i] + (self.labels[i], ))

  def __len__(self):
      return (len(self.labels))
      
# Setting parameters
max_len = 64 # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
batch_size = 64
warmup_ratio = 0.1
num_epochs = 100
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

data_train = BERTDataset(train, 0, 1, tok, max_len, True, False)
print(data_train)
data_test = BERTDataset(test, 0, 1, tok, max_len, True, False)

# pytorch용 DataLoader 사용
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)

class BERTClassifier(nn.Module):
  def __init__(self,
               bert,
               hidden_size = 768,
               num_classes = 1247, # softmax 사용 <- binary일 경우는 2
               dr_rate=None,
               params=None):
      super(BERTClassifier, self).__init__()
      self.bert = bert
      self.dr_rate = dr_rate
               
      self.classifier = nn.Linear(hidden_size , num_classes)
      if dr_rate:
          self.dropout = nn.Dropout(p=dr_rate)
  
  def gen_attention_mask(self, token_ids, valid_length):
      attention_mask = torch.zeros_like(token_ids)
      for i, v in enumerate(valid_length):
          attention_mask[i][:v] = 1
      return attention_mask.float()

  def forward(self, token_ids, valid_length, segment_ids):
      attention_mask = self.gen_attention_mask(token_ids, valid_length)
      
      _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
      if self.dr_rate:
          out = self.dropout(pooler)
      return self.classifier(out)
    
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
  {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
  {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# 옵티마이저 선언
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() # softmax용 Loss Function 정하기 <- binary classification도 해당 loss function 사용 가능

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

# 학습 평가 지표인 accuracy 계산 -> 얼마나 타겟값을 많이 맞추었는가
def calc_accuracy(X,Y):
  max_vals, max_indices = torch.max(X, 1)
  train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
  return train_acc

# 모델 학습 시작
for e in range(num_epochs):
  train_acc = 0.0
  test_acc = 0.0
  model.train()
  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
      optimizer.zero_grad()
      token_ids = token_ids.long().to(device)
      segment_ids = segment_ids.long().to(device)
      valid_length= valid_length
      label = label.long().to(device)
      out = model(token_ids, valid_length, segment_ids)
      
      loss = loss_fn(out, label)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # gradient clipping
      optimizer.step()
      scheduler.step()  # Update learning rate schedule
      train_acc += calc_accuracy(out, label)
      if batch_id % log_interval == 0:
          print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
  print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
  

  
  
def d_kind_result(content):
  test_sentence = content
  test_label = 1 # 실제 질병

  unseen_test = pd.DataFrame([[test_sentence, test_label]], columns = [['항목내용', '질병명']])
  unseen_values = unseen_test.values
  test_set = BERTDataset(unseen_test, 0, 1, tok, max_len, True, False)
  test_input = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=0)

  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_input)):
      token_ids = token_ids.long().to(device)
      segment_ids = segment_ids.long().to(device)
      valid_length= valid_length
      out = model(token_ids, valid_length, segment_ids)
      print(out)

  ## 가장 높은 값의 질병만
  print(mapping[(int(torch.argmax(out)))])

  ##높은 값부터 순번
  result_value = []
  mapping_cnt = 0

  """
  for x in torch.argsort(out, descending=True).tolist()[0][:1000]:
      result_value[x] = [mapping[mapping_cnt], out.tolist()[0][mapping_cnt]]
      mapping_cnt = mapping_cnt + 1
  result_value
  """
  for x in torch.argsort(out, descending=True).tolist()[0][:1000]:
      result_value.append([mapping[mapping_cnt], out.tolist()[0][mapping_cnt]])
      mapping_cnt = mapping_cnt + 1

  sorted_result = sorted(result_value, key = lambda result_value: result_value[1], reverse = True)
  
  return sorted_result[:10]


# In[8]:




