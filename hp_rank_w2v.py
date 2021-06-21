#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## http://static.wooridle.net/lectures/chatbot/
#https://teamlab.github.io/jekyllDecent/blog/tutorials/%EC%B9%B4%EC%B9%B4%EC%98%A4%ED%86%A1_%EC%B1%97%EB%B4%87_%EB%A7%8C%EB%93%A4%EA%B8%B0_with_python_flask_aws


import pandas as pd
import numpy as np
import operator
from IPython import get_ipython

naver_reviews_list = pd.read_excel('naver_reviews_list.xlsx')
blog_list = pd.read_excel('blog_list_df.xlsx')
google_review = pd.read_excel('google_review.xlsx')
insta_review = pd.read_excel('insta_info_df.xlsx')

blog_list_1 = blog_list[['nm', 'blog_title']]
blog_list_1.columns = ['nm', 'review_contents']

blog_list_2 = blog_list[['nm', 'blog_contents']]
blog_list_2.columns = ['nm', 'review_contents']

blog_list_re = pd.concat([blog_list_1, blog_list_2], ignore_index=True)

google_list_re = google_review[['nm', 'google_contents']]
google_list_re.columns = ['nm', 'review_contents']

insta_review_re = insta_review[['location_info', 'main_text']]
insta_review_re.columns = ['nm', 'review_contents']

train_df = pd.concat([blog_list_re, insta_review_re, naver_reviews_list[['nm', 'review_contents']], google_list_re], ignore_index=True)

train_df['review_contents'] = train_df['review_contents'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_df['review_contents'] = train_df['review_contents'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_df['review_contents'].replace('', np.nan, inplace=True)
train_df = train_df.dropna(axis=0)

from konlpy.tag import Okt
#객체 생성
okt = Okt()

train_df.isnull().any() #document에 null값이 있다.
train_df['review_contents'] = train_df['review_contents'].fillna(''); #null값을 ''값으로 대체

def tokenize(doc):
    #형태소와 품사를 join
    
    ##불용어
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를', '을', '으로','자','에','와','한','하다', '대', '전', '님', '에서', '때', '로', '고']
    
    ##품사(원하는 품사만 가져오기)
    tagset = ['Adjective', 'Noun', 'Exclamation'] 
    #okt.pos(doc, norm=True, stem=True)
    
    ##토근
    temp_X = okt.pos(doc, norm=True, stem=True)
    #return ['/'.join(word) for word in temp_X if not word[0] in stopwords if word[1] in tagset] # 불용어 제거
    return [word[0] for word in temp_X if not word[0] in stopwords if word[1] in tagset] # 불용어 제거 #원하는 품사만 

#tokenize 과정은 시간이 오래 걸릴수 있음...
train_docs = [(tokenize(row[1])) for row in  train_df.values]

import nltk
#tokens = [t for d in train_docs for t in d[0]]
tokens = [t for d in train_docs for t in d]

text = nltk.Text(tokens, name='NMSC')

#토큰개수
print(len(text.tokens))

#중복을 제외한 토큰개수
print(len(set(text.tokens)))

#출력빈도가 높은 상위 토큰 10개
print(text.vocab().most_common(50))

"""
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
plt.rcParams['font.family'] = 'Malgun Gothic'

plt.figure(figsize=(20,10))
text.plot(50)
"""
from gensim.models import Word2Vec
model = Word2Vec(sentences=train_docs, vector_size =100, window=5, min_count=5, workers=4, sg=0)

from gensim.models import KeyedVectors
model.wv.save_word2vec_format('word2vec') # 모델 저장

# annotation text 만들기 (시각화할 때 벡터 말고 단어도 필요하니까)
# vocabs = word_vectors.vocab.keys()
text=[]
word_vectors = model.wv
vocabs = model.wv.index_to_key 
word_vectors_list = [word_vectors[v] for v in vocabs]

"""
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
xys = pca.fit_transform(word_vectors_list)
xs = xys[:,0]
ys=xys[:,1]

import matplotlib.font_manager as fm
fm._rebuild()

plt.rc('font', family='NanumGothic')

import matplotlib.pyplot as plt

def plot_2d_graph(vocabs, xs, ys):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.figure(figsize=(15,10))
    plt.scatter(xs,ys,marker='o')
    for i,v in enumerate(vocabs):
        plt.annotate(v,xy=(xs[i], ys[i]))
        
plot_2d_graph(vocabs, xs,ys)

for i,v in enumerate(vocabs):
    text.append(v)
    
import plotly
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=xs,
                                y=ys,
                                mode='markers+text',
                                text=text)) 

fig.update_layout(title='Naver Word2Vec')
fig.show()

plotly.offline.plot(
fig, filename='naver_word2vec.html'
)
"""

##### 임베딩을 위한 데이터 저장하기
##https://soohee410.github.io/embedding_projector
##https://projector.tensorflow.org/
##해당사이트에서 tensor와 meta파일을 읽어오기


from gensim.models import KeyedVectors  
model.wv.save_word2vec_format('naver_w2v')

get_ipython().system('python -m gensim.scripts.word2vec2tensor --input naver_w2v --output naver_w2v')




#### 모든병원별 형태소 text list 만들기
hp_all = pd.read_excel('results.xlsx')

for hp_nm in hp_all['name']:
    globals()[hp_nm] = nltk.Text([t for d in [(tokenize(row[1])) for row in  train_df[train_df['nm'] == hp_nm].values] for t in d], name='NMSC').vocab().most_common(1000)
    #print(hp_nm)
    
def hp_rank_output(message):
    try:
        ##https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.most_similar.html
        # 유사도가 높은 단어 추출

        ##검색어 입력받기(끊어서 명사 혹은 동사 등으로 검색해야 잘검색됨)
        #input_value = input()
        input_value = message

        ##tokenize 함수로 명사와 동사 그리고 감탄사만 가져와서 검색하기
        search_text = tokenize(input_value)
        #print(search_text)

        # 유사도가 높은 단어 조회
        key_table = pd.DataFrame(model.wv.most_similar(search_text, topn=10), columns=['key', 'value'])
        key_table = key_table.set_index(key_table['key'])

        ### 직접 언급한 키워드의 유사도는 3로 고정함(기본 유사도가 99%이기 때문에 이보다 3배 더 고려하기 위함)
        for x in search_text:
            key_table.loc[x] = [x, 3]

        search_list = []

        ## 유사어
        for x in model.wv.most_similar(search_text, topn=10):
            search_list.append(x[0])

        ## 검색어
        for x in search_text:
            search_list.append(x)

        #for hp_nm in hp_all['name']:
        #    print(hp_nm, [word for word in globals()[hp_nm] if word[0] in search_list] )    

        ## 평점계산
        socre_table = {}

        for hp_nm in hp_all['name']:
            token_cnt_list = []
            for token in globals()[hp_nm]:
                token_cnt_list.append(token[1])

                ###검색어가 무조건 다 있는 상태에서 순위를 알려줘야 함
                #ex) 친절한 정형외과 -> 친절하다, 정형외과
                #    두가지 형태소가 모두 존재하는 병원에서 평점을 계산하가 
                #    두가지 중 하나라도 없으면 점수를 대폭 낮추거나 0점 처리하기
                #    두가가 중 하나라도 있으면 (해당 동일갯수 비율 * 0.1) ,  모두 없으면 0점처리

            #print('검색어 형태소 갯수 : ' + str(len(search_text)))
            match_cnt = 0
            for x in [word for word in globals()[hp_nm] if word[0] in search_text]:
                match_cnt += 1

            #print('찾은 형태소 갯수 : ' + str(match_cnt))

            match_wgt = match_cnt / len(search_text)
            if match_wgt != 1:
                match_wgt = match_wgt * 0.1
            #print(match_wgt)

            value_list = []
            for x in [word for word in globals()[hp_nm] if word[0] in search_list]:
                value_list.append((key_table.loc[x[0], 'value'].round(2)) * (x[1]))

                                                                                ###형태소 비율로 해서 가중치 고려
                #value_list.append((key_table.loc[x[0], 'value'].round(2)) * ((x[1]/sum(token_cnt_list)*100)))

            try:
                socre_table[hp_nm] = ((sum(value_list) * (((naver_reviews_score.loc[hp_nm, 'review_score']+ 5) )/ 10)).round(2)) * match_wgt


                                                            ##형태소개수 고려해서 가감
                #socre_table[hp_nm] = (sum(value_list)  *  (sum(token_cnt_list)/ 50000)  * (((naver_reviews_score.loc[hp_nm, 'review_score']+ 5) )/ 10)).round(2)
            except:
                ###review정보에 해당 병원정보가 없어서 평점산출이 안되면 0.5로 고정함..
                try:
                    socre_table[hp_nm] = ((sum(value_list) * (0.5)).round(2)) * match_wgt
                except:
                    socre_table[hp_nm] = ((sum(value_list) * (0.5))) * match_wgt


            socre_table_sort = sorted(socre_table.items(), key=operator.itemgetter(1), reverse=True)


        #print('검색어: ' +  input_value)
        #print('연관검색어: ', search_list)
        sort_count = 1
        rank_list = ""
        for x in socre_table_sort:
            token_cnt_list = []
            for token in globals()[x[0]]:
                token_cnt_list.append(token[1])
            url_text= hp_all[hp_all['name'] == x[0]]['주소'].values[0]
            name_dept= hp_all[hp_all['name'] == x[0]]['name_dept'].values[0]
            
            if sort_count < 6:
                rank_list = rank_list + (str(sort_count) + "순위:\n-" +  x[0] + "\n-추천점수: " + str(x[1].round(2)) + "\n-" +  name_dept + "\n-검색어갯수:" + str(sum(token_cnt_list)) + "\n" +  url_text + "\n") 
            sort_count += 1
    except:
        search_list = '관련 검색어가 없습니다'
        rank_list = '재검색해주시기 바랍니다.'
        
    return "연관검색어\n '{}'".format(search_list) + "\n\n검색결과: \n'{}'".format(rank_list)

