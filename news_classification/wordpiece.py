# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sentencepiece as spm
import openpyxl

wb = openpyxl.Workbook()

sheet = wb.active

templates= '--input={} \
--pad_id={} \
--bos_id={} \
--eos_id={} \
--unk_id={} \
--model_prefix={} \
--vocab_size={} \
--character_coverage={} \
--model_type={}'


train_input_file = "./wordtrain.txt"
pad_id=0  #<pad> token을 0으로 설정
vocab_size = 227993 # vocab 사이즈
prefix = 'botchan_spm1' # 저장될 tokenizer 모델에 붙는 이름
bos_id=1 #<start> token을 1으로 설정
eos_id=2 #<end> token을 2으로 설정
unk_id=3 #<unknown> token을 3으로 설정
character_coverage = 1.0 # to reduce character set
model_type ='word' # Choose from unigram (default), bpe, char, or word


cmd = templates.format(train_input_file,pad_id,bos_id,eos_id, unk_id,prefix,vocab_size,character_coverage, model_type)

spm.SentencePieceTrainer.Train(cmd)

sp = spm.SentencePieceProcessor()
sp.Load('botchan_spm1.model') # prefix이름으로 저장된 모델



with open('./botchan_spm1.vocab', encoding='utf-8') as f:
    Vo = [doc.strip().split("\t") for doc in f]


# w[0]: token name
# w[1]: token score

word2idx = {w[0]: i for i, w in enumerate(Vo)}

sp.SetEncodeExtraOptions('bos:eos')

news_data = pd.read_excel("./all_necessary_wordpiece.xlsx",sep='delimiter',header=None,index=False)
np_news_data = np.array(news_data)

for i in range(np_news_data.shape[0]):
    t = ""
    t = "".join(np_news_data[i][3])
    print(t)
    tokens = sp.EncodeAsPieces(t)
    temp = []
    for n in tokens:
        if len(n) >2:

            temp.append(n[1:])
    temp = temp[1:-1]
    sheet.append([np_news_data[i][0], np_news_data[i][1], np_news_data[i][2],  " ".join(map(str,temp)),np_news_data[i][4]])
wb.save('all_nessary_wordpiece2.xlsx')

with open('{}.vocab'.format(prefix), encoding='utf-8') as f:
    vocabs = [doc.strip() for doc in f]

