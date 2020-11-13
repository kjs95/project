# -*- coding: utf-8 -*-
#데이터 섞는 코드
import pandas as pd



news_data = pd.read_excel("C:/Users/user/PycharmProjects/crawling/shuffle.xlsx",sep='delimiter')

print(news_data.head)
news_data = news_data.sample(frac=1).reset_index(drop=True)
news_data.to_excel('all.xlsx',header=False,index=False)
print(news_data)

