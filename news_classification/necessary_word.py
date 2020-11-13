from konlpy.tag import Okt
import numpy as np
import pandas as pd
import openpyxl
#먼저 data를 구두점, 외국어 한자 및 기타기호, 조사 제거하는 코드 입니다.

news_data = pd.read_excel("./all_necessary_xlsx",sep='delimiter')
np_news_data = np.array(news_data)

okt = Okt()
wb = openpyxl.Workbook()

sheet = wb.active

n_contain = ["Punctuation","Foreign","Josa"]
for i in range(np_news_data.shape[0]):
    title = okt.pos(np_news_data[i][3])
    content = okt.pos(np_news_data[i][4])
    temptitle = []
    tempcontent = []
    for word, tag in title:
        if tag not in n_contain:
            temptitle.append(word)
            temptitle.append(" ")

    for word, tag in content:
        if tag not in n_contain:
            tempcontent.append(word)
            tempcontent.append(" ")

    t = "".join(temptitle)
    c = "".join(tempcontent)
    print(c)
    sheet.append([np_news_data[i][0],np_news_data[i][1],np_news_data[i][2],t,c])


wb.save('all_necessary.xlsx')