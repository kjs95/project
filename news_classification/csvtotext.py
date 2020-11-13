# -*- coding: utf-8 -*-
import pandas as pd
#엑셀을 text로 변환(wordpiece할때 사용)
data = pd.read_excel("./content.xlsx",sep='delimiter')
data.to_csv("./content.txt",index=False,header=None,sep="\t")