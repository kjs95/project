import pandas as pd
import glob
#data 합치는 코드
all_data = pd.DataFrame()

for f in glob.glob('C:/Users/user/Documents/news_crawling/sum/sh/*.xlsx'):
    data = pd.read_excel(f,header=None,skiprows=1)
    print(data)
    col = ['date', 'company', 'title', 'desc', 'url']
    df = pd.DataFrame(data)
    all_data = all_data.append(df, ignore_index=True)

print(all_data.shape)
all_data.head()

all_data.to_excel('C:/Users/user/Documents/news_crawling/sum/sh/all.xlsx', header=False, index= False)
print(all_data)
