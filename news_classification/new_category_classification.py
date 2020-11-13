# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn import metrics
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from konlpy.tag import Okt
import pydot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import LabelBinarizer

# classification code

news_data = pd.read_excel("./all_necessary_wordpiece.xlsx",sep='delimiter')



np_news_data = np.array(news_data)
#print(np_news_data)
#print(np_news_data.shape)
okt = Okt()
#company = []
dict = []



category = {'정치':1, '경제':2,'사회':3,'생활문화':4,'세계':5,'과학it':6}

datax = np_news_data[:,3]#3이 제목, 4가 본문
datay = np_news_data[:,0]

trnx , tstx, trny, tsty = train_test_split(datax,datay,test_size=0.3)
#print(trnx)
trans_trny = []

for i in range(len(trnx)):
    trans_trny.append(category[(trny[i])])

max_word = 10000 #본문
max_len = 50 #본문
# max_word = 5000 #제목
# max_len = 10 #제목


tok = Tokenizer(num_words=max_len)
tok.fit_on_texts(trnx)
seq = tok.texts_to_sequences(trnx)
#print(len(seq[0]))
#print(seq[0])

xtrn_seq_matrix = pad_sequences(seq,maxlen = max_len)
#print(xtrn_seq_matrix)
#print(xtrn_seq_matrix[0])
#print(len(xtrn_seq_matrix[0]))

#trans_tstx = []
trans_tsty = []


for i in range(len(tstx)):
    trans_tsty.append(category[(tsty[i])])



tok = Tokenizer(num_words=max_len)
tok.fit_on_texts(tstx)
seq = tok.texts_to_sequences(tstx)
xtst_seq_matrix = pad_sequences(seq,maxlen = max_len)

input_shape = (max_len,)

mlp_model = models.Sequential()
mlp_model.add(layers.Dense(units=10, activation='relu', input_shape=input_shape))
mlp_model.add(layers.Dense(units=20, activation='relu'))
mlp_model.add(layers.Dense(units=30, activation='relu'))
mlp_model.add(layers.Dense(units=40, activation='relu'))
mlp_model.add(layers.Dropout(0.2))
mlp_model.add(layers.Dense(units=30, activation='relu'))
mlp_model.add(layers.Dense(units=20, activation='relu'))
mlp_model.add(layers.Dropout(0.2))
mlp_model.add(layers.Dense(units=10, activation='relu'))
mlp_model.add(layers.Dense(units=6, activation='softmax'))

mlp_model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
mlp_model.summary()

encoder = LabelBinarizer()
trny_onehot = encoder.fit_transform(trans_trny)
tsty_onehot = encoder.transform(trans_tsty)
#print('trny one hot : ',trny_onehot)

#print(trny_onehot[0:5,:])
#print(tsty_onehot[0:5,:])

#print(type(xtrn_seq_matrix), type(trny_onehot))

#본문
#history = mlp_model.fit(xtrn_seq_matrix, trny_onehot, validation_data=[xtst_seq_matrix, tsty_onehot], batch_size=64, epochs=200)

#제목
history = mlp_model.fit(xtrn_seq_matrix, trny_onehot, validation_data=[xtst_seq_matrix, tsty_onehot], batch_size=64, epochs=100)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()



scaler = MinMaxScaler()
scaler.fit(xtrn_seq_matrix)
trnx_scale = scaler.transform(xtrn_seq_matrix)
tstx_scale = scaler.transform(xtst_seq_matrix)
#print(np.min(trnx_scale[:,0]), np.max(trnx_scale[:,0]))
#print(np.min(tstx_scale[:,0]), np.max(tstx_scale[:,0]))

k=6
knn_model = neighbors.KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X=xtrn_seq_matrix, y=trny)

knn_pred_trn = knn_model.predict(X=xtrn_seq_matrix)
knn_pred_tst = knn_model.predict(X=xtst_seq_matrix)

#print(knn_pred_trn)
#print(knn_pred_tst)
print('y 비교 : ',trny[:10], knn_pred_tst[:10])
#print(metrics.accuracy_score(trny,knn_pred_trn))
print('k-NN 정확도 ',metrics.accuracy_score(tsty,knn_pred_tst))


#Ensemble
rf_model = RandomForestClassifier(max_depth=15, n_estimators=100, random_state=0)
rf_model.fit(X=xtrn_seq_matrix, y= trny)

rf_pred = rf_model.predict(X=xtst_seq_matrix)
print('y 비교 : ',trny[:10], rf_pred[:10])
print('rf accuracy(random forest) : ',metrics.accuracy_score(tsty,rf_pred))

gbm_model = GradientBoostingClassifier(max_depth=3, n_estimators=30, random_state=0)
gbm_model.fit(X=xtrn_seq_matrix, y=trny)

gbm_pred = gbm_model.predict(X=xtst_seq_matrix)
print('y 비교 : ',trny[:10], gbm_pred[:10])
print('gbm accuracy(Gradient Boosting Machine): ',metrics.accuracy_score(tsty,gbm_pred))



#Decision tree
tree_model = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=3)
tree_model.fit(X=xtrn_seq_matrix, y=trny)

tree_pred = tree_model.predict(X=xtst_seq_matrix)
#print(tree_model.feature_importances_)
export_graphviz(tree_model, out_file='decision_tree.dot')
(graph,) = pydot.graph_from_dot_file('decision_tree.dot',encoding='utf8')
graph.write_png('decision_tree.png')




