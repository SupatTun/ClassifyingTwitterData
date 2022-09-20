import pandas as pd
from Tweet_extaction import gettweet
from pythainlp import word_tokenize,corpus
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model,model_selection,metrics,naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from gensim.models.keyedvectors import KeyedVectors
import string
import joblib
from gensim.models import Word2Vec
from tensorflow.keras.optimizers import Adam

#
gettweet('#รัฐประหาร',1000)

# Word Preprocessing Functions
def flatten(t):
    return [item for sublist in t for item in sublist]

def textsplit(text):
    list_of_word = []
    text = text.lower().split()
    for w in text:
        list_of_word.append(word_tokenize(w))
    list_of_word = flatten(list_of_word)
    return list_of_word
 
def sentence_words(x):
    sentence_words = []
    for i in range(len(x)):
        sentence_words.append(textsplit(x["Text"][i].lower()))

def sentences_to_vector(X, word_to_vector, max_len,stopword):
    m = X.shape[0] 
    X_indices = np.zeros([m,max_len,300])
    
    for i in range(m):
        sentence_words = textsplit(X[i].lower())
        sentence_words = [a for a in sentence_words if not a in stopword]
        j = 0
        for w in sentence_words:
            if w.isdigit() or w in string.punctuation or w in string.ascii_letters:
                pass
            else:
                if w in word_to_index:
                    X_indices[i,j] = word_to_vector[w]
                    j += 1
                else: X_indices[i,j] = np.zeros(300)
    return X_indices

#Import Data for model. 
Raw_data = pd.read_excel('Raw_data.xlsx')
Raw_data  = Raw_data.sample(frac=1).reset_index(drop=True)
Train, Test = model_selection.train_test_split(Raw_data, test_size=0.2)
Train.reset_index(drop=True,inplace=True)
Test.reset_index(drop=True,inplace=True)

#Plot Bar Chart to show Train and Test data set distribution.
Train_plot = Train['Flag'].value_counts().plot.bar()
Freq = []
Freq.append(round(Train['Flag'][Train.Flag==0].count()*100/len(Train['Flag'])))
Freq.append(round(Train['Flag'][Train.Flag==1].count()*100/len(Train['Flag'])))
Train_plot.bar_label(Train_plot.containers[0], label_type='edge')
plt.title('Train Data Ditribution')
plt.show()

Test_plot = Test['Flag'].value_counts().plot.bar()
Test_plot.bar_label(Test_plot.containers[0], label_type='edge')
plt.title('Test Data Ditribution')
plt.show()

#Query only Target data (Flag = 1) for wordCloud
Train_target = Raw_data[Raw_data['Flag']==1].reset_index(drop=True)

#Train_TargetSet (Flag = 1)
text_target = []
for i in range(len(Train_target["Text"])):
    text_target.append(word_tokenize(Train_target["Text"][i]))
text_target = np.concatenate(text_target).flatten()

#Train_set
text = []
for i in range(len(Test["Text"])):
    text.append(word_tokenize(Test["Text"][i]))
text = np.concatenate(text).flatten()

# Import Thai stopword
stopword = ' '.join(corpus.thai_stopwords())
stopword += (' มหาชน เม ผม รัฐประหาร โหน ตกหนัก กรุงเทพ กทม ไทย สมเด็จ ฝนตก เอลิซาเบธ ควีน ประชุม ยกเลิก สภา ฝน น้ํา ชัชชาติ น้ำ นํ้า นำ้ บิ๊ก ผลิต น้ำท่วม ท่วม')
stopword_list = stopword.split(' ')

#Making Wordcloud of Tweets
text_tg = [a for a in text_target if not a in stopword_list]
a = ' '.join(text_tg)
cloude = np.array(Image.open( "tw.jpg"))
wordcloud = WordCloud(font_path='C:/Windows/Fonts/Angsana.ttc',regexp=r"[\u0E00-\u0E7Fa-zA-Z']+",stopwords= stopword,max_words=50
,collocations=False,background_color="white",mask = cloude).generate(a)
plt.figure(figsize=(40,30))
plt.imshow(wordcloud, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()

#ฺ Crate Bag of Words
#vect = CountVectorizer(analyzer= lambda x:x.split(' '))
vect = CountVectorizer(stop_words=stopword_list)
vect.fit(text)
#vocabulary = vect.vocabulary_
#sorted(vocabulary.items(), key=lambda x:x[1],reverse= True)

#Crate Logistic Model
X_train = pd.DataFrame(vect.transform(Train['Text']).toarray(),columns=vect.get_feature_names_out(),index=Train['Text'])
lr_model = LogisticRegression()
lr_model.fit(X=X_train.values,y=Train['Flag'])

#Logistic Model Valuation
test_vec = vect.transform(Test['Text'])
test_pre = lr_model.predict(test_vec)
print("Logistic Model")
print(classification_report(Test['Flag'],test_pre))

#Naive Bayse model
nb_model = naive_bayes.MultinomialNB()
nb_model.fit(X=X_train.values,y=Train['Flag'])

#Naive Bayse model Valuation
nb_test_vec = vect.transform(Test['Text'])
nb_test_pre = nb_model.predict(nb_test_vec)
print("Naive Bayse Model")
print(classification_report(Test['Flag'],nb_test_pre))

#Use TF-IDF Word Tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
tfid_vec = TfidfVectorizer(tokenizer=word_tokenize,token_pattern=None,stop_words=stopword_list)
tfid_vec.fit(text)
tfid_vec.vocabulary_

#Save vocab
#joblib.dump(tfid_vec, "tfid_vec.pkl")

# loading pickled vectorizer
#vectorizer = joblib.load("tfid_vec.pkl")

#Crate TF-IDF Logistic Model
X_train_tf = pd.DataFrame(tfid_vec.transform(Train['Text']).toarray(),columns=tfid_vec.get_feature_names_out(),index=Train['Text'])
TF_lr_model = LogisticRegression()
b = X_train_tf.values
TF_lr_model.fit(X=X_train_tf.values,y=Train['Flag'])
test_tfid_vec = tfid_vec.transform(Test['Text'])
test_pre = TF_lr_model.predict(test_tfid_vec)
print(("TF-IDF Logistic Model"))
print(classification_report(Test['Flag'],test_pre))

#TF-IDF Naive Bayse model
TF_nb_model = naive_bayes.MultinomialNB()
TF_nb_model.fit(X=X_train_tf.values,y=Train['Flag'])

#TF-IDF Naive Bayse model Valuation
Tf_test_pre = TF_nb_model.predict(test_tfid_vec)
print(("TF-IDF Naive Bayse Model"))
print(classification_report(Test['Flag'],Tf_test_pre))
confusion_matrix(Test['Flag'], Tf_test_pre, normalize='pred')

#Save Model
#joblib.dump(TF_nb_model, "TF_nb_model.pkl") 
#clf2 = joblib.load("TF_nb_model.pkl")
    

#Import Fastvec
fastvec = KeyedVectors.load_word2vec_format("cc.th.300.vec")
word_to_index = fastvec.key_to_index
index_to_word = fastvec.index_to_key

#Tranform Words to vectors
X_train_for_model = sentences_to_vector(Train['Text'],fastvec,80,stopword_list)
Y_train = np.array(pd.get_dummies(Train['Flag']))
X_test_for_model = sentences_to_vector(Test['Text'],fastvec,80,stopword_list)
Y_vaild = np.array(pd.get_dummies(Test['Flag']))

#LSTM Model
LSTM_model = Sequential()
LSTM_model.add(LSTM(64, return_sequences=True))
LSTM_model.add(Dropout(0.2))
LSTM_model.add(LSTM(64, return_sequences=False))
LSTM_model.add(Dropout(0.2))
dense = Dense(2,Activation('softmax'))
LSTM_model.add(dense)
LSTM_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])

LSTM_model.fit(X_train_for_model,Y_train, epochs = 70,validation_data=(X_test_for_model,Y_vaild))
#batch_size = 16,
loss = LSTM_model.history
plt.plot(loss.history['accuracy']) 
plt.plot(loss.history['val_accuracy']) 
plt.title('Model Loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()
     
LSTM = LSTM_model.predict(X_test_for_model)
print(classification_report(Y_vaild.argmax(1),LSTM.argmax(1)))

#Check Raw Date with model
ew = tfid_vec.transform(Raw_data['Text'])
nb_test_ew = TF_nb_model.predict(ew)
print(classification_report(Raw_data['Flag'],nb_test_ew))
Raw_data['Predict']=nb_test_ew
Raw_data.to_excel("Check.xlsx") 