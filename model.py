import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle   

data = pd.read_csv('data.csv',',', error_bad_lines=False)
# data.head()
print('\n')

data.isnull().sum()

# Dropping the NULL Values
data[data['password'].isnull()]

# Dropping the null values
data.dropna(inplace=True)

# Creating a tuple of the dataset
password_tuple = np.array(data)
print('pass tuple done')

## Shufflin the Dataset As all the strengths are in ascending order
import random
random.shuffle(password_tuple)

#  x --> Input
#  Y --> Output

x = [i[0] for i in password_tuple]
y =  [i[1] for i in password_tuple]

# Dividing the words into individual characters
def word_divide_character(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(tokenizer=word_divide_character)
X = vectorizer.fit_transform(x) # x == input
pickle.dump(vectorizer, open('transform.pkl','wb'))


feature_names = vectorizer.get_feature_names()
 
#get tfidf vector for first document
first_document_vector=X[0]
 
#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  #splitting

print('Train Test Done')

from xgboost import XGBClassifier
xgb_class = XGBClassifier()
xgb_class.fit(X_train,y_train)

xgb_class.score(X_test,y_test)
print(xgb_class.score(X_test,y_test))

predict = np.array(['Khwajaavais@123'])
X_predict = vectorizer.transform(predict)
Y_predict = xgb_class.predict(X_predict)
print(Y_predict[0]) 


file = open('nlp_model.pkl', 'wb')

# dump information to that file
pickle.dump(xgb_class,file)