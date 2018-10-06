
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
'''quoting = 3 signifies we are ignoring double quotes '''

#Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = [] 
for i in range(0,1000):
    '''only keeping the letters'''
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    '''lowercase'''
    review = review.lower()
    ''' removing non significant words# ps for stemming''' '''Stemming'''
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    '''making it a string from the list'''
    review = ' '.join(review)
    corpus.append(review)

''' cleaning the text'''
''' creating the bag of words '''
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)#1500 most significant words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

'''Using Naive Bayes Classification'''

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)   

 