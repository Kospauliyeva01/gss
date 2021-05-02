# Install nltk python packa
!pip install nltk
import re

import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# Stopwords are words that link words in sentences but do not change their meaning
all_kz_stopwords = stopwords.words('kazakh')
all_en_stopwords = stopwords.words('english')
all_ru_stopwords = stopwords.words('russian')

print('All Kazakh stopwords:')
print(all_ru_stopwords)

print('\n\nAll English stopwords:')
print(all_en_stopwords)

print('\n\nAll Russian stopwords:')
print(all_ru_stopwords)
dataset = pd.read_csv('../input/qweqew/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
dataset.head(10)
from nltk.stem.porter import PorterStemmer


# Here you can see for example how we prepare our reviews
for i in range(0, 1):
    # With the "re" library we take only letters without extra characters such as: ",", ".", "(", ")"
    review = re.sub("[^a-zA-Z]", ' ', dataset['Review'][i])
    print(f'After .sub():\t\t\t {review}')
    
    # Translate all our words into lower case
    review = review.lower()
    print(f'After .lower():\t\t\t {review}')
    
    # Use split to put each word in the list
    review = review.split()
    print(f'After .split():\t\t\t {review}')
    
    # Initializing the PorterStemmer
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    
    # Delete the word "not" because the person could have written "not like". 
    # If we don't remove the word "not" from the stopwords list, 
    # then removing stopwords from the sentence would make the bad review good
    all_stopwords.remove('not')
    
    # Using .stem() we remove the endings from the words: Loved -> Love
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    
    # Using .join() we join our list words to str, separated by spaces
    review = ' '.join(review)
    print(f'After all the operations:\t {review}')
    from nltk.stem.porter import PorterStemmer


# Processing the whole dataset
corpus = []
for i in range(0, 1000):
    review = re.sub("[^a-zA-Z]", ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

corpus[:10]
import matplotlib.pyplot as plt
from nltk.probability import FreqDist

# put all the words in one list
all_words = [x.split() for x in corpus]
all_words = [item for sublist in all_words for item in sublist]

# Count the number of each word and store it as ('word': count): ('like': 10), ('bad': 5), ('place': 7)
data_analysis = nltk.FreqDist(all_words)
freq_dict = dict([(m, n) for m, n in data_analysis.items()])

# Draw a graph where you can see the number of repetitions of each word
plt.figure(figsize = (20, 7))
data_analysis.plot(50, cumulative=False)
# Translate all sentences into a vector, since we need to pass a mathematical representation of the sentences to the machine learning input
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

print(X_train, y_train)
print()
print(X_test, y_test)
# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

print(f'Accuracy score: {accuracy_score(y_test, y_pred) * 100}%')