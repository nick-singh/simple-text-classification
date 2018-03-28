# Text Analysis and Topic Modeling 

1. Import the necessary libraries 

```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from string import punctuation
from sklearn import svm
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import ngrams
from itertools import chain

from sklearn.metrics import recall_score, f1_score

```
    
  2. Load the amazon review dataset that can be found [here](https://www.kaggle.com/snap/amazon-fine-food-reviews/downloads/Reviews.csv/2)
  3. Create to seperate dataframes that contain reviews that have a score >3 (positive) and score < 3 (negative). You can assume that a score of 3 is netural.
  4. Join the two dataframes and shuffle the contents
  5. Create a training set and a test set using:
  		1. Text for the training data
  		2. Score for the testing data
  6. Map the scores such that `{1:0, 2:0, 4:1, 5:1}` Where 1 & 2 represent negative and 4 & 5 represent positive
  
  7. Copy the following funciton:
  ```Python
  def text_fit(X, y, model,clf_model,coef_show=1):
    
    X_c = model.fit_transform(X)
    print('# features: {}'.format(X_c.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(X_c, y, random_state=0)
    print('# train records: {}'.format(X_train.shape[0]))
    print('# test records: {}'.format(X_test.shape[0]))
    clf = clf_model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    recall = recall_score(y_test,y_pred)
    print ('Model Recall: {}'.format(recall))
    
    if coef_show == 1: 
        w = model.get_feature_names()
        coef = clf.coef_.tolist()[0]
        coeff_df = pd.DataFrame({'Word' : w, 'Coefficient' : coef})
        coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
        print('')
        print('-Top 20 positive-')
        print(coeff_df.head(20).to_string(index=False))
        print('')
        print('-Top 20 negative-')        
        print(coeff_df.tail(20).to_string(index=False))
  
  # X is the training data
  # y is testing data
  # model is the text represented as a vector
  # clf_model is the classification algorithm
  # coe_show indicates if to show the top pos and neg coefficients
  ```
  8. Using the function from above run the following classification experiments:
  		1. Logistic regression model on word count
		2. Logistic regression model on TFIDF
		3. Logistic regression model on TFIDF + ngram

  9. The next component would be to determine what are the general topics that the reviews were based on. Copy the following function to continue with the exercise
  ```Python
  def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        print 
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print documents[doc_index]
            print
  ```
  10. Using TFIDF and Count Vectorizer models imported for `sklearn` perform topic modelling using the following topic modeling algorithms:
  
  		1. [NMF](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
  		2. [LDA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation)
  		3. [SVD](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html). 

Using the following initializations for both vectorizers 
```Python
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()
```
  		
  11. Using the function from 9, for each topic model display the top 10 words and the top 4 documents
  12. Finally create a word cloud of all the words used.
  ```Python
  #install the following package: pip install wordcloud
  
  #use the following function:
  
  def print_cloud(data):
    wordcloud = WordCloud(
              background_color='white',
              max_words=200,
              max_font_size=40, 
              random_state=42
             ).generate(str(data))
             
  ```



# Notes

## Preprocessing

Preprocessing ususally involves:
1. Removing additional white spaces
2. Replacing emoji's with a word representation for example :) ==> smile
3. Removing links from the corpus
4. Removing punction
5. Removing anh HTML tags
6. Remove duplicate reviews

Here is a good blog on how to [process text](http://adataanalyst.com/scikit-learn/countvectorizer-sklearn-example/)

For this exercise we will only tokenize reviews, that is change `"This is a review"` to `['this', 'is', 'a', 'review']`

Once the text is 'clean' we will use sklearn:
1. CountVectorizer - Convert a collection of text documents to a matrix of token counts
2. TfidfVectorizer - Convert a collection of raw documents to a matrix of TF-IDF features
---

  
  ## Classification

Once the text is processed the next step is to do the actual classificaiton. For this exercise we will be using a Logistic Regression Classifier. However there are many other popular classifiers that may perform better:
1. Support Vector Machine and its variants
2. Naive Bayes and its variants
3. Random Forests and its variants 

We created a function that takes in the training set `X` , test set `y`, the model being used `model` and the classification algorithm `clf_model` as well as a variable that will show the top coefficients if true  `coef_show`

---

		
## NGrams

NGram Defn:
N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). Taken from [here](http://text-analytics101.rxnlp.com/2014/11/what-are-n-grams.html)

---

## Topic Modelling

Non Negative Matrix Factorization (NMF), Latent Dirichlet Allocation (LDA) and Single Value Decomposition (SVD)algorithms will be used to find topics in a document collection. The output of the derived topics involved assigning a numeric label to the topic and printing out the top words in a topic. 

The algorithms are not able to automatically determine the number of topics and this value must be set when running the algorithm. Comprehensive documentation on available parameters is available for both [NMF](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html), [LDA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation) and [SVD](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html). 

---
