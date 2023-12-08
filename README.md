<h1> Spam or Ham using Naive Bayes</h1>

<h2>Objective </h2>
Using the Naive Bayes algorithm to classify messages as spam or not in a dataset before and after data balancing methods.

<h2>Skills </h2>

* Cleaning Data;
* Naive Bayes algorithm;
* Data balancing models;

<h2>Data Source </h2>

[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data) from Kaggle

<h2>Dependencies </h2>
For the project, libraries can be divided into four types:

1. General

```
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
```

2. Preprocessing

```
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
```
3. Model Building

```
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CleanTfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
```
4. Data Balancing

```
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
```
<h2>Data Cleaning</h2>

Steps:
* Separate columns using `.iloc[]` from `pandas`;
* Rename columns using `.rename()` from `pandas`;
* Convert categorical variable into numerical form with `LabelEncoder()` from `sklearn`;
* Detect nulls using `.isnull().sum()` from `pandas`;
* Detect duplicates using `.duplicate` from `pandas`.

<h2>Data Pre Processing</h2>

Steps:
* Lowercase: Converts all texto to lowercase;
* Tokenizer: Splits text into a list of individual words;
* Remove special characters;
* Remove stopwords: Removes words that may be considered irrelevant to the results;
* Remove punctuation;
* Stemming: Reduces words to their base or root forms. Stemization aids in the grouping of related words and the reduction of vocabulary dimensionality;

All of this is basically in this function:
```
def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  filtered_text = [ps.stem(word) for word in text if word.isalnum() and  word not in stopwords.words('english') and word not in string.punctuation]

  return " ".join(filtered_text)
```
<h2>Data Building</h2>

* `TfidfVectorizer()` is a technique from the sckit-learn library is used to evaluate the relative value os a word in a document in comparison to a set of documents; 
* `GaussianNB()` is a Naive Bayes Gaussian model;
* `accuracy_score()` is the metrics that is defined as the ratio of true positives and true negatives to all positive and negative observations;  
* `precision_score()` measures the proportion of positively predicted labels that are actually correct;  
* `confusion_matrix()` is a matrix that summarizes the performance of a machine learning model on a set of test data; 

<h2>Imbalanced Data Problem</h2>

![text](https://miro.medium.com/v2/resize:fit:828/format:webp/1*7xf9e1EaoK5n05izIFBouA.png)

* `TomekLinks()` is a undersampling technique that remove examples from the majority class; 
* `SMOTE()` is a oversampling technique that generate synthetic examples of the minority class; 

<h2>Results</h2>

Before sampling techniques:

* Accuracy Score: 84.9%
* Precision Score: 45.3%

After sampling techniques:

* Undersampling
  * Accuracy Score: 87.1%
  * Precision Score: 49.1%

* Oversampling
  * Accuracy Score: 92.3%
  * Precision Score: 86.7%

In Naive Bayes Gaussian, oversampling has the best result.