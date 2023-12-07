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
* Lowercase;
* Tokenizer;
* Remove special characters;
* Remove stopwords;
* Remove punctuation;
* Stemming;

All of this is basically in this function:
```
def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  filtered_text = [ps.stem(word) for word in text if word.isalnum() and  word not in stopwords.words('english') and word not in string.punctuation]

  return " ".join(filtered_text)
```
<h2>Data Building</h2>

* `TfidfVectorizer()` is ... 
* `GaussianNB()` is ...
* `accuracy_score()` is ... 
* `precision_score()` is ... 
* `confusion_matrix()` is ... 

<h2>Imbalanced Data Problem</h2>

* `TomekLinks()` is ... 
* `SMOTE()` is ... 

<h2>Results</h2>