# Describe the setup of the program

**Fake News Detection with Python**

In this project we use machine learning algorithms to classify whether a news article is fake or real.

<b>Program</b>

The whole project is conducted in Google Colab.

<b>Machine learning algorithm</b>

Passive Aggressive Classifier

<b>Dependencies</b>

numpy, pandas, itertools, sklearn, worldcloud

<b>Dataset</b>

The dataset is in csvis available in webpages such as this and Github. It contains 6,335 entries, of which 3,171 articles are labelled as ‘real’ and 3,164 articles are labelled as ‘fake’.

Column 1: ID

Column 2: title

Column 3: Text

Column 4: label

Only the text and label columns are used in this project.

<b>Procedure</b>

## Data preprocessing

1.	Change the dataset from a csv file to a python dataframe with pandas, and import some packages for reading the dataset.

```
import numpy as np
import pandas as pd
import itertools
from google.colab import drive
drive.mount('/content/drive')
%cd drive/MyDrive/Colab\ Notebooks
df=pd.read_csv('news.csv')
```

2.	Import packages for machine learning.

```
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
The preprocessing step is split in two stages.
```

3.	Get the shape of the data and the first five records.

```
df.shape
df.head()
```
4.	Get the labels from the dataframe.
```
labels=df.label
labels.head()
```

5.	Visualize the data with bar chart and wordcloud.
```
print(df.groupby(['label'])['text'].count())
df.groupby(['label'])['text'].count().plot(kind='bar')
plt.show()
```
![image](https://user-images.githubusercontent.com/10076889/219705746-0d63486e-f1f9-40e4-9967-2582a9a6075f.png)

For the fake news sub-dataset:
```
from wordcloud import WordCloud
fake_data = df[df['label']=='FAKE']
all_words = '  '.join([text for text in fake_data.text])
wordcloud = WordCloud(width = 800, height = 500, max_font_size =  110, collocations= False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()
```
![image](https://user-images.githubusercontent.com/10076889/219705841-859458c5-ed77-456e-adc8-8ab9d7c38c3b.png)


For the real news sub-dataset:
```
real_data = df[df['label']=='REAL']
all_words = '  '.join([text for text in real_data.text])
wordcloud = WordCloud(width = 800, height = 500, max_font_size =  110, collocations= False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()
```
![image](https://user-images.githubusercontent.com/10076889/219705878-04830ef7-4354-42d8-aa3a-1c2fa51fdc28.png)


6.	split the dataset into training and testing sets.
```
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)
```

7.	Construct TF-IDF vectorizer for feature extraction. TF-IDF vectorizer measures the relative frequency of a word that appears in a document and also compares it with its frequency over all other documents. The max_df is set to be 0.7 to remove terms that appear too frequently. The terms are considered to be stopwords. Then, fit and transform the vectorizer on the training set and testing set respectively.

```
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
```

## Data analysis
8.	Use Passive Aggressive Classifier to test & train the data. It performs by reacting passively to accurate classifications and aggressively to any misclassifications. Then, predict on the testing set and calculate the accuracy with accuracy_score() from sklearn.metrics.

```
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%'
```

The accuracy is 92.9%.

9.	To visualize the analysis, build a confusion matrix to gain insight into the number of false and true negatives and positives.

```
from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
  plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation = 45)
  plt.yticks(tick_marks, classes)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    print ('normalized confusion matrix')
  else: 
      print('Confusion matrix, without normalization')
  threshold = cm.max() / 2,
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i,j],
             horizontalalignment = "center",
             color = "white" if cm[i,j] > threshold else 'black')
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes= ['FAKE', 'REAL'])
```
![image](https://user-images.githubusercontent.com/10076889/219705946-4f529037-fa6a-4c4d-a051-21bc85fc1d29.png)


## Conclusions

Passive Aggressive Classifier can show an accuracy of fake news detection up to 92% with high percentage of precision and sensitivity. In addition, the computation time for classification is less than one second.
With the outstanding accuracy and efficiency, the algorithm has a high potential to be used in real life.
