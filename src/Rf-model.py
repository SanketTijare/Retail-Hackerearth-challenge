import pandas as pd
import numpy as np
from random import shuffle
import sklearn
import tqdm as tqdm
from random import seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix as cm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as cr
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity


train_name = pd.read_csv('./data/train.csv')
test_name = pd.read_csv('./data/test.csv')

train = pd.read_csv('/home/local/ALGOANALYTICS/sanket/sanket/Retail_hackerearth_challenge/data/train-features.csv')
test = pd.read_csv('/home/local/ALGOANALYTICS/sanket/sanket/Retail_hackerearth_challenge/data/test-features.csv')

x_train , x_test = train_test_split(train,train_size = 0.80)
classifier_features = train.columns[0:25]
classifier_predictor = train.columns[26]
model_rf = RandomForestClassifier(n_estimators = 100,n_jobs=10 )
model_rf.fit(x_train[classifier_features], x_train[classifier_predictor])
predicted = model_rf.predict(x_test[classifier_features])
actual = x_test['labels']
print 'Classification Report :\n' ,cr(predicted,actual)
print 'Accuracy :' , accuracy_score(predicted, actual)*100
print 'Confusion Matrix:\n' , cm(predicted, actual)

predictions = model_rf.predict(test)

sub = pd.DataFrame({'image_id':test_name.image_id, 'label':predictions})
sub.to_csv('./submissions/RF-99-new-2.csv', index=False) ## ~0.59
