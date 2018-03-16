#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 00:36:07 2018

@author: sandeepn
"""

import pandas as pd
import numpy as np
import os
os.chdir('/Users/sandeepn/Desktop/PHD_Hackathon')

# import train data

trainDataOriginal = pd.read_csv("Train.csv")
trainDataOriginal.info()
trainAdditional = pd.read_csv("Train_AdditionalData.csv")
testOriginal = pd.read_csv("Test.csv")
testAdditional = pd.read_csv("Test_AdditionalData.csv")


#type(trainDataOriginal)
def getList(col):
    tA = set(col)
    tA_l = {x for x in tA if x==x}
    return list(tA_l)

def intersection(df):
    testA = getList(df.TestA)
    #print(testA)
    testB = getList(df.TestB)
    #print('---------------')
    #print(testB)
    interAB = pd.Series(np.intersect1d(testA,testB))
    return interAB
common = intersection(trainAdditional)
#print(common)
    



#val = np.intersect1d(ar1=pd.Series(tA), ar2=pd.Series(tB))
#val


def testedResults(tIDs, listOfIds):
    idslist = []
    for idval in tIDs:
        if idval in listOfIds:
            idslist.append("1")
        else:
            idslist.append("0")
    return idslist

#tID = getList(trainDataOriginal.ID)
#tAB = testedResults(tID, getList(common))

def addTestsToData(dataSet, additionalData):
    common = intersection(additionalData)
    tID = getList(dataSet.ID)
    tAB = testedResults(tID, getList(common))
    dict = {"tested": tAB}
    df = pd.DataFrame.from_dict(dict)
    #dfMerged = pd.merge(dataSet, df)
    dataSet["tested"] = df.tested
    ### intersection of trainID and testA ids
    interIDandA = pd.Series(np.intersect1d(getList(dataSet.ID), getList(additionalData.TestA)))
    interIDandB = pd.Series(np.intersect1d(getList(dataSet.ID), getList(additionalData.TestB)))
    tA = testedResults(tID, getList(interIDandA))
    tB = testedResults(tID, getList(interIDandB))
    dictA_B = {'testA': tA, 'testB': tB}
    df1 = pd.DataFrame.from_dict(dictA_B)
    dataSet['testA'] = df1.testA
    #dataSet['testB'] = df1.testB
    return dataSet



#Describe gives statistical information about numerical columns in the dataset
trainDataOriginal.describe()
trainAdditional.head()

# =============================================================================
# combine additional data with train and test
# =============================================================================

trainData = addTestsToData(trainDataOriginal, trainAdditional)
trainData.head()
testData = addTestsToData(testOriginal, testAdditional)
testData.head()

colNames = trainData.columns
colNames

# =============================================================================
# convert Number of Cylinders convert to object
# =============================================================================
trainData['Number of Cylinders'] = trainData['Number of Cylinders'].astype('object')
testData['Number of Cylinders'] = testData['Number of Cylinders'].astype('object')

#info method provides information about dataset like 
#total values in each column, null/not null, datatype, memory occupied etc
trainData.info()


### Drop ID column
trainData = trainData.drop('ID', axis=1)
trainData.columns
testIDs = testData[['ID']]
testData = testData.drop('ID', axis=1)

# =============================================================================
# =============================================================================
# # Feature Selection
# # Univariate Selection
# =============================================================================
# =============================================================================

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = trainData.values
X = array[:,1:]
Y = array[:,0]
# feature extraction
test_chi = SelectKBest(score_func=chi2, k=8)
fit_chi = test_chi.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])






# =============================================================================
# =============================================================================
# =============================================================================
# # # ### Drop variables

# check for different cases
# =============================================================================
# =============================================================================
# =============================================================================

trainData = trainData.drop('main bearing type', axis = 1)
trainData = trainData.drop('Bearing Vendor', axis = 1)
trainData = trainData.drop('Lubrication', axis = 1)

testData = testData.drop('main bearing type', axis = 1)
testData = testData.drop('Bearing Vendor', axis = 1)
testData = testData.drop('Lubrication', axis = 1)

trainData.info()
trainData.shape


from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
X = pd.DataFrame(trainData)
X_test = pd.DataFrame(testData)

dfImp = DataFrameImputer()
dfImp.fit(X)
train_Imputed = dfImp.fit_transform(X)
train_Imputed.head()

dfImp1 = DataFrameImputer()
dfImp1.fit(X_test)
test_Imputed = dfImp1.fit_transform(X_test)


train_Imputed.isnull().sum()
train_Imputed.info()
test_Imputed.isnull().sum()

#train_Imputed['main bearing type'].unique()
#test_Imputed['main bearing type'].unique()


#### one hot encoding

train_Imputed['Number of Cylinders'] = train_Imputed['Number of Cylinders'].astype('object')
test_Imputed['Number of Cylinders'] = test_Imputed['Number of Cylinders'].astype('object')


obj_cols = ["Number of Cylinders", 'material grade', 'Valve Type', 'Fuel Type', 'Compression ratio',
        'cam arrangement', 'Cylinder arragement', 'Turbocharger','Varaible Valve Timing (VVT)', 'Cylinder deactivation',
        'Direct injection', 'displacement', 'piston type','Max. Torque', 'Peak Power',
        'Crankshaft Design', 'Liner Design ']

# =============================================================================
# obj_cols = ['Fuel Type', 'Compression ratio', 'cam arrangement', 'Cylinder arragement', 'Turbocharger', 
#             'Varaible Valve Timing (VVT)', 'Cylinder deactivation', 'Direct injection',
#             'displacement', 'piston type','Max. Torque', 'Peak Power']
# =============================================================================
#[ print("Train: {}, test: {}".format(train_Imputed[col].unique(), test_Imputed[col].unique()))  for col in cols ]

# =============================================================================
# ##### Label Encoding
# =============================================================================
# =============================================================================
# train_Imputed['material grade'] = train_Imputed['material grade'].astype('category')
# train_Imputed['Lubrication'] = train_Imputed['Lubrication'].astype('category')
# train_Imputed['Valve Type'] = train_Imputed['Valve Type'].astype('category')
# train_Imputed['Bearing Vendor'] = train_Imputed['Bearing Vendor'].astype('category')
# train_Imputed['Liner Design '] = train_Imputed['Liner Design '].astype('category')
# train_Imputed['Crankshaft Design'] = train_Imputed['Crankshaft Design'].astype('category')
# 
# test_Imputed['material grade'] = test_Imputed['material grade'].astype('category')
# test_Imputed['Lubrication'] = test_Imputed['Lubrication'].astype('category')
# test_Imputed['Valve Type'] = test_Imputed['Valve Type'].astype('category')
# test_Imputed['Bearing Vendor'] = test_Imputed['Bearing Vendor'].astype('category')
# test_Imputed['Liner Design '] = test_Imputed['Liner Design '].astype('category')
# test_Imputed['Crankshaft Design'] = test_Imputed['Crankshaft Design'].astype('category')
# =============================================================================



# =============================================================================
# int_df = train_Imputed.select_dtypes(include=['float64']).copy()
# int_df.insert(0, 'sl.no', range(0, 0 + len(train_Imputed)))
# print(int_df.head())
# =============================================================================

obj_df = train_Imputed.select_dtypes(include=['object']).copy()
#obj_df.insert(0, 'sl.no', range(0, 0 + len(train_Imputed)))
obj_df.head()
obj_df.shape

# =============================================================================
# cat_df = train_Imputed.select_dtypes(include=['category']).copy()
# cat_df.insert(0, 'sl.no', range(0, 0 + len(train_Imputed)))
# cat_df.shape
# 
# =============================================================================
# =============================================================================
# int_df_test = test_Imputed.select_dtypes(include=['float64']).copy()
# int_df_test.insert(0, 'sl.no', range(0, 0 + len(test_Imputed)))
# print(int_df_test.head())
# =============================================================================

obj_df_test = test_Imputed.select_dtypes(include=['object']).copy()
#obj_df_test.insert(0, 'sl.no', range(0, 0 + len(test_Imputed)))
obj_df_test.shape


# =============================================================================
# cat_df_test = test_Imputed.select_dtypes(include=['category']).copy()
# cat_df_test.insert(0, 'sl.no', range(0, 0 + len(test_Imputed)))
# cat_df_test.head()
# =============================================================================

dummy_df_tr  = pd.get_dummies(obj_df, columns=obj_cols ,drop_first=True)
dummy_df_tr.head()
dummy_df_tr.shape
dummy_df_te  = pd.get_dummies(obj_df_test, columns=obj_cols ,drop_first=True)
dummy_df_te.shape


# =============================================================================
# df_train = pd.merge(dummy_df_tr,int_df, on='sl.no')
# df_train.shape
# #df_train = pd.merge(df_train, cat_df, on='sl.no')
# df_train = df_train.drop("sl.no", axis = 1)
# df_train.head()
# =============================================================================

# =============================================================================
# cleanup_nums = {"y": {"pass": 1, "fail": 0}}
# df_train.replace(cleanup_nums, inplace=True)
# df_train.head()
# =============================================================================

# =============================================================================
# df_test = pd.merge(dummy_df_te,int_df_test, on='sl.no')
# #df_test = pd.merge(df_test, cat_df_test, on='sl.no')
# df_test = df_test.drop("sl.no", axis = 1)
# df_test.head()
# =============================================================================

df_train = dummy_df_tr
df_test = dummy_df_te

cleanup_nums = {"y": {"pass": 1, "fail": 0}}
df_train.replace(cleanup_nums, inplace=True)
df_train.head()

train_IV = df_train[[col for col in df_train.columns if "y" != col]]
train_DV = df_train[[col for col in df_train.columns if "y" == col]]
train_IV.shape


# =============================================================================
# from imblearn import under_sampling, over_sampling
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=2000, ratio = 1.0)
# x_res, y_res = sm.fit_sample(train_IV, df_train['y'])
# #print training_target.value_counts(), np.bincount(y_res)
# x_res.shape
# =============================================================================

x_data = np.array(train_IV)
y_data = np.array(train_DV)
test_npData = np.array(df_test)

# =============================================================================
# ## split
# =============================================================================
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42
LABELS = ["pass", "fail"]

x_train, x_validation, y_train, y_validation = train_test_split(x_data, y_data, train_size=0.2)
x_train.shape
test_npData.shape



# =============================================================================
# =============================================================================
# # # Model Building : MLP
# =============================================================================
# =============================================================================

from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers 

# Building an MLP model with single hidden layer
mlpModel = Sequential()

mlpModel.add(Dense(32, input_dim=27, activation='sigmoid'))
mlpModel.add(Dropout(0.2))
mlpModel.add(Dense(128, activation='sigmoid'))
mlpModel.add(Dropout(0.5))
mlpModel.add(Dense(1, activation='sigmoid'))

mlpModel.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

mlpModel.fit(x_train, y_train,
          epochs=500,
          batch_size=128, validation_data=(x_validation, y_validation))
print(mlpModel)
mlpModel.summary()
score = mlpModel.evaluate(x_validation, y_validation, batch_size=128)
print(score)
## Acc: 71

# float to [0,1]
predictions = np.round(mlpModel.predict(test_npData))
predictions = pd.DataFrame(predictions)
# result
result = pd.concat([testIDs, predictions], axis = 1)
result.columns = ['ID', 'y']
result['y'] = result['y'].astype(str)
result.info()
result.head()
#cleanup_nums1 = {"y": {"1": "pass", "0": "fail"}}
#result.replace(cleanup_nums1, inplace=True)
result['y'] = result['y'].str.replace("1.0", "pass")
result['y'] = result['y'].str.replace("0.0", "fail")
result.head()
result['y'] = result['y'].astype(object)
result.shape

result.to_csv("PredictTestMLP.csv", sep='\t', encoding='utf-8')


# =============================================================================
# =============================================================================
# =============================================================================
# # # AutoEncoder
# =============================================================================
# =============================================================================
# =============================================================================


inputDim = x_train.shape[1]
inputDim
encoding_dim = 15

autoencoder = Sequential()

autoencoder.add(Dropout(0.2, input_shape=(input_dim,)))
#autoencoder.add(Dense(20, activation='sigmoid'))
autoencoder.add(Dense(encoding_dim, activation='relu'))
#autoencoder.add(Dense(20, activation='sigmoid'))
autoencoder.add(Dense(input_dim, activation='linear'))

nb_epoch = 100
batch_size = 32

autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['mse'])

hist = []
for _ in range(100):
    hist.append(autoencoder.fit(x_train, x_train,
                    epochs=1,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_validation, x_validation),
                    verbose=1).history)

_hist_ = [(x['loss'][0], x['val_loss'][0]) for x in hist]

##To extract outputs from the hidden layer
from keras import backend as K
layer_output_encoded = K.function([autoencoder.layers[0].input,K.learning_phase()],
                                  [autoencoder.layers[1].output])
encoded=layer_output_encoded([x_train,0])[0]

##To extract outputs from the output layer
layer_output_decoded = K.function([autoencoder.layers[0].input, K.learning_phase()],
                                  [autoencoder.layers[2].output])
decoded = layer_output_decoded([x_train, 0])[0]

## Making predictions on the train data
predictions=autoencoder.predict(X_train)









# =============================================================================
# =============================================================================
# # CNN
# =============================================================================
# =============================================================================
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D

# set parameters:
max_features = 500
maxlen = 1000
batch_size = 32
embedding_dims = 50
filters = 64
kernel_size = 5
hidden_dims = 250
epochs = 100
dropout = 0.3


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_validation = sequence.pad_sequences(x_validation, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_validation.shape)



print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))

# =============================================================================
# model.add(Dropout(0.5))
# # we add a Convolution1D, which will learn filters
# # word group filters of size filter_length:
# model.add(Conv1D(filters,
#                  kernel_size,
#                  #padding='valid',
#                  activation='relu',
#                  strides=1))
# # we use max pooling:
# #model.add(GlobalMaxPooling1D())
# model.add(MaxPooling1D(5))
# 
# 
# # We add a vanilla hidden layer:
# model.add(Dense(hidden_dims))
# model.add(Dropout(0.2))
# model.add(Activation('relu'))
# 
# # We project onto a single unit output layer, and squash it with a sigmoid:
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# =============================================================================
model.add(Dropout(dropout))
model.add(Conv1D(filters, kernel_size, activation='relu'))
model.add(MaxPooling1D(5))

model.add(Dropout(dropout))
model.add(BatchNormalization())
model.add(Conv1D(filters, kernel_size, activation='relu'))
model.add(MaxPooling1D(5))

model.add(Dropout(dropout))
model.add(BatchNormalization())
model.add(Conv1D(filters, kernel_size, activation='relu'))
model.add(MaxPooling1D(35)) 

model.add(Dropout(dropout))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(filters, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint 
checkpoint = ModelCheckpoint("model_cnn1.h5", save_best_only=True)
callbacks_list = [checkpoint]

model_history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs, validation_data=(x_validation, y_validation),
          callbacks=callbacks_list).history
                          

score = model.evaluate(x_validation, y_validation, batch_size=batch_size)
print(score)

test_d = sequence.pad_sequences(test_npData, maxlen=maxlen)
test_d.shape
# float to [0,1]
predictions = np.round(model.predict(test_d))
predictions = pd.DataFrame(predictions)
predictions.shape
# result
result = pd.concat([testIDs, predictions], axis = 1)
result.head()
result.columns = ['ID', 'y']
result['y'] = result['y'].astype(str)
result.info()
result.head()
#cleanup_nums1 = {"y": {"1": "pass", "0": "fail"}}
#result.replace(cleanup_nums1, inplace=True)
result['y'] = result['y'].str.replace("1.0", "pass")
result['y'] = result['y'].str.replace("0.0", "fail")
result.head()
result['y'] = result['y'].astype(object)
result.shape

result.to_csv("phd_cnn.csv", sep='\t', encoding='utf-8')


import matplotlib.pyplot as plt
#%matplotlib inline

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(model_history['acc'])
plt.plot(model_history['val_acc'])

# =============================================================================
# set parameters:
# =============================================================================
# max_features = 500
# maxlen = 1000
# batch_size = 32
# embedding_dims = 50
# filters = 64
# kernel_size = 5
# hidden_dims = 250
# epochs = 100
# dropout = 0.3
## 100 epocs => acc:86.21    valAcc:85.43
## 200 epochs
### 8s 13ms/step - loss: 0.3341 - acc: 0.8574 - val_loss: 0.4382 - val_acc: 0.8261
# =============================================================================
# trAcc: 82, valAcc: 86
# =============================================================================


# =============================================================================
# max_features = 500
# maxlen = 1000
# batch_size = 32
# embedding_dims = 50
# filters = 125
# kernel_size = 5
# hidden_dims = 250
# epochs = 200
# dropout = 0.3
# trainAcc: 89.54; validAcc:81.8
# =============================================================================














