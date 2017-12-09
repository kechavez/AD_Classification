import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('dementia.png')

def preprocess_split(raw_data, train_split):
  xnum = raw_data[['MR Delay','Age','EDUC','MMSE','CDR','eTIV','nWBV','ASF']]
  xcat_p = raw_data[['M/F','SES']] # M/F (Gender): 0: Female; 1: Male.
  y_p = raw_data[['Group']]
  
  le = preprocessing.LabelEncoder()
  xcat=xcat_p.apply(le.fit_transform)
  x=pd.concat([xcat,xnum],axis=1,join='inner')
  
  # Converting 'Group' (Demented or Nondemented) from numerical to categorical value
  y=y_p.apply(le.fit_transform) # 0: Converted; 1: Demented; 2: Nondemented
  
  # Split
  m=x.shape[0]    # number of data points
  
  x_train, y_train = x.iloc[0:int(m*train_split),:].as_matrix(), \
                     y.iloc[0:int(m*train_split),:].as_matrix().T[0]
  x_test, y_test = x.iloc[int(m*train_split)+1:m-1,:].as_matrix(), \
                   y.iloc[int(m*train_split)+1:m-1,:].as_matrix().T[0]

  return x_train, y_train, x_test, y_test

def ps_tadpole(raw_data, train_split):
  y_p = raw_data[['DX']]
  xcat_p = raw_data[['PTGENDER']] # M/F (Gender): 0: Female; 1: Male.
  # xnum = raw_data[['Ventricles',	'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp',	'ICV']] 
  xnum = raw_data.drop(['RID', 'DX_bl', 'DX', 'PTGENDER', 'PTMARRY', 'FLDSTRENG'], axis=1)
  xnum= xnum.apply(pd.to_numeric, errors='coerce')
  xnum = xnum.dropna(thresh=10)
  xnum = xnum.dropna(axis='columns')
  
  le = preprocessing.LabelEncoder()
  xcat=xcat_p.apply(le.fit_transform)
  x=pd.concat([xcat,xnum],axis=1,join='inner')
  
  # Converting 'Group' (Demented or Nondemented) from numerical to categorical value
  y=y_p.apply(le.fit_transform) # 0: Converted; 1: Demented; 2: Nondemented
  
  # Split
  m=x.shape[0]    # number of data points
  
  x_train, y_train = x.iloc[0:int(m*train_split),:].as_matrix(), \
                     y.iloc[0:int(m*train_split),:].as_matrix().T[0]
  x_test, y_test = x.iloc[int(m*train_split)+1:m-1,:].as_matrix(), \
                   y.iloc[int(m*train_split)+1:m-1,:].as_matrix().T[0]

  return x_train, y_train, x_test, y_test

def ps_tadpole2(raw_data, train_split):
  y_p = raw_data[['DX']]
  xnum = raw_data.drop(['DX'], axis=1)
  xnum= xnum.apply(pd.to_numeric, errors='coerce')
  xnum = xnum.dropna()
  # xnum = xnum.dropna(axis='columns')
  # print('xn', xnum)
  
  le = preprocessing.LabelEncoder()
  x=xnum
  
  # Converting 'Group' (Demented or Nondemented) from numerical to categorical value
  y=y_p.apply(le.fit_transform) # 0: Converted; 1: Demented; 2: Nondemented
  
  # Split
  m=x.shape[0]    # number of data points
  
  x_train, y_train = x.iloc[0:int(m*train_split),:].as_matrix(), \
                     y.iloc[0:int(m*train_split),:].as_matrix().T[0] # NOTE T[0]
  x_test, y_test = x.iloc[int(m*train_split)+1:m-1,:].as_matrix(), \
                   y.iloc[int(m*train_split)+1:m-1,:].as_matrix().T[0] # NOTE T[0]

  return x_train, y_train, x_test, y_test


def ps2(raw_data, train_split):
  y_p = raw_data[['DX']]
  xnum = raw_data.drop(['DX'], axis=1)
  xnum= xnum.apply(pd.to_numeric, errors='coerce')
  xnum = xnum.dropna()
  # xnum = xnum.dropna(axis='columns')
  # print('xn', xnum)
  
  le = preprocessing.LabelEncoder()
  x=xnum
  
  # Converting 'Group' (Demented or Nondemented) from numerical to categorical value
  y=y_p.apply(le.fit_transform) # 0: Converted; 1: Demented; 2: Nondemented
  
  return x,y

def collapse_dx(raw_data):
  ret = pd.DataFrame.copy(raw_data)
  ret = ret[ret['DX'] != 'NL to MCI']
  ret = ret[ret['DX'] != 'MCI to NL']

  if 'DX' in ret:
    ret['DX'][ret['DX'] == 'MCI to Dementia'] = 'Dementia'
    ret['DX'][ret['DX'] == 'MCI'] = 'Dementia'

  return ret
