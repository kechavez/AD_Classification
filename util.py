import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools 
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

# def ps_tadpole2(raw_data, train_split):
def preprocess_volumetric(raw_data, train_split):
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

def preprocess_all_feat(raw_data):
  # Drop missing values
  raw_data_cleaned=raw_data.dropna(how='any')

  #raw_data_cleaned=raw_data_cleaned[(raw_data_cleaned!=' ').all(1)]

  # Convert 'DX' to 2 labels only: MCI is considered Dementia
  # raw_data_cleaned=collapse_dx(raw_data_cleaned)
  raw_data_cleaned= binary_only_dx(raw_data_cleaned)

  # Set some features as categorical
  xcat_p = raw_data_cleaned[['PTGENDER','PTMARRY','APOE4']]
  raw_data_cleaned.drop(['PTGENDER','PTMARRY','APOE4'], axis=1, inplace=True)
  #PTGENDER: 0:Female; 1: Male -- #PTMARRY: 0:Divorced; 1: Married; 2: Never Married 4:Widowed

  y_p = raw_data_cleaned[['DX']]
  raw_data_cleaned.drop(['DX'], axis=1, inplace=True)
  #DX: 0: Dementia, 1:Normal

  le = preprocessing.LabelEncoder()
  xcat=xcat_p.apply(le.fit_transform)
  x=pd.concat([xcat,raw_data_cleaned],axis=1,join='inner')

  # Set 'DX' (Demented or Not) as categorical
  y=y_p.apply(le.fit_transform)
  comb=pd.concat([x,y],axis=1,join='inner')
  clean_comb = comb.apply(pd.to_numeric, errors='coerce')
  clean_comb = clean_comb.dropna()

  y = clean_comb[['DX']]
  clean_comb.drop(['DX'], axis=1, inplace=True)
  x = clean_comb
  # print('xxxxx', x)

  return x, y

def split_data(x,y,train_split):
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
  # ret = ret[ret['DX'] != 'NL to MCI']
  # ret = ret[ret['DX'] != 'MCI to NL']

  # if 'DX' in ret:
  #   ret['DX'][ret['DX'] == 'MCI to Dementia'] = 'Dementia'
  #   ret['DX'][ret['DX'] == 'MCI'] = 'NL' # 'Dementia'

  _to = 'NL'
  ret = ret.replace('NL to MCI', _to)
  ret = ret.replace('MCI to Dementia', _to)
  ret = ret.replace('MCI to NL', _to)
  ret = ret.replace('MCI', _to)

  return ret

def binary_only_dx(raw_data):
  ret = pd.DataFrame.copy(raw_data)
  # ret = ret[ret['DX'] != 'NL to MCI']
  # ret = ret[ret['DX'] != 'MCI to NL']
  # ret = ret[ret['DX'] != 'MCI to Dementia']

  # ret = ret[ret['DX'] != 'MCI']

  _to = 'Dementia'
  ret = ret.replace('MCI to Dementia', 'Dementia')
  ret = ret.replace('MCI to NL', 'NL')
  ret = ret.replace('NL to MCI', 'NL')
  ret = ret.replace('MCI', 'Demenita')

  return ret


def hold_out_CV(x, y, svc='linear', svmc=1.0):
    splits=50
    rs = ShuffleSplit(n_splits=splits, test_size=.3, random_state=0)
    sum_train =0;
    sum_test = 0;
    for train_index, test_index in rs.split(x):
        x_pca_train,x_lda_train, x_pca_test, x_lda_test = \
          run_PCA_LDA(x.iloc[train_index],y.iloc[train_index], \
                      x.iloc[test_index], components=10)

        print('\nxltr', x_lda_train.shape, 'ylte', y.iloc[test_index].shape)
        print('y.iloc[test_index]', (y.iloc[test_index]==1).size)

        svm_model = None
        if svc == 'linear':
          svm_model = \
              LinearSVC(verbose=2, max_iter=1000000, dual=False, C=svmc)# , penalty='l1')
        else:
          svm_model = svm.SVC(kernel=svc, C=svmc, max_iter=10000000)

        svm_model.fit(x_lda_train, y.iloc[train_index])

        predicted_labels = svm_model.predict(x_lda_test)
        training_score = \
           accuracy_score(y.iloc[train_index], svm_model.predict(x_lda_train))
        testing_score = accuracy_score(y.iloc[test_index], predicted_labels)

        cnf_matrix=confusion_matrix(y.iloc[test_index], predicted_labels)
        class_names=list(['Dementia','NL'])
        # plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')

        sum_train = sum_train + training_score;
        sum_test = sum_test + testing_score;
        print('train score', training_score, ' test score', testing_score)

    print("\n\nAverage Training Score : ", sum_train/splits)
    print("Average Testing Score : ", sum_test/splits)

def run_PCA_LDA(X,y,xtest,components):
    y=np.ravel(y)
    target_names = ['Dementia', 'NL'] # 'MCI','NL','MCI to Dementia']

    pca = PCA(n_components=components)
    pca1 =  pca.fit(X)
    X_r = pca1.transform(X)
    Xtest_r = pca1.transform(xtest)

    lda = LinearDiscriminantAnalysis(n_components=10)
    lda1= lda.fit(X, y)
    X_r2 = lda1.transform(X)
    # print('xr2', X_r2.shape)
    Xtest_r2 = lda1.transform(xtest)

    # Percentage of variance explained for each component
    # print('explained variance ratio (first two components): %s'
    #       % str(pca.explained_variance_ratio_))

    # plt.figure()
    # colors = ['navy', 'turquoise'] #, 'darkorange']
    # lw = 2

    # for color, i, target_name in zip(colors, [0, 1], target_names):
    #     plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
    #                 label=target_name)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('PCA of Tadpole dataset')

    # plt.figure()
    # for color, i, target_name in zip(colors, [0, 1], target_names):
    #     plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
    #                 label=target_name)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('LDA of Tadpole dataset')

    # plt.show()

    x_pca=pd.DataFrame(X_r)
    x_lda=pd.DataFrame(X_r2)
    xtest_pca=pd.DataFrame(Xtest_r)
    xtest_lda=pd.DataFrame(Xtest_r2)
    y=pd.DataFrame(y)
    return x_pca,x_lda,xtest_pca,xtest_lda
