# %% [markdown]
# # Imports

# %%
import warnings
import json
import category_encoders as ce
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import statsmodels
import statsmodels.api as sm
import pickle
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as Pipeline_imb
from matplotlib import pyplot
from numpy import isnan
from scipy import stats
from scipy.stats import norm
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    plot_confusion_matrix,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)
from sklearn_pandas import DataFrameMapper, gen_features
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# %% [markdown]
# # Functions

# %%
def confus_matr(true_y_value, predicted_y_value):

  conf = confusion_matrix(true_y_value, predicted_y_value)

  plt.figure(figsize=(20, 5))
  plt.subplots_adjust(wspace=0.6, hspace=10) 
  p1 = plt.subplot(1, 2, 1) # nrow 1 ncol 2 index 1 
  plt.title('Logistic regression confusion matrix', fontdict={'fontsize':16}, pad=12, fontweight= 'semibold');
  sns.heatmap(conf/np.sum(conf, axis=1).reshape(-1,1),
              annot=conf/np.sum(conf, axis=1).reshape(-1,1),
              annot_kws={"size": 18},
              fmt='.2f',
              yticklabels=class_names,
              xticklabels=class_names,
              cmap='YlGn',
              cbar_kws={"shrink": .82},
              linewidths=0.2, 
              linecolor='gray'
              );
  plt.xlabel('Predicted label');
  plt.ylabel('True label');   


  p2 = plt.subplot(1, 2, 2)
  res = sns.heatmap(conf, annot=True, vmin=0, vmax=5, 
                    annot_kws={"size": 18},
                    yticklabels=class_names,
                    xticklabels=class_names,
                    fmt='d', cmap='YlGn', 
                    cbar_kws={"shrink": .82},
                    linewidths=0.2, 
                    linecolor='gray')

  plt.title('Logistic regression confusion matrix', fontdict={'fontsize':16}, pad=12, fontweight= 'semibold');
  plt.xlabel('Predicted label');
  plt.ylabel('True label');



  # plt.savefig('conf_matr.png')


# %%
def model_result(model, X_tran, y_train, X_test, y_test):
  mod = model.fit(X_tran, y_train)
  y_pred = mod.predict(X_test)

  conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred), index=class_names, columns=class_names)
  conf_matrix['Total'] = conf_matrix.negative + conf_matrix.comp_hypoth + conf_matrix['prim/sec hypothyroid']
  # print('Confusion matrix:', '\n', conf_matrix, '\n')

  confus_matr(y_test, y_pred)

  # print('Classification_report:', '\n', classification_report(y_test, y_pred, target_names=class_names))

  pickle.dump(model, open('model_name.pickle', 'wb'))

  

# %%
def df_X_y(X, y):
  
  X_transform = pd.DataFrame(mapper.fit_transform(X, y), columns=X.columns)
  transform = pd.concat([X_transform, y], axis=1)

  return transform

# %%
def CV(model, X, y, cv):
    fitted = model.fit(X,y)
    CV_score = np.mean(cross_val_score(fitted, X = X, y = y, cv = cv, scoring = 'recall_macro'))
    print('Recall_macro_cv = {}'.format(CV_score))

    with open('metrics.json', 'w') as outfile:
        json.dump({'report': CV_score}, outfile)



# Params 
import yaml

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)

n_neighbors = params['n_neighbors']
random_state = params['random_state']
n_splits = params['n_splits']
test_size = params['test_size']

# %% [markdown]
# # Read data

# %%
thyroid_disease = pd.read_csv('/home/tanya/Education/HW2_dvc_cicd/data/dataset_57_hypothyroid.csv', na_values='?')
thyroid_disease = thyroid_disease.drop(thyroid_disease[['TBG']], axis=1)

# %%
# Divide the data frame by categorical and numeric for comfortable preprocessing

categorical = thyroid_disease.select_dtypes(include='object')

numerical = thyroid_disease.select_dtypes(exclude='object')

# %%
if 'TBG_measured' in categorical.columns:
  categorical.drop('TBG_measured', axis=1, inplace=True)
if 'TBG_measured' in thyroid_disease.columns:
  thyroid_disease.drop('TBG_measured', axis=1, inplace=True)

# %%
thyroid_disease.drop((thyroid_disease.loc[
    (thyroid_disease['TSH'].isna())&
    (thyroid_disease['T3'].isna())&
    (thyroid_disease['TT4'].isna())&
    (thyroid_disease['T4U'].isna())&
    (thyroid_disease['FTI'].isna())
  ].index), axis = 0, inplace=True)

# %%
thyroid_disease.drop_duplicates(inplace=True)

thyroid_disease.drop((thyroid_disease.loc[thyroid_disease['age'] > 120]).index, inplace=True)

# %%
# Correcting classes
thyroid_disease['Class'].replace(['primary_hypothyroid','secondary_hypothyroid'], 'prim/sec hypothyroid', inplace=True)

# %%
X = thyroid_disease.drop('Class', axis=1)
y = thyroid_disease.Class
sss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

for train_index, test_index in sss_train_test.split(X, y):
  X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]

# %%
train = thyroid_disease.loc[X_train.index]
test = thyroid_disease.loc[X_test.index]

train_X = thyroid_disease.loc[X_train.index].drop('Class', axis=1)
train_y = thyroid_disease.loc[X_train.index].Class

test_X= thyroid_disease.loc[X_test.index].drop('Class', axis=1)
test_y = thyroid_disease.loc[X_test.index].Class

# %%
cat = train_X.select_dtypes(include=['object']).columns.to_list()

num = train_X.select_dtypes(exclude=['object']).columns.to_list()

train_y_transform = pd.DataFrame(LabelEncoder().fit_transform(train_y), columns=['Class'])

mapper = DataFrameMapper([
                          (cat, [TargetEncoder(verbose=0, cols=None, return_df=True), StandardScaler()]),
                          (num, [KNNImputer(n_neighbors=n_neighbors), PowerTransformer()]),
                          
])

# %%
train_transform = df_X_y(train_X, train_y_transform)
train_X_transform = train_transform.drop('Class', axis=1)

# %%
class_names  = ['comp_hypoth', 'negative', 'prim/sec hypothyroid']

# %%
model_result(LogisticRegression(penalty='none', random_state=random_state),
             train_X_transform,
             train_y_transform,
             train_X_transform,  
             train_y_transform
             )

# %%
cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
CV(LogisticRegression(penalty='none'), train_X_transform, train_y_transform, cv )

# Plot Feature importance
best_model = LogisticRegression(penalty='none', random_state=random_state).fit(train_X_transform, train_y_transform)
importance = pd.DataFrame(best_model.coef_,
                   columns=train_X_transform.columns
                   )

importance_T = importance.T
importance_T['feature'] = importance_T.index
importance_T.rename(columns={0: "compensated_hypothyroid", 1: "negative", 2: 'prim/sec_hypothyroid', 3: 'secondary_hypothyroid'}, inplace=True)

for i in importance_T.columns[:-1]:
  plt.figure(figsize=(25,15))

  x1=abs(importance_T[i]).sort_values(ascending=False)
  print(x1)
  y1=abs(importance_T[i]).sort_values(ascending=False).index
  print(y1)

  fi = sns.barplot(y = y1,x = x1)
  fi.set_title(f'Feature importance for class {i}',
                                                                             fontdict={'fontsize':21}, 
                                                                             pad=12, 
                                                                             fontweight='bold'
                                                                             )
  fi.set_ylabel('features',
              fontsize = 20,
              )
  fi.set_xlabel(i,
              fontsize = 20,
              )
  fi.tick_params(axis="both", labelsize=20)


  # save plot

  with open('plots_file.json', 'w') as p:
    plot_dict = {
      'plot': [{
              'features': name,
              'x': val
            } for name, val in zip(x1, y1)]
            }
    json.dump(plot_dict, p)

# %%
