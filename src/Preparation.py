import warnings
import sklearn
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from sklearn.preprocessing import (
    LabelEncoder,
    PowerTransformer,
    StandardScaler,
)
from sklearn_pandas import DataFrameMapper

warnings.filterwarnings("ignore")
import yaml


def df_X_y(X, y):
    """Return transformed df"""
    X_transform = pd.DataFrame(mapper.fit_transform(X, y), columns=X.columns)
    transform = pd.concat([X_transform, y], axis=1)

    return transform


with open("./params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

n_neighbors = params["n_neighbors"]
random_state = params["random_state"]
n_splits = params["n_splits"]
test_size = params["test_size"]


# Read and clean data

thyroid_disease = pd.read_csv("./data/dataset_57_hypothyroid.csv", na_values="?")
thyroid_disease = thyroid_disease.drop(thyroid_disease[["TBG"]], axis=1)
thyroid_disease.drop("TBG_measured", axis=1, inplace=True)
thyroid_disease.drop(
    (
        thyroid_disease.loc[
            (thyroid_disease["TSH"].isna())
            & (thyroid_disease["T3"].isna())
            & (thyroid_disease["TT4"].isna())
            & (thyroid_disease["T4U"].isna())
            & (thyroid_disease["FTI"].isna())
        ].index
    ),
    axis=0,
    inplace=True,
)
thyroid_disease.drop_duplicates(inplace=True)
thyroid_disease.drop(
    (thyroid_disease.loc[thyroid_disease["age"] > 120]).index, inplace=True
)

# Correcting classes
thyroid_disease["Class"].replace(
    ["primary_hypothyroid", "secondary_hypothyroid"],
    "prim/sec hypothyroid",
    inplace=True,
)

X = thyroid_disease.drop("Class", axis=1)
y = thyroid_disease.Class

# Split
sss_train_test = StratifiedShuffleSplit(
    n_splits=1, test_size=test_size, random_state=random_state
)

for train_index, test_index in sss_train_test.split(X, y):
    X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]

train_X = X.loc[X_train.index]
train_y = y.loc[X_train.index]

test_X = X.loc[X_test.index]
test_y = y.loc[X_test.index]

# Transform
cat = train_X.select_dtypes(include=["object"]).columns.to_list()
num = train_X.select_dtypes(exclude=["object"]).columns.to_list()

train_y_transform = pd.DataFrame(
    LabelEncoder().fit_transform(train_y), columns=["Class"]
)

mapper = DataFrameMapper(
    [
        (cat, [TargetEncoder(verbose=0, cols=None, return_df=True), StandardScaler()]),
        (num, [KNNImputer(n_neighbors=n_neighbors), PowerTransformer()]),
    ]
)

train_transform = df_X_y(train_X, train_y_transform)
train_X_transform = train_transform.drop("Class", axis=1)

le = LabelEncoder().fit_transform(test_y)

test_y_transform = pd.DataFrame(le, columns=["Class"])

test_X_transform = pd.DataFrame(mapper.transform(test_X), columns=test_X.columns)

train_X_transform.to_csv("./data/train_X_transform.csv", sep=";")
train_y_transform.to_csv("./data/train_y_transform.csv", sep=";")
test_X_transform.to_csv("./data/test_X_transform.csv", sep=";")
test_y_transform.to_csv("./data/test_y_transform.csv", sep=";")
