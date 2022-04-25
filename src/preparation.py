""" Script to data preparation"""
import warnings
import yaml
import pandas as pd
from category_encoders import TargetEncoder
from imblearn.pipeline import Pipeline as Pipeline_imb
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import (
    LabelEncoder,
    PowerTransformer,
    StandardScaler,
)
from sklearn_pandas import DataFrameMapper

warnings.filterwarnings("ignore")


def df_x_y(x, y):
    """Return transformed df"""
    x_transform = pd.DataFrame(mapper.fit_transform(x, y), columns=x.columns)
    transform = pd.concat([x_transform, y], axis=1)

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
    "prim_sec_hypothyroid",
    inplace=True,
)

x_df = thyroid_disease.drop("Class", axis=1)
y_df = thyroid_disease.Class

# Split
sss_train_test = StratifiedShuffleSplit(
    n_splits=n_splits, test_size=test_size, random_state=random_state
)

for train_index, test_index in sss_train_test.split(x_df, y_df):
    x_train, x_test = x_df.iloc[list(train_index)], x_df.iloc[list(test_index)]

train_x = x_df.loc[x_train.index]
train_y = y_df.loc[x_train.index]

test_x = x_df.loc[x_test.index]
test_y = y_df.loc[x_test.index]

# Transform
cat = train_x.select_dtypes(include=["object"]).columns.to_list()
num = train_x.select_dtypes(exclude=["object"]).columns.to_list()

train_y_transform = pd.DataFrame(
    LabelEncoder().fit_transform(train_y), columns=["Class"]
)

mapper = DataFrameMapper(
    [
        (cat, [TargetEncoder(verbose=0, cols=None, return_df=True), StandardScaler()]),
        (num, [KNNImputer(n_neighbors=n_neighbors), PowerTransformer()]),
    ]
)

train_transform = df_x_y(train_x, train_y_transform)


# sample_pipe = Pipeline_imb(steps=[("smote", SMOTE(random_state=42, k_neighbors=5))])

# train_balanced = pd.DataFrame(
#     sample_pipe.fit_resample(train_transform, train_transform.Class)[0],
#     columns=thyroid_disease.columns,
# )

train_x_transform = train_transform.drop("Class", axis=1)

train_y_transform = train_transform.Class

le = LabelEncoder().fit_transform(test_y)

test_y_transform = pd.DataFrame(le, columns=["Class"])

test_x_transform = pd.DataFrame(mapper.transform(test_x), columns=test_x.columns)

train_x_transform.to_csv("./data/train_X_transform.csv", sep=";")
train_y_transform.to_csv("./data/train_y_transform.csv", sep=";")
test_x_transform.to_csv("./data/test_X_transform.csv", sep=";")
test_y_transform.to_csv("./data/test_y_transform.csv", sep=";")
