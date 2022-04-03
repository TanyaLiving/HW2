import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import pickle

file = open("./best_model.pickle", "rb")
best_model = pickle.load(file)
train_X_transform = pd.read_csv("./data/train_X_transform.csv", sep=";")

importance = pd.DataFrame(best_model.coef_, columns=train_X_transform.columns)

importance_T = importance.T
importance_T["feature"] = importance_T.index
importance_T.rename(
    columns={
        0: "compensated_hypothyroid",
        1: "negative",
        2: "prim/sec_hypothyroid",
        3: "secondary_hypothyroid",
    },
    inplace=True,
)

for i in importance_T.columns[:-1]:
    plt.figure(figsize=(25, 15))

    x1 = abs(importance_T[i]).sort_values(ascending=False)
    # print(x1)
    y1 = abs(importance_T[i]).sort_values(ascending=False).index
    # print(y1)

    fi = sns.barplot(y=y1, x=x1)
    fi.set_title(
        f"Feature importance for class {i}",
        fontdict={"fontsize": 21},
        pad=12,
        fontweight="bold",
    )
    fi.set_ylabel(
        "features",
        fontsize=20,
    )
    fi.set_xlabel(
        i,
        fontsize=20,
    )
    fi.tick_params(axis="both", labelsize=20)

    plt.savefig("FI_plot.png")

    # save plot

    with open("plots_file.json", "w") as p:
        plot_dict = {
            "plot": [{"features": name, "x": val} for name, val in zip(x1, y1)]
        }
        json.dump(plot_dict, p)
