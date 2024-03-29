"""Script to plotting feature importance"""

import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

with open("./best_model.pickle", "rb") as b_m:
    best_model = pickle.load(b_m)

test_X_transform = pd.read_csv("./data/test_X_transform.csv", sep=";", index_col=0)

importance = pd.DataFrame(best_model.coef_, columns = test_X_transform.columns)

importance_T = importance.T
importance_T["feature"] = importance_T.index
importance_T.rename(
    columns={
        0: "compensated_hypothyroid",
        1: "negative",
        2: "prim_sec_hypothyroid",
        3: "secondary_hypothyroid",
    },
    inplace=True,
)

for i in importance_T.columns[:-1]:
    plt.figure(figsize=(25, 15))

    x1 = abs(importance_T[i]).sort_values(ascending=False)
    y1 = abs(importance_T[i]).sort_values(ascending=False).index

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

    filename = f"FI_plot_{i}.png"

    plt.savefig(filename)

    # save plot
    with open(f"plots_file_{i}.json", "w") as plot:
        plot_dict = {
            "plot": [{"features": name, "x": val} for name, val in zip(x1, y1)]
        }
        json.dump(plot_dict, plot)
