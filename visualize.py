import seaborn as sns
import matplotlib.pyplot as plt

from constants import FIGURES_DIR
from utils import get_sentiment_score

sns.set(rc={"figure.figsize": (20, 6)})


def barplot_top(df, column, year, n=10):
    top = df[[column]].groupby(by=column).size().nlargest(n=n).compute()

    ax = sns.barplot(x=top.index, y=top.values)
    if n > 15:
        ax.set(xticklabels=[])
    ax.set_title(f"Most active {column}s ({year})")
    ax.set_ylabel("Number of comments")
    ax.set_xlabel(column)
    plt.savefig(
        fname=f"{FIGURES_DIR}/barplot_top_{column}_{year}.pdf",
        bbox_inches='tight', pad_inches=0,
        format="pdf"
    )
    plt.show()


def plot_daily_comments(df, year):
    df_d = df[["created_utc"]].groupby(by="created_utc").size().compute()

    ax = sns.lineplot(x=df_d.index, y=df_d.values)

    ax.set_title(f"Number of daily comments ({year})")
    ax.set_ylabel("Number of comments")
    ax.set_xlabel("Date")
    plt.savefig(
        fname=f"{FIGURES_DIR}/plot_daily_comments_{year}.pdf",
        bbox_inches='tight', pad_inches=0,
        format="pdf"
    )

    plt.show()


def plot_daily_sentiment(df, year):
    df["sentiment"] = df.body.apply(get_sentiment_score, meta=("body", "float"))
    df_d = df[["created_utc", "sentiment"]].groupby(
        by="created_utc")["sentiment"].mean().compute()


    ax = sns.lineplot(x=df_d.index, y=df_d.values)

    ax.set_title(f"Daily mean sentiment scores ({year})")
    ax.set_ylabel("Sentiment Score")
    ax.set_xlabel("Date")
    plt.savefig(
        fname=f"{FIGURES_DIR}/plot_daily_sentiment_{year}.pdf",
        bbox_inches='tight', pad_inches=0,
        format="pdf"
    )

    plt.show()
