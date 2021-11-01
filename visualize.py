import seaborn as sns
import matplotlib.pyplot as plt


def barplot_top(df, column, n=10):
    df_grouped = df.groupby(by=column).size()
    top = df_grouped.nlargest(n=n)
    top = top.compute()

    sns.set(rc={'figure.figsize': (20, 6)})

    ax = sns.barplot(x=top.index, y=top.values)
    if n > 15:
        ax.set(xticklabels=[])
    ax.set_title(f"Most active {column}s")
    ax.set_ylabel("Number of comments")
    ax.set_xlabel(column)
    plt.show()


def plot_daily_comments(df):
    df_d = df.groupby('created_utc').size().compute()

    ax = sns.lineplot(x=df_d.index, y=df_d.values)

    ax.set_title(f"Number of daily comments")
    ax.set_ylabel("Number of comments")
    ax.set_xlabel("Date")

    plt.show()
