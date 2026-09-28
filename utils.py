import numpy as np
import matplotlib.pyplot as plt


def plot_data(
    X,
    y,
    ax,
    pos_label="y=1",
    neg_label="y=0",
    s=80,
    loc="best"
):
    """Plot binary classification data."""

    # Find positive and negative examples
    pos = y == 1
    neg = y == 0

    # Convert to 1D
    pos = pos.reshape(-1)
    neg = neg.reshape(-1)

    # Plot positive examples
    ax.scatter(
        X[pos, 0],
        X[pos, 1],
        marker="x",
        s=s,
        c="green",
        label=pos_label
    )

    # Plot negative examples
    ax.scatter(
        X[neg, 0],
        X[neg, 1],
        marker=".",
        s=s,
        label=neg_label,
        facecolors="none",
        edgecolors="#0096ff",
        linewidth=3
    )

    ax.legend(loc=loc)


# -------------------------------
# Testing the function
# -------------------------------

if __name__ == "__main__":

    X = np.array([
        [1, 2],
        [2, 3],
        [3, 1],
        [4, 5],
        [5, 2],
        [6, 4]
    ])

    y = np.array([0, 1, 0, 1, 0, 1])

    fig, ax = plt.subplots()

    plot_data(X, y, ax)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Logistic Regression Data")

    plt.show()