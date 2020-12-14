import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

base_dir = "./checkpoints"


def main():
    for dir in os.listdir(base_dir):
        metrics_file = os.path.join(base_dir, dir, "metrics.csv")
        params_file = os.path.join(base_dir, dir, "params.pkl")

        df = pd.read_csv(metrics_file)
        with open(params_file, "rb") as f:
            params = pickle.load(f)

        risk_path = os.path.join(base_dir, dir, "train_risk.png")
        plot_risk(10, params, risk_path, df["train_loss_start"], df["train_loss_end"], df["test_loss_start"],
                  df["test_loss_end"])

        acc_path = os.path.join(base_dir, dir, "train_acc.png")
        plot_acc(10, params, acc_path, df["train_acc_start"], df["train_acc_end"], df["test_acc_start"],
                  df["test_acc_end"])


def plot_risk(num_iters, params, image_path, train_loss_start, train_loss_end, test_loss_start, test_loss_end):
    offset = 0.8
    plt.figure()
    for i in range(num_iters):
        if i == 0:
            label = "train"
        else:
            label = None
        plt.plot([i, i + offset], [train_loss_start[i], train_loss_end[i]], 'b*-', label=label)
        if i < num_iters - 1:
            plt.plot([i + offset, i + 1], [train_loss_end[i], train_loss_start[i + 1]], 'g--')

    for i in range(num_iters):
        if i == 0:
            label = "val"
        else:
            label = None
        plt.plot([i, i + offset], [test_loss_start[i], test_loss_end[i]], 'r*-', label=label)
        if i < num_iters - 1:
            plt.plot([i + offset, i + 1], [test_loss_end[i], test_loss_start[i + 1]], 'y--')

    epsilon = params['epsilon']
    attack = params['attack']
    plt.title(f"Performative Risk, $\epsilon$={epsilon}, att={attack}", fontsize=16)
    plt.xlabel("Step", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(loc="best")

    plt.savefig(image_path, dpi=300)


def plot_acc(num_iters, params, image_path, train_acc_start, train_acc_end, test_acc_start, test_acc_end):
    offset = 0.8
    plt.figure()
    for i in range(num_iters):
        if i == 0:
            label = "train"
        else:
            label = None
        plt.plot([i, i + offset], [train_acc_start[i], train_acc_end[i]], 'b*-', label=label)
        if i < num_iters - 1:
            plt.plot([i + offset, i + 1], [train_acc_end[i], train_acc_start[i + 1]], 'g--')

    for i in range(num_iters):
        if i == 0:
            label = "val"
        else:
            label = None
        plt.plot([i, i + offset], [test_acc_start[i], test_acc_end[i]], 'r*-', label=label)
        if i < num_iters - 1:
            plt.plot([i + offset, i + 1], [test_acc_end[i], test_acc_start[i + 1]], 'y--')

    epsilon = params['epsilon']
    attack = params['attack']
    plt.title(f"Performative Accuracy, $\epsilon$={epsilon}, att={attack}", fontsize=16)
    plt.xlabel("Step", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.legend(loc="best")

    plt.savefig(image_path, dpi=300)


if __name__ == "__main__":
    main()
