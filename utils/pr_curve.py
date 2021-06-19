"""
Refer to [Blog](https://towardsdatascience.com/how-to-efficiently-implement-area-under-precision-recall-curve-pr-auc-a85872fd7f14)
for implementation details.
"""
__all__ = ["precision_recall", "pr_curve_vis", "pr_auc"]

import sys
import numpy as np
import matplotlib.pyplot as plt


def precision_recall(probs, labels):
    """
    Calculate precision and recall given a sequence of probabilities `probs` from
    a binary classifier, companying with a set of corresponding target `labels`,
    in which contains only 1's and 0's representing the ground truth labels for
    positive and negative samples.

    @Params
    probs: (numpy array, list)
        Probabilities for a set of predictions.
    labels: (numpy array, list)
        True labels correspond to predictions.

    @Return
    A tuple of (precision, recall, threshold), each of which is a numpy array.
    """
    sorted_idx = np.argsort(probs)[::-1]
    threshold = np.array(probs)[sorted_idx]
    labels = np.array(labels)[sorted_idx]

    cumsum = np.cumsum(labels)
    rank = np.arange(len(labels)) + 1
    precision = cumsum / rank
    recall = cumsum / cumsum[-1]
    return precision, recall, threshold


def pr_curve_vis(prec, rec, smooth=True, title=None):
    """Plot a Precision-Recall Curve, where recalls are on x-axis and precisions are on y-axis.
    If `smooth` is True, then a pr-curve without zigzag pattern is plotted.

    @Params
    prec: (numpy array, list)  A set of precisions.
    rec:  (numpy array, list)  A set of recalls.
    smooth: (bool)  Whether to plot a smoothed PR-Curve without zigzag pattern.
    tittle: (str)   If provided, add title above the plot.

    @Return
    fig: (plt.figure) Figure of PR-Curve
    """
    if isinstance(prec, list):
        prec = np.array(prec)
    if isinstance(rec, list):
        rec = np.array(rec)
    if smooth:
        prec = np.concatenate([[0.], prec, [0.]])
        rec = np.concatenate([[0.], rec, [1.]])
        for i in range(prec.size - 1, 0, -1):
            prec[i - 1] = np.maximum(prec[i - 1], prec[i])

    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec)
    plt.scatter(rec, prec, c="r")
    plt.grid(True)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    if title is not None:
        plt.title(str(title))
    return plt.gcf()


def pr_auc(prec, rec):
    """Return the area under a smoothed precision-recall curve (AUC).

    @Params
    prec: (numpy array) A set of precisions.
    rec: (numpy array) A set of recalls.

    @Return
    auc: (float) The area under the pr-curve
    """
    prec = np.concatenate([[0.], prec, [0.]])
    rec = np.concatenate([[0.], rec, [1.]])
    for i in range(prec.size - 1, 0, -1):
        prec[i - 1] = np.maximum(prec[i - 1], prec[i])

    # Pick indices
    i = np.where(rec[1:] != rec[:-1])[0]
    auc = np.sum((rec[i + 1] - rec[i]) * prec[i + 1])
    return auc


def _main():
    """Run in command line.

    [Usage]: python3 pr_curve.py -p PROBABILITIES -l LABELS

    Use -h/--help flag for more details.
    """
    import os
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Calculate Precision and Recall and plot the PR-Curve")
    parser.add_argument("-p",
                        "--probs",
                        type=json.loads,
                        required=True,
                        help="Probabilities of the output from a binary  classifier, values must be included in a "
                             "square bracket and separated by comma without any white space. eg. [0.65,0.8,0.1]", )
    parser.add_argument("-l",
                        "--labels",
                        type=json.loads,
                        required=True,
                        help="Ground truth labels, values must be included in a square bracket and separated by comma "
                             "without any white space, eg. [1,0,1]")
    parser.add_argument("-s", "--smooth", help="Don't smooth the pr-curve", action="store_true")
    parser.add_argument("-o", "--output", type=str, help="File path to store the pr-curve plot", default="")
    parser.add_argument("-f", "--force", help="If present, the plot will forced to write to `output` file, "
                                              "override the file if it has already existed", action="store_true")
    args = parser.parse_args()

    # Get probs
    probs = [float(x) for x in args.probs]
    labels = [float(int(x)) for x in args.labels]

    assert len(probs) == len(labels), "[Error] `probs` and `labels` must have same length."

    prec, rec, _ = precision_recall(probs, labels)
    auc = pr_auc(prec, rec)

    fig = pr_curve_vis(prec, rec, args.smooth, f"Precision-Recall Curve: AUC={auc:.2f}")
    if args.output:
        if os.path.exists(args.output) and not args.force:
            raise FileExistsError("File already exists, use -f or --force to override and save the plot.")
        fig.savefig(args.output)
    else:
        plt.show()

    np.set_printoptions(precision=2)
    sys.stdout.write("Precision: {}\n".format(prec))
    sys.stdout.write("Recall:    {}\n".format(rec))
    sys.stdout.write("AUC: {:.2f}\n".format(auc))
    sys.stdout.flush()


if __name__ == '__main__':
    sys.exit(_main())
