import matplotlib.pyplot as plt

def plot_pr_curve(precision, recall):
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.grid(True)
    plt.savefig('pr_curve.png')
