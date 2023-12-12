import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """
    Parses the input arguments for the process of training models.
    :return: argparse.Namespace: The validated input arguments for training models.
    """
    desc = "Classification using MLP"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--plot', default="hidden_unit", choices=["hidden_unit", "lr", "iter", "activation", "opt"],
                        help='Choose experiment to plot')
    args = parser.parse_args()
    return args

def process_data(data_path):
    df = pd.read_csv(data_path, index_col=0)
    data = df[['current_price_x', 'raw_price_x', 'discount_x', 'likes_count_x', 'cluster']]
    X = data.drop(['cluster'], axis = 'columns')
    y = data['cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
    return X_train, X_test, y_train, y_test, X, y

def evaluate_metrics(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1score = f1_score(y_test, y_pred, average='macro')
    print('Accuracy: %.4f \n' % accuracy)
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"F1 Score = {f1score}")
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.savefig('confusion_matrix.png')


def plot_increase_unit(X, y):
    cv_scores = []
    cv_scores_std = []
    hidden_unit_numbers = [[20],[40],[60],[80],[100],[120],[140],[160],[200]]
    for i in hidden_unit_numbers:
        clf_mlp = MLPClassifier(hidden_layer_sizes=i, random_state=42)
        scores = cross_val_score(clf_mlp, X, y, scoring='accuracy', cv=10, verbose=1)
        cv_scores.append(scores.mean())
        cv_scores_std.append(scores.std())

    # Plot the relationship
    plt.clf()
    plt.errorbar(hidden_unit_numbers, cv_scores, yerr=cv_scores_std, marker='x', label='Accuracy')
    plt.xlabel('Size of hidden units')
    plt.ylim(0.7, 1) # y range
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig('hidden_unit.png')
    plt.plot()

def plot_lr(X, y):
    cv_scores = []
    cv_scores_std = []
    alphas = [0.0001,0.001,0.01, 0.1,1]
    for i in alphas:
        clf_mlp = MLPClassifier(alpha=i, random_state=42)
        scores = cross_val_score(clf_mlp, X, y, scoring='accuracy', cv=10)
        cv_scores.append(scores.mean())
        cv_scores_std.append(scores.std())

    # Plot the relationship
    plt.clf()
    plt.errorbar(alphas, cv_scores, yerr=cv_scores_std, marker='x', label='Accuracy')
    plt.xscale('log')
    plt.xlabel('alpha')
    plt.ylim([0.8, 1])
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig("lr.png")
    plt.plot()

def plot_iter(X, y):
    cv_scores = []
    cv_scores_std = []
    interation_numbers = [10, 30, 50, 70, 90]
    for i in interation_numbers:
        clf_mlp = MLPClassifier(random_state=42, max_iter=i)
        scores = cross_val_score(clf_mlp, X, y, scoring='accuracy', cv=10, verbose=1)
        cv_scores.append(scores.mean())
        cv_scores_std.append(scores.std())

    plt.clf()
    # Plot the relationship
    plt.errorbar(interation_numbers, cv_scores,
    yerr=cv_scores_std, marker='x', label='Accuracy')
    plt.ylim([0.8, 1])
    plt.xlabel('Number of iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig("iter.png")
    plt.show()

def plot_activation(X, y):
    cv_scores = []
    cv_scores_std = []
    functions = ['identity', 'logistic', 'tanh', 'relu']
    for i in functions:
        clf_mlp = MLPClassifier(activation=i, random_state=42)
        scores = cross_val_score(clf_mlp, X, y, scoring='accuracy', cv=10)
        cv_scores.append(scores.mean())
        cv_scores_std.append(scores.std())

    plt.clf()
    plt.bar(functions, cv_scores, yerr=cv_scores_std, label='Accuracy')
    plt.xlabel('Activation function')
    plt.ylim([0.5, 1])
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig("activation.png")
    plt.show()

def plot_opt(X, y):
    cv_scores = []
    cv_scores_std = []
    solvers = ['lbfgs', 'sgd', 'adam']
    for i in solvers:
        clf_mlp = MLPClassifier(solver=i, random_state=42)
        scores = cross_val_score(clf_mlp, X, y, scoring='accuracy', cv=10)
        cv_scores.append(scores.mean())
        cv_scores_std.append(scores.std())

    # Plot the relationship
    plt.clf()
    plt.bar(solvers, cv_scores, yerr=cv_scores_std,
    label='Accuracy')
    plt.xlabel('Solvers')
    plt.ylim([0.5, 1])
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig("optimizer.png")
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    data_path = "output_cluster_kmeansv3.csv"
    X_train, X_test, y_train, y_test, X, y = process_data(data_path=data_path)
    clf = MLPClassifier(random_state=42)
    clf.fit(X_train, y_train)
    evaluate_metrics(clf=clf, X_test=X_test, y_test=y_test)

    # 1 Hidden layer evaluation 10-fold
    scores_mlp_default = cross_val_score(clf, X, y, cv=10, verbose=1)
    print('Accuracy range for MLP with one hidden layer: [%.4f, %.4f]; mean: %.4f; std: %.4f\n' % (scores_mlp_default.min(), 
                                                scores_mlp_default.max(), scores_mlp_default.mean(), scores_mlp_default.std()))
    # Increase 2 hidden layers evaluation 10-fold
    clf_mlp = MLPClassifier(hidden_layer_sizes=[100, 100], random_state=42)
    scores_mlp_2layers = cross_val_score(clf_mlp, X, y, scoring='accuracy', cv=10, verbose=1)
    print('Accuracy range for MLP with two hidden layers: [%.4f, %.4f]; mean: %.4f; std: %.4f\n' % (scores_mlp_2layers.min(), 
                                                scores_mlp_2layers.max(), scores_mlp_2layers.mean(), scores_mlp_2layers.std()))
    
    if args.plot == 'hidden_unit':
        plot_increase_unit(X=X, y=y)
    if args.plot == 'lr':
        plot_lr(X=X, y=y)
    if args.plot == 'iter':
        plot_iter(X=X, y=y)
    if args.plot == 'activation':
        plot_activation(X=X, y=y)
    if args.plot == 'opt':
        plot_opt(X=X, y=y)