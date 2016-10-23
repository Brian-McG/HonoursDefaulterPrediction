import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cycler import cycler
from lifelines.estimation import KaplanMeierFitter
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import interp
from sklearn.decomposition import PCA
from sklearn.metrics import auc
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import math

from config import constants as const

almost_black = "#262626"
palette = sns.color_palette()


def visualise_data_set(x_arr, y_arr):
    """Apply principle component analysis to the X-array to 3 dimensions and visualise the resulting matrix"""
    # Instantiate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=3)

    # Fit and transform x to visualise inside a 3D feature space
    x_visualisation = pca.fit_transform(x_arr)

    figure = plt.figure()
    axis = Axes3D(figure)

    axis.scatter(x_visualisation[y_arr == 0, 0], x_visualisation[y_arr == 0, 1], x_visualisation[y_arr == 0, 2],
                 label="Class #0",
                 edgecolor=almost_black, facecolor=palette[0], linewidth=0.3, marker="o")
    axis.scatter(x_visualisation[y_arr == 1, 0], x_visualisation[y_arr == 1, 1], x_visualisation[y_arr == 1, 2],
                 label="Class #1",
                 edgecolor=almost_black, facecolor=palette[2], linewidth=0.3, marker="^")
    axis.set_title("PCA to 3 components")

    plt.show()


def visualise_two_data_sets(x_arr, y_arr, x_arr_two, y_arr_two):
    """Apply principle component analysis to the two X-array"s to 3 dimensions and visualise the resulting matrices"""
    # Instantiate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=3)

    # Fit and transform x to visualise inside a 3D feature space
    x_visualisation = pca.fit_transform(x_arr)

    figure = plt.figure()
    axis = Axes3D(figure)

    axis.scatter(x_visualisation[y_arr == 0, 0], x_visualisation[y_arr == 0, 1], x_visualisation[y_arr == 0, 2],
                 label="Class #0",
                 edgecolor=almost_black, facecolor=palette[0], linewidth=0.3, marker="o")
    axis.scatter(x_visualisation[y_arr == 1, 0], x_visualisation[y_arr == 1, 1], x_visualisation[y_arr == 1, 2],
                 label="Class #1",
                 edgecolor=almost_black, facecolor=palette[2], linewidth=0.3, marker="^")
    axis.set_title("PCA to 3 components - data-set 1")

    x_visualisation_two = pca.transform(x_arr_two)
    figure_two = plt.figure()
    axis_two = Axes3D(figure_two)
    axis_two.scatter(x_visualisation_two[y_arr_two == 0, 0], x_visualisation_two[y_arr_two == 0, 1],
                     x_visualisation_two[y_arr_two == 0, 2],
                     label="Class #0", edgecolor=almost_black,
                     facecolor=palette[0], linewidth=0.3, marker="o")
    axis_two.scatter(x_visualisation_two[y_arr_two == 1, 0], x_visualisation_two[y_arr_two == 1, 1],
                     x_visualisation_two[y_arr_two == 1, 2],
                     label="Class #1", edgecolor=almost_black,
                     facecolor=palette[2], linewidth=0.3, marker="^")
    axis_two.set_title("PCA to 3 components - data-set 2")

    plt.show()


def plot_roc_curve_of_classifier(roc_list, data_set_description, classifier_description="classifier"):
    if const.RECORD_RESULTS is True and not (None, None) in roc_list:
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        fig = plt.figure(figsize=(12, 10))
        i = 1
        for (tpr, fpr) in roc_list:
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label="ROC fold %d (area = %0.2f)" % (i, roc_auc))
            i += 1

        mean_tpr /= len(roc_list)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, "k--", dashes=[8, 4, 2, 4, 2, 4], label="Mean ROC (area = %0.2f)" % mean_auc, lw=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random classification")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("{0} ROC curve".format(classifier_description))
        plt.legend(loc="lower right")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/{0}_{1}_roc_plot_{2}.png".format(data_set_description, classifier_description, current_time),
                    bbox_inches="tight")
        plt.close(fig)


def plot_mean_roc_curve_of_balancers(balancer_roc_list, data_set_description, classifier_description):
    if const.RECORD_RESULTS is True and not (None, None) in balancer_roc_list[0][0]:
        fig = plt.figure(figsize=(12, 10))
        monochrome = (cycler("color", ["k"]) * cycler("marker", [""]) *
                      cycler("linestyle", ["-", "--", "-."]))
        color = iter(cm.brg(np.linspace(0, 1, len(balancer_roc_list))))
        plt.rc("axes", prop_cycle=monochrome)

        for (test_run_roc_list, balancer) in balancer_roc_list:
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            for test_result in test_run_roc_list:
                for (tpr, fpr) in test_result:
                    mean_tpr += interp(mean_fpr, fpr, tpr)
                    mean_tpr[0] = 0.0

            mean_tpr /= (len(test_result) * len(test_run_roc_list))
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            c = next(color)
            plt.plot(mean_fpr, mean_tpr, c=c, lw=1, alpha=0.7, label="{0} (area = {1:.4f})".format(balancer, mean_auc))

        plt.plot([0, 1], [0, 1], "k--", label="Random classification")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("{0} ROC curve for each balancer".format(classifier_description))
        plt.legend(loc="lower right")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(
            os.path.dirname(os.path.realpath(__file__)) + "/../results/{0}_{1}_roc_balancer_plot_{2}.png".format(data_set_description, classifier_description, current_time),
            bbox_inches="tight")
        plt.close(fig)


def plot_mean_roc_curve_of_classifiers(classifier_roc_list, data_set_description):
    if const.RECORD_RESULTS is True:
        fig = plt.figure(figsize=(8, 6.66))
        monochrome = (cycler("color", ["k"]) * cycler("marker", [""]) *
                      cycler("linestyle", ["-", "--", "-."]))
        color_arr = ["#64B3DE", "#1f78b4", "#6ABF20", "#FBAC44", "#bc1659", "#B9B914", "#33a02c", "#ff7f00", "#6a3d9a", "black", "#b15928", "#e31a1c"]
        plt.rc("axes", prop_cycle=monochrome)
        line_style_index = 0
        color_index = 0

        for (test_run_roc_list, classifier_description) in classifier_roc_list:
            if not (None, None) in test_run_roc_list[0]:
                mean_tpr = 0.0
                mean_fpr = np.linspace(0, 1, 100)
                count = 0
                for roc_list in test_run_roc_list:
                    for (tpr, fpr) in roc_list:
                        mean_tpr += interp(mean_fpr, fpr, tpr)
                        mean_tpr[0] = 0.0
                        count += 1

                mean_tpr /= float(count)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                line_width = 0.5
                if line_style_index == 1:
                    line_width = 0.8
                elif line_style_index == 2:
                    line_width = 1.5

                plt.plot(mean_fpr, mean_tpr, c=color_arr[color_index], lw=line_width, alpha=1, label="{0} ({1:.3f})".format(classifier_description, mean_auc))
                line_style_index = (line_style_index + 1) % 3
                color_index += 1

        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=10)
        plt.plot([0, 1], [0, 1], "k--", label="Random classification", lw=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve for each classifier")
        plt.legend(loc="lower right", fancybox=True, frameon=True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/{0}_roc_classifier_plot_{1}.png".format(data_set_description, current_time), bbox_inches="tight")
        plt.close(fig)


def plot_balancer_results_per_classifier(data_balancer_results_per_classifier, parameter=(2,"Balanced Accuracy")):
    classifier_arr = []
    color = iter(cm.Set1(np.linspace(0, 1, len(data_balancer_results_per_classifier) + 1)))
    mean_classifier_arr = [0] * len(data_balancer_results_per_classifier[0][1])
    for (classifier_name, data_balancer_results) in data_balancer_results_per_classifier:
        individual_data_balance_plot = []
        x = 0
        for (data_balancer_name, result_arr) in data_balancer_results:
            individual_data_balance_plot.append(result_arr[parameter[0]])  # Average True rate
            mean_classifier_arr[x] += result_arr[parameter[0]]
            x += 1
        classifier_arr.append(individual_data_balance_plot)

    classifier_arr.append([value / float(len(data_balancer_results_per_classifier)) for value in mean_classifier_arr])

    fig = plt.figure(figsize=(12, 10))

    classifiers = np.arange(len(classifier_arr))
    data_balancers = np.arange(len(classifier_arr[0])) * 3
    bar_width = 0.2
    opacity = 0.9

    for i in range(len(classifier_arr)):
        if i + 1 != len(classifier_arr):
            label = data_balancer_results_per_classifier[i][0]
        else:
            label = "Mean classification"

        plt.bar(data_balancers + (i * bar_width), classifier_arr[i], bar_width,
                alpha=opacity,
                color=color.next(),
                label=label)

    plt.locator_params(axis='y', nbins=10)
    plt.xlabel("Data balance algorithm")
    plt.ylabel(parameter[1])
    plt.legend(loc="lower right", fancybox=True, frameon=True)
    plt.title("{0} per data balance algorithm".format(parameter[1]))
    plt.ylim([0.0, 1.00])
    data_balance_labels = [filter(str.isupper, data_balance_name) if data_balance_name != "None" and len(filter(str.isupper, data_balance_name)) < 6 else data_balance_name for
                           (data_balance_name, _) in data_balancer_results_per_classifier[0][1]]
    plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), data_balance_labels)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/data_balancer_results_per_classifier_plot_{0}_{1}.png".format(parameter[1], current_time))
    plt.close(fig)


def plot_kaplan_meier_graph_of_time_to_default(time_to_default, data_set_description=""):
    kmf = KaplanMeierFitter()
    kmf.fit(time_to_default, event_observed=[1] * len(time_to_default))
    ax = kmf.plot(title="{0} Kaplan Meier analysis of time to default".format(data_set_description).replace("_", " "))
    ax.get_figure().set_size_inches(12, 10)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.ylim([0.0, 1.00])
    plt.ylabel("Percentage Remaining")
    plt.xlabel("Time to default (days)")
    plt.locator_params(axis='x', nbins=10)

    ax.get_figure().savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/kaplan_meier_time_to_default_{0}.png".format(current_time), bbox_inches="tight")
    plt.close(ax.get_figure())


def plot_percentage_difference_graph(results_per_classifier, dataset, name_suffix="", parameter="Balanced Accuracy", x_label="Feature selection approach", difference_from="no feature selection"):
    print(results_per_classifier)
    patterns = (None, "////")
    no_feature_selection = results_per_classifier[0][1]
    color = iter(cm.Set1(np.linspace(0, 1, len(no_feature_selection) + 1)))
    classifier_arr = []
    for i in range(len(no_feature_selection) + 1):
        classifier_arr.append(list())
    for i in range(1, len(results_per_classifier)):
        data_balancer_results = results_per_classifier[i][1]
        x = 0
        mean_classification = 0
        for (result_arr, data_balancer_name, _) in data_balancer_results:
            value = result_arr[2] - no_feature_selection[x][0][2]
            classifier_arr[x].append(value)
            mean_classification += value
            x += 1
        mean_classification /= float(len(data_balancer_results))
        classifier_arr[x].append(mean_classification)


    fig = plt.figure(figsize=(11, 8))

    classifiers = np.arange(len(classifier_arr))
    data_balancers = np.arange(len(classifier_arr[0])) * 3
    bar_width = 0.2
    opacity = 0.9

    for i in range(len(classifier_arr)):
        if i + 1 != len(classifier_arr):
            label = results_per_classifier[0][1][i][1]
        else:
            label = "Mean classification"
        plt.bar(data_balancers + (i * bar_width), classifier_arr[i], bar_width,
                alpha=opacity,
                color=color.next(),
                hatch=patterns[i % len(patterns)],
                label=label)

    plt.xlabel(x_label)
    plt.ylabel("Difference in {0} from {1}".format(parameter, difference_from))
    legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.23), fancybox=True, frameon=True, ncol=4)
    legend.get_frame().set_facecolor('#ffffff')
    plt.title("{0} per {1} on {2} dataset".format(parameter, x_label, dataset))
    feature_selection_labels = [results_per_classifier[i][0] for i in range(1, len(results_per_classifier))]
    plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plt.locator_params(axis='y', nbins=15)
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/{3}_results_per_classifier_plot{0}_{4}_{1}_{2}.png".format(name_suffix, parameter, current_time, x_label, dataset), bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)


def plot_time_to_default_results(time_to_default_results_per_classifier, parameter="Difference in balanced accuracy from no feature selection"):
    color = iter(cm.Set1(np.linspace(0, 1, len(time_to_default_results_per_classifier[0][1]) + 1)))
    classifier_arr = []
    for i in range(len(time_to_default_results_per_classifier[0][1]) + 1):
        classifier_arr.append(list())

    for i in range(0, len(time_to_default_results_per_classifier)):
        data_balancer_results = time_to_default_results_per_classifier[i][1]
        x = 0
        mean_classification = 0
        for (result_arr, data_balancer_name, _) in data_balancer_results:
            result = result_arr[2]
            classifier_arr[x].append(result)
            mean_classification += result
            x += 1
        mean_classification /= float(len(data_balancer_results))
        classifier_arr[x].append(mean_classification)

    fig = plt.figure(figsize=(12, 10))

    classifiers = np.arange(len(classifier_arr))
    data_balancers = np.arange(len(classifier_arr[0])) * 3
    bar_width = 0.2
    opacity = 0.9

    for i in range(len(classifier_arr)):
        if i == len(classifier_arr) - 1:
            label = "Mean classification"
        else:
            label = time_to_default_results_per_classifier[0][1][i][1]
        plt.bar(data_balancers + (i * bar_width), classifier_arr[i], bar_width,
                alpha=opacity,
                color=color.next(),
                label=label)

    plt.locator_params(axis='y', nbins=10)
    plt.xlabel("Default range (days)")
    plt.ylabel(parameter)
    plt.ylim([0.0, 1.00])
    plt.legend(loc="lower right", fancybox=True, frameon=True)
    plt.title("{0} when trained on different default ranges".format(parameter))
    feature_selection_labels = [time_to_default_results_per_classifier[i][0] for i in range(0, len(time_to_default_results_per_classifier))]
    plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/time_to_default_results_per_classifier_plot_{0}_{1}.png".format(parameter, current_time))
    plt.close(fig)


def visualise_dataset_classifier_results(dataset_results):
    sns.set(style='ticks')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["s", "o", "^", "*"]
    colors = ["#64B3DE", "#1f78b4", "#6ABF20", "#FBAC44", "#bc1659", "#B9B914", "#33a02c", "#ff7f00", "#6a3d9a", "black", "#b15928", "#e31a1c"]

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_axis_on()
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    plt.xlabel("Difference in TPR")
    plt.ylabel("Difference in TNR")
    plt.title("Comparision of TNR toTPR across data sets for each classifier", y=1.03, fontsize=16)

    ax.xaxis.set_label_coords(0.1, 0.52)
    ax.yaxis.set_label_coords(0.53, 0.9)

    plt.ylim(-0.2, 0.2)
    plt.xlim(-0.2, 0.2)
    data_set_labels = []
    classifier_labels = []
    data_set_index = 0
    for (data_set, dataset_result) in dataset_results:
        data_set_labels.append(mlines.Line2D(range(1), range(1), color="white", marker=markers[data_set_index], markeredgecolor="black", markeredgewidth=1.0, label=data_set.replace("_", " ")))
        median_true_pos = np.median(np.array([result_arr[3] for (result_arr, classifier_description) in dataset_result]))
        median_true_neg = np.median(np.array([result_arr[4] for (result_arr, classifier_description) in dataset_result]))

        i = 0
        for (result_arr, classifier_description) in dataset_result:
            if data_set_index == 0:
                classifier_labels.append(mpatches.Patch(color=colors[i], label=classifier_description, alpha=0.8))
            ax.scatter(result_arr[3] - median_true_pos, result_arr[4] - median_true_neg, marker=markers[data_set_index], s=100, alpha=0.8, color=colors[i], edgecolor="white", zorder=data_set_index, lw=0)
            i += 1
        data_set_index += 1

    plt.legend(handles=data_set_labels + classifier_labels)
    sns.despine()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/classifier_dataset_plt_{0}.png".format(current_time), bbox_inches='tight')
    plt.close(fig)


def visualise_dataset_balancer_results(dataset_results):
    sns.set(style='ticks')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["s", "o", "^", "*", "v", "^", "<", ">", "H", "d"]
    colors = ["#64B3DE", "#1f78b4", "#6ABF20", "#FBAC44", "#bc1659", "#B9B914", "#33a02c", "#ff7f00", "#6a3d9a", "black", "#b15928", "#e31a1c"]
    hatches = ["///////", "*", "."]

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_axis_on()
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    plt.xlabel("Difference in TPR")
    plt.ylabel("Difference in TNR")
    plt.title("Comparision of TNR toTPR across data sets for each classifier", y=1.03, fontsize=16)

    ax.xaxis.set_label_coords(0.1, 0.52)
    ax.yaxis.set_label_coords(0.53, 0.9)

    plt.ylim(-0.2, 0.2)
    plt.xlim(-0.2, 0.2)
    data_set_labels = []
    classifier_labels = []
    data_set_index = 0
    for (data_set, dataset_result) in dataset_results:
        data_set_labels.append(mlines.Line2D(range(1), range(1), color="white", marker=markers[data_set_index], markeredgecolor="black", markeredgewidth=1.0, label=data_set.replace("_", " ")))
        true_pos_arr = []
        true_neg_arr = []

        for (classifier_description, result_arr) in dataset_result:
            for (balancer_description, results) in result_arr:
                true_pos_arr.append(results[3])
                true_neg_arr.append(results[4])

        median_true_pos = np.median(np.array(true_pos_arr))
        median_true_neg = np.median(np.array(true_neg_arr))

        i = 0
        for (classifier_description, result_arr) in dataset_result:
            balancer_index = 0
            for (balancer_description, results) in result_arr:
                if data_set_index == 0:
                    classifier_labels.append(mpatches.Patch(color=colors[i], label=classifier_description, alpha=0.8))
                ax.scatter(results[3] - median_true_pos, results[4] - median_true_neg, marker=markers[data_set_index], s=100, alpha=0.8, color=colors[i], edgecolor="white", zorder=data_set_index, lw=0)
                balancer_index += 1
            i += 1
        data_set_index += 1

    plt.legend(handles=data_set_labels + classifier_labels)
    sns.despine()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/classifier_dataset_plt_{0}.png".format(current_time), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    results_per_classifier = [('Default without balancer', [([0.29075294901609111, 0.25670862675721368, 0.5943164437320252, 0.21188034188034188, 0.9767525455837083, 0.023247454416291734, 0.7881196581196581, 0.0, 0.0003902439024390244, 0.0, 0.0, 0.0, 1.0494232655858058], 'AdaBoost', 'Default without balancer'), ([0.23563279133799758, 0.21752118681443772, 0.5841167780833895, 0.20384615384615384, 0.9643874023206254, 0.03561259767937485, 0.7961538461538462, 0.0941025641025641, 0.9226526166232535, 0.01712431920435709, 0.6506837606837607, 0.6506837606837607, 4.9051758789419155], 'Artificial neural network', 'Default without balancer'), ([0.35522989690952878, 0.34394719569158905, 0.7128674043175585, 0.5550997150997151, 0.8706350935354014, 0.1293649064645986, 0.44490028490028494, 0.5193732193732193, 0.8480620412029362, 0.11360975609756097, 0.41729344729344725, 0.41729344729344725, 0.016181466377283106], 'Bernoulli Naive Bayes', 'Default without balancer'), ([0.21961877275669592, 0.2176303428983252, 0.5993414762327858, 0.26746438746438744, 0.9312185650011839, 0.068781434998816, 0.7325356125356125, 0.2576923076923076, 0.8908572105138527, 0.06527871181624438, 0.6400569800569801, 0.6400569800569801, 0.0189000208283651], 'Decision Tree', 'Default without balancer'), ([0.18094112443913724, 0.11708439363399012, 0.5376981871760466, 0.08444444444444445, 0.9909519299076486, 0.009048070092351408, 0.9155555555555555, 0.009658119658119657, 0.9346360407293395, 0.0014591522614255269, 0.6565527065527066, 0.6565527065527066, 0.10037373331844161], 'Extreme Learning Machine', 'Default without balancer'), ([0.064427073210118674, 0.013827249821256004, 0.5277472483139071, 0.9662962962962963, 0.08919820033151787, 0.9108017996684822, 0.0337037037037037, 0.9655555555555555, 0.08559933696424342, 0.906619464835425, 0.03293447293447293, 0.03293447293447293, 0.00700998252228473], 'Gaussian Naive Bayes', 'Default without balancer'), ([0.1952485748472354, 0.14537859464425371, 0.5481347700457339, 0.11056980056980056, 0.985699739521667, 0.014300260478332938, 0.8894301994301996, 0.02997150997150997, 0.9390101823348329, 0.002528534217381009, 0.6955840455840456, 0.6955840455840456, 0.012967800554819543], 'K-nearest neighbours', 'Default without balancer'), ([0.29007436911929785, 0.25073707202477885, 0.590507805242354, 0.20153846153846153, 0.9794771489462468, 0.020522851053753254, 0.7984615384615384, 0.04849002849002849, 0.9387198673928487, 0.0032124082405872605, 0.6119658119658119, 0.6119658119658119, 0.02201588389242912], 'Logistic regression', 'Default without balancer'), ([0.20139239218675367, 0.18823638374303303, 0.5740484236034792, 0.18954415954415954, 0.958552687662799, 0.04144731233720104, 0.8104558404558405, 0.06512820512820512, 0.8844290788538954, 0.014690977977740941, 0.5751851851851851, 0.5751851851851851, 0.2915858536158336], 'Random forest', 'Default without balancer'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16040105168151322], 'SVM (RDF)', 'Default without balancer'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16190814992800645], 'SVM (linear)', 'Default without balancer'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16703322577042323], 'SVM (polynomial)', 'Default without balancer')], 'Default without balancer'), ('Default with balancer', [([0.2926201320005194, 0.25557303729122605, 0.704299430671678, 0.6357264957264958, 0.7728723656168601, 0.22712763438313996, 0.36427350427350424, 0.0, 0.0003902439024390244, 0.0, 0.0, 0.0, 1.1963364414888957], 'AdaBoost', 'Default with balancer'), ([0.18719912645439868, 0.18158829946559474, 0.6114975325058204, 0.3731054131054131, 0.8498896519062278, 0.1501103480937722, 0.6268945868945869, 0.2507407407407407, 0.7434785697371536, 0.06897750414397348, 0.479059829059829, 0.479059829059829, 7.018106460113681], 'Artificial neural network', 'Default with balancer'), ([0.35483029367069402, 0.34264239217682579, 0.7140925969777495, 0.5603703703703704, 0.8678148235851291, 0.13218517641487096, 0.43962962962962965, 0.5260683760683762, 0.846507695950746, 0.11633293866919255, 0.415042735042735, 0.415042735042735, 0.016853618852991126], 'Bernoulli Naive Bayes', 'Default with balancer'), ([0.1789584809028813, 0.14081196277624683, 0.6340663136952503, 0.5927350427350427, 0.6753975846554583, 0.32460241534454176, 0.40726495726495726, 0.5927350427350427, 0.5782282737390481, 0.31701160312574006, 0.3520797720797721, 0.3520797720797721, 0.005968896307945888], 'Decision Tree', 'Default with balancer'), ([0.24823058452821672, 0.20665443440003309, 0.6795903811241248, 0.6294871794871794, 0.7296935827610704, 0.2703064172389297, 0.3705128205128204, 0.3745299145299145, 0.35984797537295765, 0.09638692872365617, 0.14122507122507122, 0.14122507122507122, 0.041637532110396266], 'Extreme Learning Machine', 'Default with balancer'), ([0.17391062417073452, 0.11548560779441104, 0.6336880500420639, 0.7386609686609686, 0.5287151314231588, 0.4712848685768411, 0.2613390313390313, 0.728917378917379, 0.5121780724603362, 0.45951977267345495, 0.25689458689458694, 0.25689458689458694, 0.010046426124203301], 'Gaussian Naive Bayes', 'Default with balancer'), ([0.25730112314317516, 0.23542349001024826, 0.6704960651647841, 0.5365242165242164, 0.8044679138053515, 0.19553208619464837, 0.4634757834757834, 0.40139601139601133, 0.7162424816481175, 0.12587544399715841, 0.3657549857549857, 0.3657549857549857, 0.023681066412298522], 'K-nearest neighbours', 'Default with balancer'), ([0.29820020705993311, 0.26066411766793846, 0.7080185654059677, 0.6417094017094017, 0.7743277291025337, 0.22567227089746628, 0.3582905982905983, 0.44606837606837607, 0.49330475964953824, 0.09065735259294341, 0.1627065527065527, 0.1627065527065527, 0.05122896410578424], 'Logistic regression', 'Default with balancer'), ([0.21578651587401326, 0.17165727944821083, 0.6607234881833508, 0.6305982905982905, 0.6908486857684111, 0.3091513142315889, 0.36940170940170935, 0.42703703703703705, 0.4298749704001894, 0.1530305470044992, 0.15692307692307692, 0.15692307692307692, 0.2514452772436532], 'Random forest', 'Default with balancer'), ([0.3260278968347689, 0.31143507753583782, 0.7014173644126285, 0.5499145299145299, 0.8529201989107269, 0.14707980108927302, 0.45008547008547006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02086326027306293], 'SVM (RDF)', 'Default with balancer'), ([0.33774330296453819, 0.32227595921264329, 0.709127288967687, 0.5646438746438747, 0.8536107032914989, 0.14638929670850104, 0.4353561253561253, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8090625424491142], 'SVM (linear)', 'Default with balancer'), ([0.30353224300145498, 0.28041486280418287, 0.6985124218008424, 0.5772649572649573, 0.8197598863367274, 0.18024011366327256, 0.42273504273504275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.865370095901038], 'SVM (polynomial)', 'Default with balancer')], 'Default with balancer'), ('Optimal parameters', [([0.28033815302725351, 0.23595812419132262, 0.7014216159911163, 0.6609971509971511, 0.7418460809850818, 0.2581539190149183, 0.339002849002849, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6648202270593668], 'AdaBoost', 'Optimal parameters'), ([0.29685456769618174, 0.26011580801995132, 0.7066042625076487, 0.6372364672364673, 0.7759720577788302, 0.22402794222116978, 0.3627635327635327, 0.44980056980056976, 0.4997139474307365, 0.09143452521903858, 0.15817663817663816, 0.15817663817663816, 3.096958907748456], 'Artificial neural network', 'Optimal parameters'), ([0.30583424585933461, 0.27712701596863165, 0.7053344060102289, 0.6065811965811966, 0.8040876154392611, 0.1959123845607388, 0.39341880341880336, 0.5864387464387464, 0.7525327965901019, 0.16001515510300734, 0.3584330484330484, 0.3584330484330484, 0.011976189254374426], 'Bernoulli Naive Bayes', 'Optimal parameters'), ([0.24201450694305099, 0.21568310866791379, 0.6639576596214058, 0.5565811965811965, 0.7713341226616149, 0.22866587733838503, 0.44341880341880335, 0.39165242165242165, 0.435562869997632, 0.11420127871181622, 0.21068376068376066, 0.21068376068376066, 0.0040381068525323005], 'Decision Tree', 'Optimal parameters'), ([0.24823058452821672, 0.20665443440003309, 0.6795903811241248, 0.6294871794871794, 0.7296935827610704, 0.2703064172389297, 0.3705128205128204, 0.3745299145299145, 0.35984797537295765, 0.09638692872365617, 0.14122507122507122, 0.14122507122507122, 0.041446484685114214], 'Extreme Learning Machine', 'Optimal parameters'), ([0.17391062417073452, 0.11548560779441104, 0.6336880500420639, 0.7386609686609686, 0.5287151314231588, 0.4712848685768411, 0.2613390313390313, 0.728917378917379, 0.5121780724603362, 0.45951977267345495, 0.25689458689458694, 0.25689458689458694, 0.01020236718898646], 'Gaussian Naive Bayes', 'Optimal parameters'), ([0.21681850079573234, 0.17203754464391316, 0.6617256875083065, 0.6350142450142451, 0.6884371300023682, 0.31156286999763205, 0.36498575498575503, 0.3595156695156695, 0.12801941747572815, 0.08025668955718683, 0.03643874643874644, 0.03643874643874644, 0.0016164622568902054], 'K-nearest neighbours', 'Optimal parameters'), ([0.29771378769173962, 0.259592051286544, 0.7081704429615867, 0.6439601139601139, 0.7723807719630595, 0.22761922803694054, 0.356039886039886, 0.44980056980056987, 0.49339758465545824, 0.09162964717025811, 0.1627065527065527, 0.1627065527065527, 0.06241192469187621], 'Logistic regression', 'Optimal parameters'), ([0.27596839795514594, 0.24007738705982118, 0.6930308546401373, 0.6191452991452993, 0.766916410134975, 0.2330835898650249, 0.38085470085470086, 0.41393162393162386, 0.34672460336253846, 0.086666824532323, 0.10917378917378917, 0.10917378917378917, 0.3657594052143363], 'Random forest', 'Optimal parameters'), ([0.28164622892150804, 0.24561993802060264, 0.6964378167854369, 0.6236182336182337, 0.7692573999526402, 0.23074260004735966, 0.37638176638176635, 0.44170940170940176, 0.3838006156760597, 0.09776651669429316, 0.12618233618233618, 0.12618233618233618, 0.09085993980904306], 'SVM (RDF)', 'Optimal parameters'), ([0.29436573703910784, 0.26341340784619965, 0.7004665921864596, 0.6088034188034188, 0.7921297655695002, 0.20787023443049962, 0.39119658119658124, 0.4364102564102564, 0.48736206488278483, 0.08618044044518115, 0.15002849002849003, 0.15002849002849003, 7.5021912345787145], 'SVM (linear)', 'Optimal parameters'), ([0.32171580886726503, 0.3050508721582415, 0.7016786718775825, 0.558005698005698, 0.8453516457494672, 0.15464835425053278, 0.44199430199430195, 0.46638176638176637, 0.7188003788775751, 0.09571347383376747, 0.3195726495726495, 0.3195726495726495, 2.654576776674647], 'SVM (polynomial)', 'Optimal parameters')], 'Optimal parameters')]
    results_per_classifier2 = [('Default without balancer', [([1.075294901609111, 0.25670862675721368, 1.43164437320252, 0.21188034188034188, 0.9767525455837083, 0.023247454416291734, 0.7881196581196581, 0.0, 0.0003902439024390244, 0.0, 0.0, 0.0, 1.0494232655858058], 'AdaBoost', 'Default without balancer'), ([0.23563279133799758, 0.21752118681443772, 0.5841167780833895, 0.20384615384615384, 0.9643874023206254, 0.03561259767937485, 0.7961538461538462, 0.0941025641025641, 0.9226526166232535, 0.01712431920435709, 0.6506837606837607, 0.6506837606837607, 4.9051758789419155], 'Artificial neural network', 'Default without balancer'), ([0.35522989690952878, 0.34394719569158905, 0.7128674043175585, 0.5550997150997151, 0.8706350935354014, 0.1293649064645986, 0.44490028490028494, 0.5193732193732193, 0.8480620412029362, 0.11360975609756097, 0.41729344729344725, 0.41729344729344725, 0.016181466377283106], 'Bernoulli Naive Bayes', 'Default without balancer'), ([0.21961877275669592, 0.2176303428983252, 0.5993414762327858, 0.26746438746438744, 0.9312185650011839, 0.068781434998816, 0.7325356125356125, 0.2576923076923076, 0.8908572105138527, 0.06527871181624438, 0.6400569800569801, 0.6400569800569801, 0.0189000208283651], 'Decision Tree', 'Default without balancer'), ([0.18094112443913724, 0.11708439363399012, 0.5376981871760466, 0.08444444444444445, 0.9909519299076486, 0.009048070092351408, 0.9155555555555555, 0.009658119658119657, 0.9346360407293395, 0.0014591522614255269, 0.6565527065527066, 0.6565527065527066, 0.10037373331844161], 'Extreme Learning Machine', 'Default without balancer'), ([0.064427073210118674, 0.013827249821256004, 0.5277472483139071, 0.9662962962962963, 0.08919820033151787, 0.9108017996684822, 0.0337037037037037, 0.9655555555555555, 0.08559933696424342, 0.906619464835425, 0.03293447293447293, 0.03293447293447293, 0.00700998252228473], 'Gaussian Naive Bayes', 'Default without balancer'), ([0.1952485748472354, 0.14537859464425371, 0.5481347700457339, 0.11056980056980056, 0.985699739521667, 0.014300260478332938, 0.8894301994301996, 0.02997150997150997, 0.9390101823348329, 0.002528534217381009, 0.6955840455840456, 0.6955840455840456, 0.012967800554819543], 'K-nearest neighbours', 'Default without balancer'), ([0.29007436911929785, 0.25073707202477885, 0.590507805242354, 0.20153846153846153, 0.9794771489462468, 0.020522851053753254, 0.7984615384615384, 0.04849002849002849, 0.9387198673928487, 0.0032124082405872605, 0.6119658119658119, 0.6119658119658119, 0.02201588389242912], 'Logistic regression', 'Default without balancer'), ([0.20139239218675367, 0.18823638374303303, 0.5740484236034792, 0.18954415954415954, 0.958552687662799, 0.04144731233720104, 0.8104558404558405, 0.06512820512820512, 0.8844290788538954, 0.014690977977740941, 0.5751851851851851, 0.5751851851851851, 0.2915858536158336], 'Random forest', 'Default without balancer'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16040105168151322], 'SVM (RDF)', 'Default without balancer'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16190814992800645], 'SVM (linear)', 'Default without balancer'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16703322577042323], 'SVM (polynomial)', 'Default without balancer')], 'Default without balancer'), ('Default with balancer', [([0.2926201320005194, 0.25557303729122605, 0.704299430671678, 0.6357264957264958, 0.7728723656168601, 0.22712763438313996, 0.36427350427350424, 0.0, 0.0003902439024390244, 0.0, 0.0, 0.0, 1.1963364414888957], 'AdaBoost', 'Default with balancer'), ([0.18719912645439868, 0.18158829946559474, 0.6114975325058204, 0.3731054131054131, 0.8498896519062278, 0.1501103480937722, 0.6268945868945869, 0.2507407407407407, 0.7434785697371536, 0.06897750414397348, 0.479059829059829, 0.479059829059829, 7.018106460113681], 'Artificial neural network', 'Default with balancer'), ([0.35483029367069402, 0.34264239217682579, 0.7140925969777495, 0.5603703703703704, 0.8678148235851291, 0.13218517641487096, 0.43962962962962965, 0.5260683760683762, 0.846507695950746, 0.11633293866919255, 0.415042735042735, 0.415042735042735, 0.016853618852991126], 'Bernoulli Naive Bayes', 'Default with balancer'), ([0.1789584809028813, 0.14081196277624683, 0.6340663136952503, 0.5927350427350427, 0.6753975846554583, 0.32460241534454176, 0.40726495726495726, 0.5927350427350427, 0.5782282737390481, 0.31701160312574006, 0.3520797720797721, 0.3520797720797721, 0.005968896307945888], 'Decision Tree', 'Default with balancer'), ([0.24823058452821672, 0.20665443440003309, 0.6795903811241248, 0.6294871794871794, 0.7296935827610704, 0.2703064172389297, 0.3705128205128204, 0.3745299145299145, 0.35984797537295765, 0.09638692872365617, 0.14122507122507122, 0.14122507122507122, 0.041637532110396266], 'Extreme Learning Machine', 'Default with balancer'), ([0.17391062417073452, 0.11548560779441104, 0.6336880500420639, 0.7386609686609686, 0.5287151314231588, 0.4712848685768411, 0.2613390313390313, 0.728917378917379, 0.5121780724603362, 0.45951977267345495, 0.25689458689458694, 0.25689458689458694, 0.010046426124203301], 'Gaussian Naive Bayes', 'Default with balancer'), ([0.25730112314317516, 0.23542349001024826, 0.6704960651647841, 0.5365242165242164, 0.8044679138053515, 0.19553208619464837, 0.4634757834757834, 0.40139601139601133, 0.7162424816481175, 0.12587544399715841, 0.3657549857549857, 0.3657549857549857, 0.023681066412298522], 'K-nearest neighbours', 'Default with balancer'), ([0.29820020705993311, 0.26066411766793846, 0.7080185654059677, 0.6417094017094017, 0.7743277291025337, 0.22567227089746628, 0.3582905982905983, 0.44606837606837607, 0.49330475964953824, 0.09065735259294341, 0.1627065527065527, 0.1627065527065527, 0.05122896410578424], 'Logistic regression', 'Default with balancer'), ([0.21578651587401326, 0.17165727944821083, 0.6607234881833508, 0.6305982905982905, 0.6908486857684111, 0.3091513142315889, 0.36940170940170935, 0.42703703703703705, 0.4298749704001894, 0.1530305470044992, 0.15692307692307692, 0.15692307692307692, 0.2514452772436532], 'Random forest', 'Default with balancer'), ([0.3260278968347689, 0.31143507753583782, 0.7014173644126285, 0.5499145299145299, 0.8529201989107269, 0.14707980108927302, 0.45008547008547006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02086326027306293], 'SVM (RDF)', 'Default with balancer'), ([0.33774330296453819, 0.32227595921264329, 0.709127288967687, 0.5646438746438747, 0.8536107032914989, 0.14638929670850104, 0.4353561253561253, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8090625424491142], 'SVM (linear)', 'Default with balancer'), ([0.30353224300145498, 0.28041486280418287, 0.6985124218008424, 0.5772649572649573, 0.8197598863367274, 0.18024011366327256, 0.42273504273504275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.865370095901038], 'SVM (polynomial)', 'Default with balancer')], 'Default with balancer'), ('Optimal parameters', [([0.28033815302725351, 0.23595812419132262, 0.7014216159911163, 0.6609971509971511, 0.7418460809850818, 0.2581539190149183, 0.339002849002849, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6648202270593668], 'AdaBoost', 'Optimal parameters'), ([0.29685456769618174, 0.26011580801995132, 0.7066042625076487, 0.6372364672364673, 0.7759720577788302, 0.22402794222116978, 0.3627635327635327, 0.44980056980056976, 0.4997139474307365, 0.09143452521903858, 0.15817663817663816, 0.15817663817663816, 3.096958907748456], 'Artificial neural network', 'Optimal parameters'), ([0.30583424585933461, 0.27712701596863165, 0.7053344060102289, 0.6065811965811966, 0.8040876154392611, 0.1959123845607388, 0.39341880341880336, 0.5864387464387464, 0.7525327965901019, 0.16001515510300734, 0.3584330484330484, 0.3584330484330484, 0.011976189254374426], 'Bernoulli Naive Bayes', 'Optimal parameters'), ([0.24201450694305099, 0.21568310866791379, 0.6639576596214058, 0.5565811965811965, 0.7713341226616149, 0.22866587733838503, 0.44341880341880335, 0.39165242165242165, 0.435562869997632, 0.11420127871181622, 0.21068376068376066, 0.21068376068376066, 0.0040381068525323005], 'Decision Tree', 'Optimal parameters'), ([0.24823058452821672, 0.20665443440003309, 0.6795903811241248, 0.6294871794871794, 0.7296935827610704, 0.2703064172389297, 0.3705128205128204, 0.3745299145299145, 0.35984797537295765, 0.09638692872365617, 0.14122507122507122, 0.14122507122507122, 0.041446484685114214], 'Extreme Learning Machine', 'Optimal parameters'), ([0.17391062417073452, 0.11548560779441104, 0.6336880500420639, 0.7386609686609686, 0.5287151314231588, 0.4712848685768411, 0.2613390313390313, 0.728917378917379, 0.5121780724603362, 0.45951977267345495, 0.25689458689458694, 0.25689458689458694, 0.01020236718898646], 'Gaussian Naive Bayes', 'Optimal parameters'), ([0.21681850079573234, 0.17203754464391316, 0.6617256875083065, 0.6350142450142451, 0.6884371300023682, 0.31156286999763205, 0.36498575498575503, 0.3595156695156695, 0.12801941747572815, 0.08025668955718683, 0.03643874643874644, 0.03643874643874644, 0.0016164622568902054], 'K-nearest neighbours', 'Optimal parameters'), ([0.29771378769173962, 0.259592051286544, 0.7081704429615867, 0.6439601139601139, 0.7723807719630595, 0.22761922803694054, 0.356039886039886, 0.44980056980056987, 0.49339758465545824, 0.09162964717025811, 0.1627065527065527, 0.1627065527065527, 0.06241192469187621], 'Logistic regression', 'Optimal parameters'), ([0.27596839795514594, 0.24007738705982118, 0.6930308546401373, 0.6191452991452993, 0.766916410134975, 0.2330835898650249, 0.38085470085470086, 0.41393162393162386, 0.34672460336253846, 0.086666824532323, 0.10917378917378917, 0.10917378917378917, 0.3657594052143363], 'Random forest', 'Optimal parameters'), ([0.28164622892150804, 0.24561993802060264, 0.6964378167854369, 0.6236182336182337, 0.7692573999526402, 0.23074260004735966, 0.37638176638176635, 0.44170940170940176, 0.3838006156760597, 0.09776651669429316, 0.12618233618233618, 0.12618233618233618, 0.09085993980904306], 'SVM (RDF)', 'Optimal parameters'), ([0.29436573703910784, 0.26341340784619965, 0.7004665921864596, 0.6088034188034188, 0.7921297655695002, 0.20787023443049962, 0.39119658119658124, 0.4364102564102564, 0.48736206488278483, 0.08618044044518115, 0.15002849002849003, 0.15002849002849003, 7.5021912345787145], 'SVM (linear)', 'Optimal parameters'), ([0.32171580886726503, 0.3050508721582415, 0.7016786718775825, 0.558005698005698, 0.8453516457494672, 0.15464835425053278, 0.44199430199430195, 0.46638176638176637, 0.7188003788775751, 0.09571347383376747, 0.3195726495726495, 0.3195726495726495, 2.654576776674647], 'SVM (polynomial)', 'Optimal parameters')], 'Optimal parameters')]
    parameter = "Balanced Accuracy"
    x_label = "Parameter tuning approach"
    difference_from = "using default parameters"
    dataset = "Lima TB"
    name_suffix = ""
    patterns = (None, "////")
    no_feature_selection = results_per_classifier[0][1]
    color = iter(cm.Set1(np.linspace(0, 1, len(no_feature_selection) + 1)))
    classifier_arr = []
    for i in range(len(no_feature_selection) + 1):
        classifier_arr.append(list())
    for i in range(1, len(results_per_classifier)):
        data_balancer_results = results_per_classifier[i][1]
        x = 0
        mean_classification = 0
        for (result_arr, data_balancer_name, _) in data_balancer_results:
            value = result_arr[2] - no_feature_selection[x][0][2]
            classifier_arr[x].append(value)
            mean_classification += value
            x += 1
        mean_classification /= float(len(data_balancer_results))
        classifier_arr[x].append(mean_classification)

    no_feature_selection2 = results_per_classifier2[0][1]
    color = iter(cm.Set1(np.linspace(0, 1, len(no_feature_selection2) + 1)))
    classifier_arr2 = []
    for i in range(len(no_feature_selection2) + 1):
        classifier_arr2.append(list())
    for i in range(1, len(results_per_classifier2)):
        data_balancer_results2 = results_per_classifier2[i][1]
        x = 0
        mean_classification = 0
        for (result_arr2, data_balancer_name, _) in data_balancer_results2:
            value = result_arr2[2] - no_feature_selection2[x][0][2]
            classifier_arr2[x].append(value)
            mean_classification += value
            x += 1
        mean_classification /= float(len(data_balancer_results2))
        classifier_arr2[x].append(mean_classification)


    fig = plt.figure(figsize=(8, 5))
    st = fig.suptitle("{0} per {1} on {2} dataset".format(parameter, x_label, dataset))

    classifiers = np.arange(len(classifier_arr))
    data_balancers = np.arange(len(classifier_arr[0])) * 3
    bar_width = 0.2
    opacity = 0.9
    ax1 = plt.subplot(121)
    for i in range(len(classifier_arr)):
        if i + 1 != len(classifier_arr):
            label = results_per_classifier[0][1][i][1]
        else:
            label = "Mean classification"
        plt.bar(data_balancers + (i * bar_width), classifier_arr[i], bar_width,
                alpha=opacity,
                color=color.next(),
                hatch=patterns[i % len(patterns)],
                label=label)

    legend = plt.legend(loc='lower center', bbox_to_anchor=(1, -0.23), fancybox=True, frameon=True, ncol=4)
    legend.get_frame().set_facecolor('#ffffff')

    plt.xlabel(x_label)
    plt.ylabel("Difference in {0} from {1}".format(parameter, difference_from))
    feature_selection_labels = [results_per_classifier2[i][0] for i in range(1, len(results_per_classifier2))]
    plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)

    ax2 = plt.subplot(122, sharey=ax1)
    color = iter(cm.Set1(np.linspace(0, 1, len(no_feature_selection) + 1)))
    for i in range(len(classifier_arr2)):
        if i + 1 != len(classifier_arr2):
            label = results_per_classifier2[0][1][i][1]
        else:
            label = "Mean classification"
        plt.bar(data_balancers + (i * bar_width), classifier_arr2[i], bar_width,
                alpha=opacity,
                color=color.next(),
                hatch=patterns[i % len(patterns)],
                label=label)

    plt.xlabel(x_label)
    feature_selection_labels = [results_per_classifier2[i][0] for i in range(1, len(results_per_classifier2))]
    plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plt.locator_params(axis='y', nbins=15)
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/{3}_results_per_classifier_plot{0}_{4}_{1}_{2}.png".format(name_suffix, parameter, current_time, x_label, dataset), bbox_extra_artists=(legend, st), bbox_inches='tight')
    plt.close(fig)