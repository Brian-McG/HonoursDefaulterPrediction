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


def plot_percentage_difference_graph(results, datasets, name_suffix="", parameter="Balanced Accuracy", x_label="Feature selection approach", difference_from="no feature selection", figsize=(16, 5), legend_y=-0.31):
    print(results)
    patterns = (None, "////")

    colors = ["#64B3DE", "#1f78b4", "#B9B914", "#FBAC44", "#bc1659", "#33a02c", "#6ABF20", "#ff7f00", "#6a3d9a", "#5a2add", "#b15928", "#e31a1c", "grey"]
    classifier_arr = []
    for i in range(len(results)):
        classifier_arr.append(list())
    index = 0
    for results_per_classifier in results:
        no_feature_selection = results[index][0][1]
        for i in range(len(no_feature_selection) + 1):
            classifier_arr[index].append(list())
        for i in range(1, len(results_per_classifier)):
            data_balancer_results = results_per_classifier[i][1]
            x = 0
            mean_classification = 0
            for (result_arr, data_balancer_name, _) in data_balancer_results:
                value = result_arr[2] - no_feature_selection[x][0][2]
                classifier_arr[index][x].append(value)
                mean_classification += value
                x += 1
            mean_classification /= float(len(data_balancer_results))
            classifier_arr[index][x].append(mean_classification)
        index += 1

    fig = plt.figure(figsize=figsize)

    classifiers = np.arange(len(classifier_arr[0]))
    data_balancers = np.arange(len(classifier_arr[0][0])) * 3
    bar_width = 0.2
    opacity = 0.9
    subplt_val = 101 + (10 * len(results))
    ax1 = plt.subplot(subplt_val)
    for i in range(len(classifier_arr[0])):
        if i + 1 != len(classifier_arr[0]):
            label = results[0][0][1][i][1]
        else:
            label = "Mean classification"
        plt.bar(data_balancers + (i * bar_width), classifier_arr[0][i], bar_width,
                alpha=opacity,
                color=colors[i],
                hatch=patterns[i % len(patterns)],
                label=label)

    legend = plt.legend(loc='lower center', bbox_to_anchor=(len(results) / 2.0, legend_y), fancybox=True, frameon=True, ncol=6)
    legend.get_frame().set_facecolor('#ffffff')

    plt.xlabel(x_label, x=len(results) / 2.0)
    plt.ylabel("Difference in {0} from {1}".format(parameter, difference_from))
    feature_selection_labels = [results[0][i][0] for i in range(1, len(results[0]))]
    plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)
    plt.title(datasets[0].replace("_", " "))

    for z in range(1, len(results)):
        ax2 = plt.subplot(subplt_val + z, sharey=ax1)
        color = iter(cm.Set1(np.linspace(0, 1, len(no_feature_selection) + 1)))
        for i in range(len(classifier_arr[z])):
            if i + 1 != len(classifier_arr[z]):
                label = results[z][0][1][i][1]
            else:
                label = "Mean classification"
            plt.bar(data_balancers + (i * bar_width), classifier_arr[z][i], bar_width,
                    alpha=opacity,
                    color=colors[i],
                    hatch=patterns[i % len(patterns)],
                    label=label)

        feature_selection_labels = [results[z][i][0] for i in range(1, len(results[z]))]
        plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)
        plt.title(datasets[z].replace("_", " "))

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plt.locator_params(axis='y', nbins=15)
    name = "{3}_results_per_classifier_plot{0}_{4}_{1}_{2}".format(name_suffix, parameter, current_time, x_label, datasets)
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/{0}".format(name.replace(" ", "_")), bbox_extra_artists=(legend,), bbox_inches='tight')
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
    results = [[(None, [([0.28781600971411797, 0.247621520646717, 0.7036949050546019, 0.6473789173789173, 0.7600108927302865, 0.23998910726971348, 0.3526210826210826, 0.0, 0.0, 0.0, 0.0, 0.0, 0.692468989940823], 'AdaBoost', None), ([0.30233371176878665, 0.26558558520561892, 0.7101606586640923, 0.6415384615384615, 0.778782855789723, 0.22121714421027705, 0.35846153846153844, 0.4556695156695157, 0.4985219038598153, 0.09056831636277528, 0.16319088319088318, 0.16319088319088318, 3.682197062541732], 'Artificial neural network', None), ([0.32377610015569191, 0.29888055991244444, 0.7123407030958535, 0.6025071225071225, 0.8221742836845843, 0.17782571631541558, 0.3974928774928775, 0.5569800569800569, 0.7655571868340043, 0.13988586313047596, 0.35136752136752125, 0.35136752136752125, 0.005547791050215878], 'Bernoulli Naive Bayes', None), ([0.27095390804606867, 0.24260098371312608, 0.6841320681143082, 0.5803133903133904, 0.7879507459152262, 0.21204925408477382, 0.41968660968660965, 0.4014814814814815, 0.3949273028652617, 0.10629552450864313, 0.16743589743589746, 0.16743589743589746, 0.004506072815832776], 'Decision Tree', None), ([0.28997953029033657, 0.25772214219267087, 0.699203011860838, 0.6124216524216524, 0.7859843713000236, 0.2140156286999763, 0.3875783475783475, 0.40336182336182336, 0.47158891783092577, 0.06984418659720579, 0.1394301994301994, 0.1394301994301994, 0.2451939388963281], 'Extreme Learning Machine', None), ([0.19442643074152813, 0.13182614703330225, 0.6505513754888608, 0.7353561253561252, 0.5657466256215959, 0.434253374378404, 0.2646438746438746, 0.7264102564102564, 0.5499857920909306, 0.41850153919014915, 0.2557264957264957, 0.2557264957264957, 0.010191662873294583], 'Gaussian Naive Bayes', None), ([0.22649072673991627, 0.18023513576770905, 0.6687959795530244, 0.644985754985755, 0.6926062041202937, 0.3073937958797064, 0.355014245014245, 0.34415954415954414, 0.1297840397821454, 0.08005399005446366, 0.046296296296296294, 0.046296296296296294, 0.0017021573317544281], 'K-nearest neighbours', None), ([0.30104970362265604, 0.26360090683083742, 0.7099570645893165, 0.6437606837606837, 0.7761534454179492, 0.22384655458205072, 0.3562393162393162, 0.4601139601139601, 0.49746199384323936, 0.09144210277054227, 0.15498575498575498, 0.15498575498575498, 0.05868900469573224], 'Logistic regression', None), ([0.28360341656394555, 0.24688903354231595, 0.6985979040298245, 0.6287179487179487, 0.7684778593417001, 0.23152214065829982, 0.3712820512820513, 0.39977207977207974, 0.36605114847264986, 0.08200331517878286, 0.11623931623931623, 0.11623931623931623, 0.4158563340453211], 'Random forest', None), ([0.28018827883578729, 0.24255262504365235, 0.6971702614835459, 0.6309116809116809, 0.7634288420554107, 0.2365711579445892, 0.3690883190883191, 0.41082621082621074, 0.38116410134975137, 0.08743973478569737, 0.12663817663817667, 0.12663817663817667, 0.09380327071150799], 'SVM (RDF)', None), ([0.29802241408889329, 0.26888915713469935, 0.7014120941284095, 0.6035042735042735, 0.7993199147525456, 0.20068008524745445, 0.3964957264957264, 0.43689458689458693, 0.48998721288183755, 0.0855055647643855, 0.14017094017094017, 0.14017094017094017, 7.322054451888979], 'SVM (linear)', None), ([0.32473605608202932, 0.29984185850234873, 0.7128648811656152, 0.6031623931623932, 0.8225673691688373, 0.17743263083116267, 0.3968376068376068, 0.0, 0.9320014207909069, 0.0, 0.6190883190883191, 0.6190883190883191, 1.4603749823412007], 'SVM (polynomial)', None)], None), ('Logistic regression', [([0.29152300384615731, 0.24896984797854421, 0.7075749696580859, 0.6601994301994303, 0.7549505091167417, 0.2450494908832584, 0.33980056980056983, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7734601828787523], 'AdaBoost', 'Logistic regression'), ([0.28455727442665346, 0.23895085298231766, 0.7053454485104971, 0.669059829059829, 0.7416310679611651, 0.258368932038835, 0.3309401709401709, 0.47202279202279207, 0.4138579209093062, 0.09232346672981293, 0.1200854700854701, 0.1200854700854701, 3.3056676595842305], 'Artificial neural network', 'Logistic regression'), ([0.25797968180900621, 0.21757930216218954, 0.685193191807447, 0.6323361823361824, 0.7380502012787116, 0.2619497987212882, 0.3676638176638176, 0.5009116809116809, 0.4439341700213119, 0.123241297655695, 0.22658119658119658, 0.22658119658119658, 0.0048284289162396025], 'Bernoulli Naive Bayes', 'Logistic regression'), ([0.26565883353066683, 0.23406798584195684, 0.6829560910844358, 0.5926495726495726, 0.773262609519299, 0.22673739048070093, 0.4073504273504273, 0.43826210826210826, 0.41686194648354247, 0.11681695477148946, 0.17925925925925928, 0.17925925925925928, 0.004027655189151318], 'Decision Tree', 'Logistic regression'), ([0.29053360431332076, 0.25760638912113254, 0.7001236465887188, 0.6162108262108262, 0.7840364669666113, 0.21596353303338853, 0.38378917378917377, 0.4115384615384615, 0.45258441865972066, 0.06702628463177836, 0.1431908831908832, 0.1431908831908832, 0.2877832501608914], 'Extreme Learning Machine', 'Logistic regression'), ([0.066859565743264904, 0.010524588210984186, 0.5219264170635234, 0.9925925925925926, 0.05126024153445417, 0.9487397584655459, 0.007407407407407407, 0.9925925925925926, 0.05116268055884442, 0.9482543215723419, 0.007407407407407407, 0.007407407407407407, 0.008206723263369931], 'Gaussian Naive Bayes', 'Logistic regression'), ([0.042194565285298136, 0.019759730914440367, 0.5308626494579608, 0.7419943019943019, 0.31973099692161966, 0.6802690030783805, 0.258005698005698, 0.33897435897435896, 0.02568600520956666, 0.30128202699502726, 0.009715099715099715, 0.009715099715099715, 0.0016275979565271825], 'K-nearest neighbours', 'Logistic regression'), ([0.2866565505843483, 0.24115981498638281, 0.7066285515556177, 0.6698860398860399, 0.7433710632251954, 0.25662893677480464, 0.3301139601139601, 0.46527065527065525, 0.4130996921619702, 0.08911390007103955, 0.11561253561253562, 0.11561253561253562, 0.03780996475484765], 'Logistic regression', 'Logistic regression'), ([0.28929690455052121, 0.25381289318694311, 0.7009966429935646, 0.6266951566951567, 0.7752981292919724, 0.2247018707080275, 0.37330484330484326, 0.4248717948717948, 0.44320719867392844, 0.09309022022259057, 0.157977207977208, 0.157977207977208, 0.3596382632676443], 'Random forest', 'Logistic regression'), ([0.27357321079284502, 0.23645141646363474, 0.6924807285837359, 0.6234757834757834, 0.7614856736916884, 0.23851432630831165, 0.37652421652421647, 0.46703703703703703, 0.42790859578498697, 0.11070092351408951, 0.1688319088319088, 0.1688319088319088, 0.07245354075530155], 'SVM (RDF)', 'Logistic regression'), ([0.28130216018001997, 0.24345099120166536, 0.6976512518274299, 0.633133903133903, 0.7621686005209566, 0.2378313994790433, 0.3668660968660969, 0.46564102564102566, 0.26426000473596967, 0.10770636987923277, 0.08056980056980055, 0.08056980056980055, 3.941228617759742], 'SVM (linear)', 'Logistic regression'), ([0.29212020988952647, 0.25583945183089984, 0.7034679009872, 0.6325356125356125, 0.7744001894387875, 0.22559981056121242, 0.3674643874643875, 0.0, 0.9192611887283922, 0.0, 0.5517948717948717, 0.5517948717948717, 1.021208218073193], 'SVM (polynomial)', 'Logistic regression')], 'Logistic regression'), ('Bernoulli Naive Bayes', [([0.23721134260472171, 0.19531575012113805, 0.6733293340700397, 0.6277777777777778, 0.7188808903623016, 0.2811191096376983, 0.3722222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4713710714453887], 'AdaBoost', 'Bernoulli Naive Bayes'), ([0.2400425269873499, 0.20485913935182606, 0.6709678898556474, 0.5955270655270656, 0.7464087141842292, 0.25359128581577073, 0.4044729344729344, 0.32253561253561247, 0.3011650485436893, 0.07489841345015391, 0.12977207977207977, 0.12977207977207977, 4.578897191375507], 'Artificial neural network', 'Bernoulli Naive Bayes'), ([0.17526725782436287, 0.11917603595168855, 0.6365178533239153, 0.7012535612535611, 0.5717821453942695, 0.42821785460573053, 0.2987464387464388, 0.44772079772079765, 0.30312337201041917, 0.1783173099692162, 0.15002849002849003, 0.15002849002849003, 0.004538764234199633], 'Bernoulli Naive Bayes', 'Bernoulli Naive Bayes'), ([0.14288396299150713, 0.11734160362652543, 0.6026119905037736, 0.5058974358974359, 0.6993265451101113, 0.3006734548898887, 0.49410256410256415, 0.23569800569800564, 0.1651006393559081, 0.09092256689557186, 0.0964102564102564, 0.0964102564102564, 0.003938063194956924], 'Decision Tree', 'Bernoulli Naive Bayes'), ([0.23613556668262273, 0.2010198190250779, 0.6686090362571537, 0.5934472934472935, 0.743770779067014, 0.25622922093298606, 0.4065527065527065, 0.3001424501424501, 0.2881079801089273, 0.06400189438787593, 0.12683760683760684, 0.12683760683760684, 0.5480793565314618], 'Extreme Learning Machine', 'Bernoulli Naive Bayes'), ([0.11969024960878252, 0.06585198104222964, 0.5883500509015546, 0.7749572649572651, 0.4017428368458442, 0.5982571631541558, 0.225042735042735, 0.7571225071225072, 0.3536869524035046, 0.5684878048780488, 0.19515669515669518, 0.19515669515669518, 0.010530349589894784], 'Gaussian Naive Bayes', 'Bernoulli Naive Bayes'), ([0.098392660457273268, 0.054516092204689427, 0.5756185905025593, 0.7276923076923076, 0.42354487331281077, 0.5764551266871891, 0.2723076923076923, 0.2972934472934473, 0.04795927066066777, 0.2232521903859815, 0.009629629629629629, 0.009629629629629629, 0.0016050490280733244], 'K-nearest neighbours', 'Bernoulli Naive Bayes'), ([0.24054207074270675, 0.20524547362845036, 0.6713237784132882, 0.5962393162393163, 0.7464082405872601, 0.25359175941273976, 0.40376068376068375, 0.3060683760683761, 0.2923021548662088, 0.0705195358749704, 0.12011396011396011, 0.12011396011396011, 0.027467492716123088], 'Logistic regression', 'Bernoulli Naive Bayes'), ([0.20234409976149612, 0.17005915162263824, 0.6454803642783752, 0.565014245014245, 0.7259464835425053, 0.2740535164574947, 0.43498575498575487, 0.2417378917378917, 0.15759175941273976, 0.05475775515036705, 0.06418803418803419, 0.06418803418803419, 0.45655210885306535], 'Random forest', 'Bernoulli Naive Bayes'), ([0.20325187178666154, 0.17129815179135671, 0.6460819302517148, 0.5614529914529914, 0.730710869050438, 0.26928913094956186, 0.4385470085470085, 0.25564102564102564, 0.15374331044281317, 0.058928723656168594, 0.06056980056980057, 0.06056980056980057, 0.07398934314095948], 'SVM (RDF)', 'Bernoulli Naive Bayes'), ([0.2491018251959681, 0.22078212685650747, 0.6711936013136581, 0.565014245014245, 0.7773729576130713, 0.2226270423869287, 0.434985754985755, 0.29569800569800564, 0.2403168363722472, 0.06564764385507933, 0.09606837606837607, 0.09606837606837607, 5.57512696223975], 'SVM (linear)', 'Bernoulli Naive Bayes'), ([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.5122258660056347], 'SVM (polynomial)', 'Bernoulli Naive Bayes')], 'Bernoulli Naive Bayes'), ('SVM (linear)', [([0.29003119951921386, 0.24799192316553142, 0.7063875689565957, 0.6570370370370372, 0.7557381008761543, 0.2442618991238456, 0.3429629629629629, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4950936610645518], 'AdaBoost', 'SVM (linear)'), ([0.27462948056961495, 0.230352145805203, 0.6983439838680188, 0.6572079772079771, 0.7394799905280605, 0.2605200094719394, 0.3427920227920228, 0.4557264957264957, 0.3970547004499171, 0.09183376746388823, 0.12216524216524216, 0.12216524216524216, 5.185620092513999], 'Artificial neural network', 'SVM (linear)'), ([0.25709609043301984, 0.21630377631379819, 0.6848446224143595, 0.6337891737891738, 0.7359000710395452, 0.26409992896045464, 0.3662108262108262, 0.4851282051282052, 0.40024863840871416, 0.1160416765332702, 0.1952991452991453, 0.1952991452991453, 0.004536621029486554], 'Bernoulli Naive Bayes', 'SVM (linear)'), ([0.26843552167955032, 0.2378857716578146, 0.6845111366124863, 0.5899145299145298, 0.7791077433104426, 0.22089225668955717, 0.41008547008547, 0.4262108262108262, 0.45506843476201764, 0.11021027705422685, 0.19917378917378917, 0.19917378917378917, 0.0038574364373783697], 'Decision Tree', 'SVM (linear)'), ([0.294048124642104, 0.262187966981278, 0.7012769105286274, 0.6131623931623931, 0.7893914278948615, 0.21060857210513856, 0.3868376068376068, 0.40940170940170945, 0.47518588681032436, 0.07198484489699267, 0.15806267806267804, 0.15806267806267804, 0.7245781509033453], 'Extreme Learning Machine', 'SVM (linear)'), ([0.065017369671672542, 0.010772024680339465, 0.5222172319134195, 0.9837037037037037, 0.060730760123135205, 0.9392692398768648, 0.016296296296296295, 0.9837037037037037, 0.05682832109874496, 0.9389765569500357, 0.010370370370370367, 0.010370370370370367, 0.008494546600515917], 'Gaussian Naive Bayes', 'SVM (linear)'), ([0.033506634526789161, 0.015728012964292059, 0.524390438873271, 0.7366951566951566, 0.31208572105138527, 0.6879142789486148, 0.26330484330484333, 0.3445299145299146, 0.030533270187070805, 0.3402510063935591, 0.020854700854700852, 0.020854700854700852, 0.0015646299983448132], 'K-nearest neighbours', 'SVM (linear)'), ([0.2704536777351586, 0.22616180234172517, 0.6957497296381975, 0.6549287749287748, 0.7365706843476202, 0.2634293156523798, 0.34507122507122506, 0.4511680911680912, 0.4013431210040256, 0.09222211697845133, 0.12299145299145298, 0.12299145299145298, 0.0414146539660156], 'Logistic regression', 'SVM (linear)'), ([0.28133858566455838, 0.24844205239552583, 0.694311266548065, 0.6086324786324787, 0.7799900544636513, 0.22000994553634853, 0.3913675213675213, 0.4406267806267806, 0.4574975136159128, 0.09766706133080749, 0.17219373219373219, 0.17219373219373219, 0.42807610248261607], 'Random forest', 'SVM (linear)'), ([0.27675495651651599, 0.24118008624231918, 0.6933505609290597, 0.6188034188034187, 0.7678977030547004, 0.23210229694529955, 0.3811965811965812, 0.4642165242165242, 0.41533791143736687, 0.11399242244849632, 0.1726210826210826, 0.1726210826210826, 0.0682522859844075], 'SVM (RDF)', 'SVM (linear)'), ([0.27764192531831638, 0.24178554189788937, 0.6944039019802696, 0.6213960113960114, 0.7674117925645276, 0.23258820743547243, 0.3786039886039886, 0.47960113960113954, 0.24131044281316605, 0.11468434762017524, 0.07316239316239316, 0.07316239316239316, 5.095171661641706], 'SVM (linear)', 'SVM (linear)'), ([0.29488676349874376, 0.26544254687120106, 0.6996334285249748, 0.6019943019943019, 0.7972725550556476, 0.20272744494435235, 0.39800569800569796, 0.0007407407407407407, 0.9194544162917356, 9.708737864077669e-05, 0.5592592592592592, 0.5592592592592592, 1.3392814408131368], 'SVM (polynomial)', 'SVM (linear)')], 'SVM (linear)'), ('Random forest', [([0.29310986359251967, 0.24941805286919685, 0.7093141209480305, 0.666096866096866, 0.7525313757991949, 0.24746862420080515, 0.33390313390313386, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5981820492417885], 'AdaBoost', 'Random forest'), ([0.30504389831994128, 0.26356819187873853, 0.7150903450309086, 0.6637321937321937, 0.7664484963296234, 0.2335515036703765, 0.3362678062678062, 0.46071225071225064, 0.4174672034098982, 0.08988112716078617, 0.12230769230769231, 0.12230769230769231, 3.7423512947975075], 'Artificial neural network', 'Random forest'), ([0.32505597992044361, 0.30777370094616563, 0.7045572036999933, 0.5645584045584046, 0.8445560028415816, 0.1554439971584182, 0.4354415954415954, 0.49313390313390304, 0.735288657352593, 0.10592896045465308, 0.3424501424501424, 0.3424501424501424, 0.004561464092567746], 'Bernoulli Naive Bayes', 'Random forest'), ([0.25761975680478633, 0.22925990377019981, 0.6758374887756844, 0.5717663817663817, 0.7799085957849868, 0.220091404215013, 0.4282336182336183, 0.4316809116809116, 0.46333601704949084, 0.11402415344541797, 0.21638176638176637, 0.21638176638176637, 0.003878415694778865], 'Decision Tree', 'Random forest'), ([0.30400877812601118, 0.27228112870017079, 0.7070926023748662, 0.6198290598290599, 0.7943561449206724, 0.2056438550793275, 0.38017094017094016, 0.4190883190883191, 0.4619417475728155, 0.06916504854368931, 0.13726495726495727, 0.13726495726495727, 0.10715724723164288], 'Extreme Learning Machine', 'Random forest'), ([0.27265623351298257, 0.2271453126942003, 0.6899151411379686, 0.691025641025641, 0.688804641250296, 0.311195358749704, 0.30897435897435893, 0.6641880341880342, 0.6698299786881364, 0.28929765569500354, 0.29185185185185186, 0.29185185185185186, 0.008609706121354676], 'Gaussian Naive Bayes', 'Random forest'), ([0.16960254818786091, 0.13550220818795378, 0.6253631787126933, 0.577037037037037, 0.6736893203883495, 0.3263106796116505, 0.42296296296296293, 0.22478632478632474, 0.05670850106559318, 0.052335780251006395, 0.03800569800569801, 0.03800569800569801, 0.0016166706310935196], 'K-nearest neighbours', 'Random forest'), ([0.30548401798818203, 0.26286200949334393, 0.716255738990051, 0.669088319088319, 0.763423158891783, 0.23657684110821692, 0.3309116809116809, 0.4615954415954415, 0.43227847501775984, 0.08969168837319443, 0.12603988603988606, 0.12603988603988606, 0.0361430042779579], 'Logistic regression', 'Random forest'), ([0.2858122962794753, 0.25080339098650473, 0.6986967650358605, 0.6220797720797722, 0.7753137579919488, 0.22468624200805115, 0.37792022792022795, 0.4223931623931624, 0.44715462941037176, 0.09446128344778593, 0.1676638176638177, 0.1676638176638177, 0.3601089834689491], 'Random forest', 'Random forest'), ([0.27854225646628389, 0.23943409353948217, 0.6973996652438518, 0.6363247863247863, 0.7584745441629173, 0.24152545583708265, 0.36367521367521366, 0.448945868945869, 0.41293535401373427, 0.10961022969452996, 0.14658119658119656, 0.14658119658119656, 0.06957406975859651], 'SVM (RDF)', 'Random forest'), ([0.31172057712470436, 0.28410608883231386, 0.7079206077422984, 0.6064957264957264, 0.8093454889888705, 0.19065451101112948, 0.3935042735042735, 0.4787464387464387, 0.29852758702344306, 0.10340989817665167, 0.09783475783475784, 0.09783475783475784, 5.423735720521277], 'SVM (linear)', 'Random forest'), ([0.30759788286515688, 0.27384547625101285, 0.7106011780555943, 0.6315954415954415, 0.789606914515747, 0.2103930854842529, 0.3684045584045584, 0.0, 0.9212095666587732, 0.0, 0.5722222222222222, 0.5722222222222222, 0.9403098772155], 'SVM (polynomial)', 'Random forest')], 'Random forest')], [(None, [([0.39529206777162551, 0.3846873302229315, 0.7117380952380953, 0.7023333333333333, 0.7211428571428572, 0.27885714285714286, 0.29766666666666663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.26563977981137377], 'AdaBoost', None), ([0.4027157252817023, 0.38435392041455929, 0.7181904761904762, 0.7476666666666667, 0.6887142857142857, 0.3112857142857143, 0.25233333333333335, 0.5760000000000001, 0.5107142857142856, 0.189, 0.13433333333333333, 0.13433333333333333, 5.2049931538193075], 'Artificial neural network', None), ([0.39638088563136087, 0.37895607908683326, 0.7146428571428571, 0.74, 0.6892857142857143, 0.3107142857142857, 0.26, 0.5943333333333334, 0.558, 0.19985714285714287, 0.15966666666666668, 0.15966666666666668, 0.03180573147275005], 'Bernoulli Naive Bayes', None), ([0.26160080907839911, 0.24273247891755587, 0.6423571428571428, 0.682, 0.6027142857142858, 0.3972857142857143, 0.318, 0.354, 0.4362857142857143, 0.15628571428571428, 0.1893333333333333, 0.1893333333333333, 0.008258462036298084], 'Decision Tree', None), ([0.40274300645791439, 0.39108720727420143, 0.7161666666666666, 0.7133333333333334, 0.719, 0.28099999999999997, 0.2866666666666667, 0.33599999999999997, 0.4272857142857143, 0.08099999999999999, 0.08533333333333334, 0.08533333333333334, 0.13193704049389124], 'Extreme Learning Machine', None), ([0.34667516895915479, 0.311590445690344, 0.688904761904762, 0.7876666666666666, 0.5901428571428571, 0.40985714285714286, 0.21233333333333332, 0.7506666666666668, 0.5609999999999999, 0.37785714285714284, 0.18166666666666667, 0.18166666666666667, 0.006835555222838252], 'Gaussian Naive Bayes', None), ([0.3720613565724204, 0.35690888679444804, 0.7011666666666667, 0.7153333333333334, 0.687, 0.31299999999999994, 0.2846666666666667, 0.36366666666666664, 0.3745714285714286, 0.10042857142857145, 0.06866666666666667, 0.06866666666666667, 0.00576875243752184], 'K-nearest neighbours', None), ([0.40100087410863977, 0.38196396709549263, 0.717404761904762, 0.7496666666666667, 0.6851428571428572, 0.31485714285714284, 0.25033333333333335, 0.5943333333333334, 0.5275714285714286, 0.1962857142857143, 0.14466666666666667, 0.14466666666666667, 0.020166741326661346], 'Logistic regression', None), ([0.37743611516145181, 0.37046245971816433, 0.700547619047619, 0.6676666666666666, 0.7334285714285713, 0.26657142857142857, 0.3323333333333333, 0.35733333333333334, 0.4822857142857144, 0.07528571428571429, 0.11266666666666666, 0.11266666666666666, 2.228396647424105], 'Random forest', None), ([0.39823421202999654, 0.38358102137959377, 0.7148809523809524, 0.7273333333333334, 0.7024285714285715, 0.29757142857142854, 0.2726666666666666, 0.23366666666666663, 0.6768571428571429, 0.04114285714285714, 0.24466666666666667, 0.24466666666666667, 1.415874234332567], 'SVM (RDF)', None), ([0.38623194065391092, 0.37248891492453662, 0.7082619047619048, 0.7156666666666667, 0.7008571428571428, 0.2991428571428572, 0.2843333333333333, 0.14933333333333335, 0.6627142857142856, 0.022571428571428572, 0.242, 0.242, 1.351895740033494], 'SVM (linear)', None), ([0.39124000524490454, 0.37295845815266693, 0.7120714285714286, 0.7409999999999999, 0.6831428571428572, 0.31685714285714284, 0.25899999999999995, 0.532, 0.5478571428571428, 0.1662857142857143, 0.15899999999999997, 0.15899999999999997, 0.7391651040390054], 'SVM (polynomial)', None)], None), ('Logistic regression', [([0.40176730747867662, 0.39165557190082001, 0.7149761904761905, 0.7036666666666667, 0.7262857142857142, 0.2737142857142857, 0.29633333333333334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22081498528735238], 'AdaBoost', 'Logistic regression'), ([0.38068991131777447, 0.35725315400225166, 0.7072380952380952, 0.7583333333333334, 0.6561428571428571, 0.34385714285714286, 0.24166666666666664, 0.6083333333333333, 0.5062857142857141, 0.20257142857142857, 0.134, 0.134, 4.470596921754314], 'Artificial neural network', 'Logistic regression'), ([0.38270139406145987, 0.3620045746950119, 0.7079761904761905, 0.7476666666666667, 0.6682857142857143, 0.33171428571428574, 0.2523333333333333, 0.5623333333333334, 0.5114285714285713, 0.18728571428571433, 0.14866666666666667, 0.14866666666666667, 0.023821841127556455], 'Bernoulli Naive Bayes', 'Logistic regression'), ([0.30499894633438063, 0.28124038667486051, 0.6661428571428571, 0.724, 0.6082857142857143, 0.3917142857142857, 0.27599999999999997, 0.42700000000000005, 0.4465714285714286, 0.17485714285714288, 0.15333333333333332, 0.15333333333333332, 0.006840717025738949], 'Decision Tree', 'Logistic regression'), ([0.38816607040265205, 0.37786889746270425, 0.7078809523809524, 0.6963333333333332, 0.7194285714285714, 0.2805714285714286, 0.30366666666666664, 0.36, 0.42028571428571426, 0.08057142857142859, 0.089, 0.089, 0.05371034014771756], 'Extreme Learning Machine', 'Logistic regression'), ([0.32331944870357798, 0.28660409810848775, 0.6758333333333334, 0.7846666666666666, 0.567, 0.433, 0.21533333333333332, 0.7433333333333334, 0.5328571428571429, 0.39828571428571424, 0.18266666666666667, 0.18266666666666667, 0.005641277036072978], 'Gaussian Naive Bayes', 'Logistic regression'), ([0.36697050714837276, 0.33963330146190945, 0.7001190476190475, 0.7676666666666666, 0.6325714285714286, 0.3674285714285713, 0.23233333333333334, 0.5966666666666667, 0.43814285714285717, 0.1861428571428571, 0.09866666666666667, 0.09866666666666667, 0.004211940608474763], 'K-nearest neighbours', 'Logistic regression'), ([0.39125329594228553, 0.36669872745181065, 0.7130238095238095, 0.7683333333333333, 0.6577142857142857, 0.3422857142857143, 0.2316666666666667, 0.6203333333333333, 0.5107142857142856, 0.213, 0.13033333333333333, 0.13033333333333333, 0.01191793880456255], 'Logistic regression', 'Logistic regression'), ([0.38003101813106183, 0.36510602818200849, 0.7053095238095237, 0.7183333333333334, 0.6922857142857143, 0.3077142857142857, 0.2816666666666666, 0.4986666666666667, 0.5035714285714284, 0.14614285714285716, 0.14, 0.14, 2.0114453772583643], 'Random forest', 'Logistic regression'), ([0.38739657017848345, 0.36882069609953894, 0.7100952380952382, 0.7403333333333333, 0.6798571428571429, 0.3201428571428572, 0.2596666666666666, 0.23899999999999996, 0.6639999999999999, 0.04042857142857143, 0.2483333333333333, 0.2483333333333333, 0.8811407342921196], 'SVM (RDF)', 'Logistic regression'), ([0.38277545743787073, 0.36774586388403563, 0.706738095238095, 0.7203333333333333, 0.6931428571428572, 0.30685714285714283, 0.2796666666666667, 0.1906666666666667, 0.67, 0.033285714285714293, 0.2523333333333333, 0.2523333333333333, 0.7108298850880046], 'SVM (linear)', 'Logistic regression'), ([0.36287886723674811, 0.33847957683318242, 0.6975238095238095, 0.7523333333333334, 0.6427142857142858, 0.35728571428571426, 0.24766666666666665, 0.5546666666666666, 0.5054285714285714, 0.18314285714285714, 0.13833333333333334, 0.13833333333333334, 0.5264097759113453], 'SVM (polynomial)', 'Logistic regression')], 'Logistic regression'), ('Bernoulli Naive Bayes', [([0.34587090810544074, 0.33570831492141695, 0.6856666666666668, 0.6713333333333333, 0.7, 0.29999999999999993, 0.3286666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22835218347281327], 'AdaBoost', 'Bernoulli Naive Bayes'), ([0.34978514732300647, 0.31640210695355386, 0.6906904761904762, 0.7826666666666666, 0.5987142857142858, 0.40128571428571425, 0.21733333333333338, 0.5693333333333335, 0.45114285714285707, 0.24900000000000003, 0.11433333333333336, 0.11433333333333336, 6.48836372659228], 'Artificial neural network', 'Bernoulli Naive Bayes'), ([0.35041980010810525, 0.3309181278872661, 0.6904761904761905, 0.7256666666666668, 0.6552857142857144, 0.3447142857142857, 0.2743333333333333, 0.37466666666666665, 0.41400000000000003, 0.13442857142857143, 0.10899999999999999, 0.10899999999999999, 0.011714726774604856], 'Bernoulli Naive Bayes', 'Bernoulli Naive Bayes'), ([0.29356026609273839, 0.26603107515962843, 0.6591190476190476, 0.7366666666666666, 0.5815714285714286, 0.4184285714285714, 0.2633333333333333, 0.262, 0.3897142857142857, 0.11414285714285714, 0.131, 0.131, 0.006783363660182511], 'Decision Tree', 'Bernoulli Naive Bayes'), ([0.33136862269113859, 0.32553195175182825, 0.6757857142857142, 0.6300000000000001, 0.7215714285714285, 0.27842857142857147, 0.37, 0.1333333333333333, 0.3095714285714286, 0.06542857142857143, 0.07233333333333333, 0.07233333333333333, 0.30997685942629805], 'Extreme Learning Machine', 'Bernoulli Naive Bayes'), ([0.23797994764729907, 0.22775598929723376, 0.628547619047619, 0.6146666666666667, 0.6424285714285713, 0.35757142857142854, 0.3853333333333333, 0.5673333333333332, 0.61, 0.33128571428571424, 0.33366666666666667, 0.33366666666666667, 0.005535354425684957], 'Gaussian Naive Bayes', 'Bernoulli Naive Bayes'), ([0.32588052855742489, 0.26677039310693917, 0.6731190476190475, 0.8556666666666667, 0.49057142857142855, 0.5094285714285713, 0.14433333333333334, 0.7193333333333334, 0.32399999999999995, 0.3857142857142858, 0.06266666666666666, 0.06266666666666666, 0.005841138422047009], 'K-nearest neighbours', 'Bernoulli Naive Bayes'), ([0.35931301101175805, 0.32568244881398722, 0.6959047619047619, 0.7876666666666667, 0.6041428571428573, 0.39585714285714285, 0.21233333333333332, 0.5916666666666667, 0.45257142857142857, 0.24700000000000003, 0.11600000000000002, 0.11600000000000002, 0.007977822962838315], 'Logistic regression', 'Bernoulli Naive Bayes'), ([0.30774973797638783, 0.28010534846541651, 0.6678333333333333, 0.7436666666666667, 0.592, 0.40800000000000003, 0.25633333333333336, 0.6183333333333334, 0.44699999999999995, 0.288, 0.15266666666666667, 0.15266666666666667, 2.155311827229931], 'Random forest', 'Bernoulli Naive Bayes'), ([0.32682061140182733, 0.2774637870410363, 0.6759047619047618, 0.8276666666666666, 0.5241428571428572, 0.4758571428571428, 0.17233333333333337, 0.0003333333333333333, 0.5529999999999999, 0.00014285714285714287, 0.18633333333333335, 0.18633333333333335, 1.1990318450034323], 'SVM (RDF)', 'Bernoulli Naive Bayes'), ([0.33741404273109971, 0.31125948381554414, 0.6840238095238097, 0.7493333333333333, 0.6187142857142857, 0.38128571428571423, 0.2506666666666667, 0.006333333333333334, 0.6214285714285714, 0.003, 0.252, 0.252, 0.9030409961856994], 'SVM (linear)', 'Bernoulli Naive Bayes'), ([0.047777290975883055, 0.031159559307002184, 0.5210952380952381, 0.7523333333333333, 0.28985714285714287, 0.7101428571428572, 0.24766666666666665, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5741643312778699], 'SVM (polynomial)', 'Bernoulli Naive Bayes')], 'Bernoulli Naive Bayes'), ('SVM (linear)', [([0.38338501069826941, 0.37254465213961729, 0.7056666666666667, 0.6973333333333332, 0.714, 0.2859999999999999, 0.30266666666666664, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31342485320556995], 'AdaBoost', 'SVM (linear)'), ([0.36826631031015655, 0.34172594801931477, 0.7007857142857142, 0.7650000000000001, 0.6365714285714286, 0.36342857142857143, 0.23500000000000001, 0.6133333333333334, 0.5042857142857142, 0.21514285714285714, 0.13933333333333336, 0.13933333333333336, 6.9034140345496695], 'Artificial neural network', 'SVM (linear)'), ([0.38743750589285242, 0.36467320745550336, 0.7107857142857144, 0.7590000000000001, 0.6625714285714286, 0.3374285714285714, 0.24099999999999996, 0.5443333333333333, 0.4944285714285714, 0.18385714285714289, 0.13466666666666666, 0.13466666666666666, 0.013071948891508357], 'Bernoulli Naive Bayes', 'SVM (linear)'), ([0.29659764012436884, 0.27775250473560925, 0.6610714285714285, 0.6940000000000001, 0.6281428571428573, 0.3718571428571429, 0.306, 0.385, 0.4255714285714286, 0.1587142857142857, 0.14533333333333331, 0.14533333333333331, 0.007090385281796269], 'Decision Tree', 'SVM (linear)'), ([0.38301328357261211, 0.37188229348818946, 0.705547619047619, 0.6986666666666668, 0.7124285714285714, 0.28757142857142853, 0.30133333333333334, 0.3436666666666667, 0.40828571428571425, 0.08642857142857141, 0.08700000000000001, 0.08700000000000001, 0.26264333813491525], 'Extreme Learning Machine', 'SVM (linear)'), ([0.32442477156189581, 0.29015765986871717, 0.676404761904762, 0.7756666666666667, 0.5771428571428572, 0.4228571428571429, 0.22433333333333336, 0.732, 0.539, 0.38842857142857146, 0.19233333333333333, 0.19233333333333333, 0.005686978612624927], 'Gaussian Naive Bayes', 'SVM (linear)'), ([0.35053920750406525, 0.32441737242014901, 0.6911428571428571, 0.7550000000000001, 0.6272857142857143, 0.3727142857142857, 0.24500000000000002, 0.5936666666666668, 0.469, 0.20414285714285713, 0.13033333333333333, 0.13033333333333333, 0.004391818874443487], 'K-nearest neighbours', 'SVM (linear)'), ([0.37724759435988076, 0.35114396909467777, 0.7056190476190476, 0.7676666666666666, 0.6435714285714285, 0.35642857142857143, 0.23233333333333334, 0.6196666666666667, 0.5065714285714286, 0.22071428571428572, 0.13666666666666666, 0.13666666666666666, 0.010961103551201137], 'Logistic regression', 'SVM (linear)'), ([0.34776346135035147, 0.33144861027160821, 0.6884523809523809, 0.7083333333333333, 0.6685714285714285, 0.3314285714285714, 0.29166666666666663, 0.515, 0.5005714285714286, 0.17557142857142854, 0.15433333333333332, 0.15433333333333332, 2.1543455135782565], 'Random forest', 'SVM (linear)'), ([0.36675537023082627, 0.34308059836393434, 0.6997619047619048, 0.7526666666666666, 0.6468571428571429, 0.3531428571428571, 0.24733333333333332, 0.192, 0.641, 0.03371428571428571, 0.23833333333333334, 0.23833333333333334, 0.9731276541024536], 'SVM (RDF)', 'SVM (linear)'), ([0.37740855909327292, 0.36011657194337687, 0.7044999999999999, 0.7289999999999999, 0.6799999999999999, 0.32, 0.27099999999999996, 0.18266666666666667, 0.6595714285714285, 0.034428571428571426, 0.24933333333333335, 0.24933333333333335, 0.7698861324391855], 'SVM (linear)', 'SVM (linear)'), ([0.35109270982921215, 0.33149793905014169, 0.690595238095238, 0.7253333333333333, 0.6558571428571429, 0.3441428571428572, 0.27466666666666667, 0.537, 0.48500000000000004, 0.19585714285714287, 0.13533333333333333, 0.13533333333333333, 0.5742670843601593], 'SVM (polynomial)', 'SVM (linear)')], 'SVM (linear)'), ('Random forest', [([0.39574723333962697, 0.38501959059025087, 0.7120238095238094, 0.7033333333333334, 0.7207142857142859, 0.2792857142857143, 0.29666666666666663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23849053790212243], 'AdaBoost', 'Random forest'), ([0.38529099594401617, 0.36510193503467792, 0.7092619047619046, 0.7466666666666667, 0.6718571428571429, 0.3281428571428571, 0.2533333333333333, 0.5596666666666665, 0.4852857142857142, 0.1831428571428571, 0.129, 0.129, 5.505032576711635], 'Artificial neural network', 'Random forest'), ([0.36334540053424402, 0.33955483754727095, 0.697952380952381, 0.7513333333333333, 0.6445714285714286, 0.3554285714285714, 0.24866666666666667, 0.5653333333333334, 0.5067142857142858, 0.20142857142857146, 0.13266666666666665, 0.13266666666666665, 0.01629508729182092], 'Bernoulli Naive Bayes', 'Random forest'), ([0.26405361751608736, 0.24947369141417131, 0.6434047619047619, 0.6586666666666666, 0.628142857142857, 0.37185714285714283, 0.3413333333333334, 0.36233333333333334, 0.43900000000000006, 0.16657142857142856, 0.195, 0.195, 0.007627484457240904], 'Decision Tree', 'Random forest'), ([0.37752761246244104, 0.36267926198169825, 0.703952380952381, 0.7163333333333333, 0.6915714285714285, 0.30842857142857144, 0.2836666666666666, 0.33699999999999997, 0.39699999999999996, 0.08185714285714285, 0.07866666666666668, 0.07866666666666668, 0.17533944136614638], 'Extreme Learning Machine', 'Random forest'), ([0.38390116780368128, 0.36407653100540815, 0.7084285714285714, 0.744, 0.6728571428571428, 0.3271428571428571, 0.256, 0.6603333333333332, 0.6144285714285714, 0.2757142857142857, 0.19666666666666666, 0.19666666666666666, 0.0053904013407453984], 'Gaussian Naive Bayes', 'Random forest'), ([0.37343305047200193, 0.35434720082886184, 0.7027380952380953, 0.7353333333333334, 0.6701428571428572, 0.32985714285714285, 0.2646666666666667, 0.43566666666666665, 0.3648571428571429, 0.11457142857142859, 0.07466666666666666, 0.07466666666666666, 0.004194915714713687], 'K-nearest neighbours', 'Random forest'), ([0.38765377900078213, 0.36655978438495135, 0.7106428571428571, 0.752, 0.6692857142857143, 0.3307142857142856, 0.248, 0.5783333333333334, 0.5018571428571429, 0.2015714285714286, 0.13033333333333333, 0.13033333333333333, 0.016490813198280063], 'Logistic regression', 'Random forest'), ([0.36278355459007322, 0.35418408417607578, 0.6938333333333333, 0.6706666666666667, 0.7170000000000001, 0.28300000000000003, 0.3293333333333333, 0.35866666666666663, 0.47928571428571426, 0.09085714285714284, 0.12133333333333333, 0.12133333333333333, 1.9156357348898012], 'Random forest', 'Random forest'), ([0.38423335291510785, 0.36791188260864233, 0.7079285714285715, 0.728, 0.6878571428571427, 0.31214285714285717, 0.27199999999999996, 0.20833333333333334, 0.6592857142857143, 0.039285714285714285, 0.248, 0.248, 0.9298349490883208], 'SVM (RDF)', 'Random forest'), ([0.38343315722927895, 0.36570101641190311, 0.7078095238095238, 0.7343333333333333, 0.6812857142857143, 0.31871428571428567, 0.26566666666666666, 0.146, 0.6497142857142857, 0.023857142857142855, 0.23500000000000001, 0.23500000000000001, 0.8546450188903876], 'SVM (linear)', 'Random forest'), ([0.36824953799372573, 0.3427360870331505, 0.7006904761904762, 0.7606666666666666, 0.6407142857142857, 0.3592857142857143, 0.23933333333333331, 0.4956666666666667, 0.5198571428571428, 0.16157142857142853, 0.15599999999999997, 0.15599999999999997, 0.4776473106101358], 'SVM (polynomial)', 'Random forest')], 'Random forest')], [(None, [([0.73863928505828969, 0.73603839673556981, 0.8714239076490322, 0.8989687995769435, 0.8438790157211209, 0.15612098427887905, 0.1010312004230566, 0.6243997884717081, 0.6480587833219412, 0.04620642515379357, 0.06841353781068218, 0.06841353781068218, 0.14839513208781904], 'AdaBoost', None), ([0.71744305428862332, 0.71654276723243615, 0.8601032989666126, 0.8674246430460073, 0.8527819548872181, 0.14721804511278197, 0.13257535695399258, 0.7510576414595451, 0.8018660287081338, 0.07148667122351333, 0.08146483342147012, 0.08146483342147012, 7.90301554940298], 'Artificial neural network', None), ([0.72455355970922897, 0.72280179202111439, 0.864168317348826, 0.8836647276573242, 0.8446719070403281, 0.1553280929596719, 0.11633527234267582, 0.8364569011105235, 0.8088790157211211, 0.12659261790840737, 0.09057112638815443, 0.09057112638815443, 0.01569613703952797], 'Bernoulli Naive Bayes', None), ([0.68033894359309577, 0.67530845245138504, 0.8420342284006734, 0.8830089899524062, 0.8010594668489406, 0.19894053315105947, 0.11699101004759387, 0.8537440507667899, 0.7354955570745045, 0.17229323308270675, 0.08536224219989423, 0.08536224219989423, 0.0057552894896034], 'Decision Tree', None), ([0.72332866633037729, 0.71291324183120774, 0.8632177711236411, 0.9331887890005289, 0.7932467532467533, 0.20675324675324674, 0.06681121099947118, 0.8002908514013749, 0.7747231715652768, 0.11254613807245387, 0.05831835007932311, 0.05831835007932311, 0.048845929541088975], 'Extreme Learning Machine', None), ([0.70029918327396978, 0.68882027318298489, 0.8513701842703485, 0.9256795346377578, 0.7770608339029391, 0.22293916609706085, 0.07432046536224221, 0.9256795346377578, 0.7762747778537251, 0.2224162679425837, 0.07366472765732417, 0.07366472765732417, 0.005618154573960865], 'Gaussian Naive Bayes', None), ([0.73226774244412463, 0.7319939423906715, 0.8670330147878229, 0.8651084082496034, 0.8689576213260424, 0.13104237867395765, 0.1348915917503966, 0.7051824431517716, 0.76973000683527, 0.0501161995898838, 0.07493389740877843, 0.07493389740877843, 0.00494545996795317], 'K-nearest neighbours', None), ([0.71646901487249404, 0.71335705812151595, 0.8603502389091329, 0.8911686938127973, 0.8295317840054682, 0.1704682159945318, 0.10883130618720256, 0.7856425171866737, 0.7854203691045797, 0.11018455228981545, 0.07136964569011106, 0.07136964569011106, 0.022550860360855365], 'Logistic regression', None), ([0.72140472465485284, 0.71884967675625555, 0.8627343736727522, 0.8889000528820731, 0.8365686944634314, 0.1634313055365687, 0.111099947117927, 0.7103648863035431, 0.7916848940533151, 0.050389610389610394, 0.06420412480169221, 0.06420412480169221, 2.453944251321239], 'Random forest', None), ([0.72302866618060724, 0.71958950196453819, 0.8636601985228444, 0.8970333157059758, 0.830287081339713, 0.1697129186602871, 0.10296668429402431, 0.8116287678476996, 0.7916951469583049, 0.11512303485987696, 0.07984135378106821, 0.07984135378106821, 0.5009829159417007], 'SVM (RDF)', None), ([0.72226361128671446, 0.71177422561655546, 0.862661743958955, 0.9328662083553676, 0.7924572795625426, 0.20754272043745728, 0.06713379164463248, 0.9211263881544157, 0.7924572795625426, 0.205974025974026, 0.06582231623479642, 0.06582231623479642, 0.7619422687059512], 'SVM (linear)', None), ([0.73039160692352656, 0.72947605709011809, 0.8667480380678633, 0.8765309360126917, 0.8569651401230349, 0.14303485987696515, 0.12346906398730831, 0.8002855631940772, 0.7687012987012988, 0.09708817498291183, 0.07525118984664199, 0.07525118984664199, 0.5413697372249904], 'SVM (polynomial)', None)], None), ('Logistic regression', [([0.72040681589895506, 0.71161803181396444, 0.8620323677686116, 0.9250661025912216, 0.7989986329460014, 0.20100136705399865, 0.07493389740877843, 0.5330037017451084, 0.5974538619275462, 0.03158578263841422, 0.06189317821258593, 0.06189317821258593, 0.14041021541924123], 'AdaBoost', 'Logistic regression'), ([0.72207226523294588, 0.71885780762191476, 0.8630985063254262, 0.8937704918032787, 0.8324265208475735, 0.16757347915242654, 0.10622950819672132, 0.7246747752511898, 0.794289131920711, 0.06943267259056733, 0.0723320994182972, 0.0723320994182972, 5.915993589704891], 'Artificial neural network', 'Logistic regression'), ([0.72098875206363711, 0.71374975731337797, 0.8625301514567152, 0.9182284505552618, 0.8068318523581681, 0.19316814764183182, 0.08177154944473823, 0.8973611845584347, 0.7976794258373205, 0.17436773752563225, 0.07754098360655737, 0.07754098360655737, 0.007846845987500118], 'Bernoulli Naive Bayes', 'Logistic regression'), ([0.70013479629684983, 0.69182601572419256, 0.8516318917938083, 0.9094394500264411, 0.7938243335611757, 0.20617566643882435, 0.09056054997355896, 0.876679005817028, 0.7637833219412167, 0.18038619275461382, 0.07233738762559493, 0.07233738762559493, 0.005254383306426733], 'Decision Tree', 'Logistic regression'), ([0.71920684102840082, 0.7093274517784034, 0.8612314230482703, 0.9289529349550503, 0.7935099111414902, 0.2064900888585099, 0.07104706504494977, 0.7812903225806451, 0.7822795625427205, 0.11903964456596035, 0.058635642517186684, 0.058635642517186684, 0.07911522472256038], 'Extreme Learning Machine', 'Logistic regression'), ([0.69895592460927114, 0.68695618664848745, 0.8505881946826589, 0.9270068746694872, 0.7741695146958305, 0.2258304853041695, 0.07299312533051297, 0.9263511369645689, 0.7739097744360903, 0.22452153110047846, 0.07266525647805396, 0.07266525647805396, 0.005088994311756423], 'Gaussian Naive Bayes', 'Logistic regression'), ([0.71074861975993742, 0.70052215699896236, 0.8569043203171625, 0.9260232681121099, 0.7877853725222147, 0.21221462747778536, 0.07397673188789, 0.9191803278688525, 0.7794292549555709, 0.20021872863978127, 0.06255949233209943, 0.06255949233209943, 0.00446646880761854], 'K-nearest neighbours', 'Logistic regression'), ([0.7011810798382907, 0.69819809817796297, 0.8526128687060665, 0.881681649920677, 0.8235440874914559, 0.17645591250854412, 0.11831835007932309, 0.759497620306716, 0.7838414217361585, 0.09684894053315106, 0.06907985193019567, 0.06907985193019567, 0.011189309574148005], 'Logistic regression', 'Logistic regression'), ([0.71264050203958063, 0.70711092645403117, 0.8584278454296405, 0.9048016922263352, 0.812053998632946, 0.187946001367054, 0.09519830777366474, 0.7393072448439978, 0.7901093643198907, 0.06892002734107996, 0.06418826017979905, 0.06418826017979905, 2.5656110125706513], 'Random forest', 'Logistic regression'), ([0.72494268906140402, 0.71415411073820501, 0.8639739929001389, 0.9354838709677418, 0.792464114832536, 0.20753588516746416, 0.06451612903225808, 0.919497620306716, 0.7916814764183185, 0.19577922077922078, 0.06321523003701746, 0.06321523003701746, 0.35585082812222657], 'SVM (RDF)', 'Logistic regression'), ([0.71669471542145557, 0.70698347785833326, 0.8599882379859558, 0.9269962982548915, 0.7929801777170198, 0.20701982228298016, 0.0730037017451084, 0.9185245901639345, 0.7927204374572796, 0.20623376623376624, 0.06811210999471178, 0.06811210999471178, 0.27546924108819315], 'SVM (linear)', 'Logistic regression'), ([0.70107293367368939, 0.70009485964868212, 0.8510624950072888, 0.8480750925436278, 0.8540498974709501, 0.14595010252904989, 0.15192490745637227, 0.7793072448439979, 0.7720334928229666, 0.10546138072453862, 0.06451084082496034, 0.06451084082496034, 0.3435990927301291], 'SVM (polynomial)', 'Logistic regression')], 'Logistic regression'), ('Bernoulli Naive Bayes', [([0.72040681589895506, 0.71161803181396444, 0.8620323677686116, 0.9250661025912216, 0.7989986329460014, 0.20100136705399865, 0.07493389740877843, 0.0, 0.0, 0.0, 0.0, 0.0, 0.26159509972844763], 'AdaBoost', 'Bernoulli Naive Bayes'), ([0.71472299604041345, 0.70720750514369979, 0.8593277262552081, 0.9162665256478052, 0.802388926862611, 0.19761107313738896, 0.08373347435219461, 0.7555896351136966, 0.796907040328093, 0.12791866028708135, 0.0742781597038604, 0.0742781597038604, 8.564618738974564], 'Artificial neural network', 'Bernoulli Naive Bayes'), ([0.69926073788653587, 0.69340039378782015, 0.8516943381842905, 0.8999471179270226, 0.8034415584415585, 0.19655844155844157, 0.10005288207297727, 0.8329613960867267, 0.764025974025974, 0.1631168831168831, 0.07982548915917505, 0.07982548915917505, 0.015725266512036384], 'Bernoulli Naive Bayes', 'Bernoulli Naive Bayes'), ([0.65617983601122609, 0.64213070466836564, 0.8280196440815996, 0.9112956107879429, 0.7447436773752563, 0.2552563226247436, 0.08870438921205713, 0.86520888418826, 0.7196821599453178, 0.22211893369788105, 0.07826017979904812, 0.07826017979904812, 0.0048911855725483205], 'Decision Tree', 'Bernoulli Naive Bayes'), ([0.42015674766299915, 0.40569830049191014, 0.6971566668461933, 0.5389476467477524, 0.8553656869446344, 0.1446343130553657, 0.46105235325224747, 0.20338445267054467, 0.7864695830485304, 0.023212576896787424, 0.05540983606557377, 0.05540983606557377, 0.19484474745803895], 'Extreme Learning Machine', 'Bernoulli Naive Bayes'), ([0.68311941036227075, 0.6744477944722328, 0.8431737168868038, 0.9048334214701214, 0.7815140123034859, 0.21848598769651403, 0.09516657852987838, 0.9038551031200424, 0.7781373889268626, 0.21822624743677377, 0.09483870967741935, 0.09483870967741935, 0.005114078862689908], 'Gaussian Naive Bayes', 'Bernoulli Naive Bayes'), ([0.71108284621506568, 0.70194641850906681, 0.8572556345794539, 0.9217874140666312, 0.7927238550922762, 0.20727614490772384, 0.07821258593336858, 0.9113432046536225, 0.7835850991114148, 0.20126452494873548, 0.06712850343733474, 0.06712850343733474, 0.0054399063510100374], 'K-nearest neighbours', 'Bernoulli Naive Bayes'), ([0.67563182539373479, 0.67171827827054054, 0.8398097167104097, 0.875156002115283, 0.8044634313055365, 0.19553656869446343, 0.12484399788471708, 0.30926493918561604, 0.7799487354750514, 0.049839371155160635, 0.06483870967741936, 0.06483870967741936, 0.008072516387969826], 'Logistic regression', 'Bernoulli Naive Bayes'), ([0.70317059696315953, 0.6963408568717131, 0.8535794693213493, 0.9071179270227393, 0.8000410116199589, 0.19995898838004098, 0.09288207297726071, 0.7448387096774194, 0.7781203007518797, 0.12446343130553657, 0.06126916975145426, 0.06126916975145426, 2.785270055868216], 'Random forest', 'Bernoulli Naive Bayes'), ([0.72256533915386145, 0.71229794943141, 0.8628726378828665, 0.9322421998942358, 0.7935030758714969, 0.20649692412850307, 0.06775780010576414, 0.9169169751454257, 0.7895967190704033, 0.2004818865345181, 0.0625383395029085, 0.0625383395029085, 0.4277649151955022], 'SVM (RDF)', 'Bernoulli Naive Bayes'), ([0.71931154176803225, 0.70911959709312111, 0.8612256866265466, 0.9302538339502909, 0.7921975393028025, 0.20780246069719754, 0.06974616604970915, 0.9169063987308302, 0.7921975393028025, 0.20518455228981541, 0.06844526705446854, 0.06844526705446854, 0.1250162098722651], 'SVM (linear)', 'Bernoulli Naive Bayes'), ([0.0065660141329927897, 0.0064655356057576443, 0.5032467532467533, 0.06000000000000001, 0.9464935064935066, 0.0535064935064935, 0.9399999999999998, 0.5820994182971972, 0.35515379357484617, 0.12675666438824335, 0.024077207826546803, 0.024077207826546803, 0.5066373540658707], 'SVM (polynomial)', 'Bernoulli Naive Bayes')], 'Bernoulli Naive Bayes'), ('SVM (linear)', [([0.72040681589895506, 0.71161803181396444, 0.8620323677686116, 0.9250661025912216, 0.7989986329460014, 0.20100136705399865, 0.07493389740877843, 0.42022739291380223, 0.24266575529733422, 0.04076555023923445, 0.021147540983606557, 0.021147540983606557, 0.3481788495277095], 'AdaBoost', 'SVM (linear)'), ([0.70919606886361541, 0.70305849926234132, 0.8560322405335488, 0.8965996827075623, 0.8154647983595351, 0.18453520164046477, 0.10340031729243786, 0.7586885245901639, 0.7821667805878332, 0.12131578947368422, 0.07068217874140667, 0.07068217874140667, 7.147994835782223], 'Artificial neural network', 'SVM (linear)'), ([0.72087203511333631, 0.71393697595565198, 0.8624948265572832, 0.9165891062929667, 0.8084005468215996, 0.19159945317840052, 0.08341089370703332, 0.9028873611845585, 0.8029186602870814, 0.18533492822966507, 0.0778900052882073, 0.0778900052882073, 0.007663315217724076], 'Bernoulli Naive Bayes', 'SVM (linear)'), ([0.67494370272603921, 0.66346418544978181, 0.8384107102644357, 0.9113907985193019, 0.7654306220095695, 0.23456937799043062, 0.08860920148069804, 0.8772236911686939, 0.7169207108680794, 0.20414217361585787, 0.06549444738233738, 0.06549444738233738, 0.00499638371937265], 'Decision Tree', 'SVM (linear)'), ([0.71699317743618307, 0.70659495329435029, 0.8600273058373062, 0.9299418297197249, 0.7901127819548872, 0.2098872180451128, 0.07005817028027497, 0.8017768376520358, 0.7810013670539986, 0.15478127136021871, 0.06059756742464305, 0.06059756742464305, 0.1363719649504543], 'Extreme Learning Machine', 'SVM (linear)'), ([0.70416727752713926, 0.69256204006843169, 0.8533176135979582, 0.9279851930195665, 0.77865003417635, 0.22134996582365005, 0.07201480698043364, 0.9276573241671076, 0.777863978127136, 0.22134996582365005, 0.07201480698043364, 0.07201480698043364, 0.005148158836218997], 'Gaussian Naive Bayes', 'SVM (linear)'), ([0.71138837827988777, 0.70155890337435134, 0.8572920583271554, 0.9247276573241671, 0.7898564593301437, 0.21014354066985647, 0.07527234267583288, 0.9159386567953464, 0.7775734791524264, 0.19840396445659605, 0.06581702802749868, 0.06581702802749868, 0.004500367665256633], 'K-nearest neighbours', 'SVM (linear)'), ([0.69897688174480277, 0.69371068825551918, 0.8515449833781126, 0.8959915388683235, 0.8070984278879016, 0.1929015721120984, 0.10400846113167635, 0.7645901639344262, 0.7835953520164047, 0.127365003417635, 0.06713907985193021, 0.06713907985193021, 0.010519120404638132], 'Logistic regression', 'SVM (linear)'), ([0.71313015103456878, 0.70681385507683492, 0.8586377200271965, 0.9094182971972501, 0.8078571428571429, 0.19214285714285712, 0.09058170280274987, 0.7888365943945003, 0.7906254272043746, 0.12553656869446345, 0.06289793759915388, 0.06289793759915388, 2.7375018866238667], 'Random forest', 'SVM (linear)'), ([0.72063864857842241, 0.71053254413462663, 0.8619160985970528, 0.9305922792173453, 0.7932399179767602, 0.2067600820232399, 0.06940772078265468, 0.9181914331041776, 0.7919412166780588, 0.2028468899521531, 0.06514542570068746, 0.06514542570068746, 0.3473489463281155], 'SVM (RDF)', 'SVM (linear)'), ([0.7190667157954348, 0.7088416913785005, 0.8610905282893789, 0.9302432575356955, 0.7919377990430622, 0.2080622009569378, 0.06975674246430459, 0.9195081967213115, 0.791678058783322, 0.2067600820232399, 0.06714965626652564, 0.06714965626652564, 0.22649704358493689], 'SVM (linear)', 'SVM (linear)'), ([0.71693691745181887, 0.70970032797761418, 0.8604708754603687, 0.9159280803807508, 0.8050136705399863, 0.19498632946001368, 0.08407191961924908, 0.8173347435219462, 0.7934996582365004, 0.1404032809295967, 0.06714436805922792, 0.06714436805922792, 0.36499017748149853], 'SVM (polynomial)', 'SVM (linear)')], 'SVM (linear)'), ('Random forest', [([0.73217714822857316, 0.72982044413055425, 0.8681316976880449, 0.8934267583289264, 0.8428366370471634, 0.15716336295283664, 0.10657324167107352, 0.6175832892649392, 0.6525085440874915, 0.04280246069719754, 0.06841353781068218, 0.06841353781068218, 0.17791035125615745], 'AdaBoost', 'Random forest'), ([0.71947781449194959, 0.7181069524848257, 0.8614148313069101, 0.8755261766261236, 0.8473034859876964, 0.15269651401230347, 0.12447382337387625, 0.7393759915388685, 0.7948017771701983, 0.06521531100478468, 0.0778688524590164, 0.0778688524590164, 10.69242368078204], 'Artificial neural network', 'Random forest'), ([0.71838275453887079, 0.71515291980745144, 0.8613224955205666, 0.8931200423056584, 0.829524948735475, 0.17047505126452492, 0.10687995769434162, 0.8494500264410364, 0.8018489405331511, 0.13339712918660285, 0.08732416710735061, 0.08732416710735061, 0.0072487409426926774], 'Bernoulli Naive Bayes', 'Random forest'), ([0.70418681933888583, 0.69638595211297272, 0.8539308667201873, 0.9119936541512429, 0.795868079289132, 0.20413192071086805, 0.08800634584875729, 0.871909042834479, 0.7509398496240601, 0.1785304169514696, 0.0677895293495505, 0.0677895293495505, 0.005150815202624649], 'Decision Tree', 'Random forest'), ([0.72752862420029096, 0.71673556313235953, 0.8652790812544076, 0.9367953463775779, 0.7937628161312372, 0.2062371838687628, 0.063204653622422, 0.8142834479111581, 0.7843574846206425, 0.11693096377306904, 0.059619249074563715, 0.059619249074563715, 0.06857660417363405], 'Extreme Learning Machine', 'Random forest'), ([0.70503276064601961, 0.69206637155985606, 0.8534555082119027, 0.9335378106821789, 0.7733732057416267, 0.2266267942583732, 0.06646218931782126, 0.9335378106821789, 0.7728537252221462, 0.22610731373889265, 0.06646218931782126, 0.06646218931782126, 0.005055910475621062], 'Gaussian Naive Bayes', 'Random forest'), ([0.7230538053147455, 0.72077326888151549, 0.8635463981452597, 0.887916446324696, 0.8391763499658236, 0.16082365003417637, 0.11208355367530407, 0.7459016393442621, 0.7885167464114832, 0.06029733424470267, 0.07428873611845585, 0.07428873611845585, 0.0041034823756078255], 'K-nearest neighbours', 'Random forest'), ([0.72061194538768392, 0.71622042515778905, 0.862504468589386, 0.9022474881015338, 0.8227614490772386, 0.17723855092276142, 0.09775251189846643, 0.8067794817556848, 0.7877614490772386, 0.10545112781954888, 0.06841882601797991, 0.06841882601797991, 0.017836594828056544], 'Logistic regression', 'Random forest'), ([0.71664287460217235, 0.71340364781520749, 0.8604376642172713, 0.8921364357482814, 0.828738892686261, 0.1712611073137389, 0.10786356425171865, 0.7370333157059757, 0.7911722488038279, 0.05456596035543404, 0.06288736118455843, 0.06288736118455843, 2.678916480217919], 'Random forest', 'Random forest'), ([0.71642301233513273, 0.70804640360592608, 0.8600590929152119, 0.9211263881544157, 0.7989917976760081, 0.20100820232399177, 0.07887361184558433, 0.8713008989952407, 0.7877614490772386, 0.16103896103896104, 0.06842411422527764, 0.06842411422527764, 0.36940614514144776], 'SVM (RDF)', 'Random forest'), ([0.72101730862225466, 0.71060178087763848, 0.8620532910686409, 0.931909042834479, 0.7921975393028025, 0.20780246069719754, 0.06809095716552088, 0.9257165520888417, 0.7921975393028025, 0.20649692412850307, 0.06712321523003703, 0.06712321523003703, 0.4948669643406965], 'SVM (linear)', 'Random forest'), ([0.71398429980867706, 0.70853510745946735, 0.8591420543329864, 0.9051877313590693, 0.8130963773069035, 0.18690362269309638, 0.09481226864093072, 0.8100423056583818, 0.7906356801093646, 0.1080451127819549, 0.07786356425171867, 0.07786356425171867, 0.3358353185647396], 'SVM (polynomial)', 'Random forest')], 'Random forest')]]
    parameter = "Balanced Accuracy"
    x_label = "Feature selection approach"
    difference_from = "without feature selection"
    datasets = ["Lima TB", "Test", "blash"]
    name_suffix = ""
    patterns = (None, "////")

    colors = ["#64B3DE", "#1f78b4", "#B9B914", "#FBAC44", "#bc1659", "#33a02c", "#6ABF20", "#ff7f00", "#6a3d9a", "#5a2add", "#b15928", "#e31a1c", "grey"]
    classifier_arr = []
    for i in range(len(results)):
        classifier_arr.append(list())
    index = 0
    for results_per_classifier in results:
        no_feature_selection = results[index][0][1]
        for i in range(len(no_feature_selection) + 1):
            classifier_arr[index].append(list())
        for i in range(1, len(results_per_classifier)):
            data_balancer_results = results_per_classifier[i][1]
            x = 0
            mean_classification = 0
            for (result_arr, data_balancer_name, _) in data_balancer_results:
                value = result_arr[2] - no_feature_selection[x][0][2]
                classifier_arr[index][x].append(value)
                mean_classification += value
                x += 1
            mean_classification /= float(len(data_balancer_results))
            classifier_arr[index][x].append(mean_classification)
        index += 1

    fig = plt.figure(figsize=(15, 2.5))

    classifiers = np.arange(len(classifier_arr[0]))
    data_balancers = np.arange(len(classifier_arr[0][0])) * 3
    bar_width = 0.2
    opacity = 0.9
    subplt_val = 101 + (10 * len(results))
    ax1 = plt.subplot(subplt_val)
    for i in range(len(classifier_arr[0])):
        if i + 1 != len(classifier_arr[0]):
            label = results[0][0][1][i][1]
        else:
            label = "Mean classification"
        plt.bar(data_balancers + (i * bar_width), classifier_arr[0][i], bar_width,
                alpha=opacity,
                color=colors[i],
                hatch=patterns[i % len(patterns)],
                label=label)

    legend = plt.legend(loc='lower center', bbox_to_anchor=(len(results) / 2.0, -0.60), fancybox=True, frameon=True, ncol=6)
    legend.get_frame().set_facecolor('#ffffff')

    plt.xlabel(x_label, x=len(results) / 2.0)
    plt.ylabel("Difference in {0}".format(parameter, difference_from))
    feature_selection_labels = [results[0][i][0] for i in range(1, len(results[0]))]
    plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)
    plt.title(datasets[0].replace("_", " "))

    for z in range(1, len(results)):
        ax2 = plt.subplot(subplt_val + z, sharey=ax1)
        color = iter(cm.Set1(np.linspace(0, 1, len(no_feature_selection) + 1)))
        for i in range(len(classifier_arr[z])):
            if i + 1 != len(classifier_arr[z]):
                label = results[z][0][1][i][1]
            else:
                label = "Mean classification"
            plt.bar(data_balancers + (i * bar_width), classifier_arr[z][i], bar_width,
                    alpha=opacity,
                    color=colors[i],
                    hatch=patterns[i % len(patterns)],
                    label=label)

        feature_selection_labels = [results[z][i][0] for i in range(1, len(results[z]))]
        plt.xticks(data_balancers + (bar_width / 2) * len(classifiers), feature_selection_labels)
        plt.title(datasets[z].replace("_", " "))

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plt.locator_params(axis='y', nbins=15)
    name = "{3}_results_per_classifier_plot{0}_{4}_{1}_{2}".format(name_suffix, parameter, current_time, x_label, datasets)
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../results/{0}".format(name.replace(" ", "_")), bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)