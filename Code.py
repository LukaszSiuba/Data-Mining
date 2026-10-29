from sklearn.datasets import load_digits
import numpy as np; import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay,accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


'''
Before we start, I am omitting including a readme.md since this is a basic script that runs everything at once.
Ensure before running that you have the following packages installed, you can run this command to install them:

pip install numpy matplotlib scikit-learn

After installing, you can run the script by running:

python Code.py

Afterwards, you will be able to see the terminal outputs that we're used in the reports and after a few minutes or seconds depending on your compute performane there will be the six plots used in the report displayed.

@Date: 10/28/2026
@Author: Lukasz Siuba
'''

def digits():
    #Load and standardize the data
    digits = load_digits()
    data = digits.data
    target = digits.target
    mean = np.mean(data, axis=0)
    std =  np.std(data, axis=0)
    #Check if in std array we have 0, if so replace value to 1 to prevent division by zero so Nan  avoided
    for i in range(len(std)):
        if std[i] == 0:
            std[i] = 1
    #Returns standardized data
    standardized = (data-mean)/std

    #Splits standardized data and target labels into training and testing subsets, stratify based on the target/labels, random_state = 1 to reproducible data
    standardized_train, standardized_test, target_train, target_test = train_test_split(standardized,target, train_size=0.85, test_size=0.15, stratify=target, random_state = 1)

    """
    To plot the class distribution we have to realize that the digits dataset has a total of 10 classes,
    with each class representing a digit from (0-9) inclusive. So we are plotting to ensure that our 
    training and validation sets are stratified based on the above classes.
    """
    #Need to find the count of training and validation labels in each class for each split
    classes, training_labels = np.unique(target_train, return_counts=True, equal_nan=True)
    _, validation_labels = np.unique(target_test, return_counts=True, equal_nan=True)
    digit_classes = np.arange(len(classes)) #total class labels

    #Get the proportions for training and validation for bar plot
    training_labels_proportion = (training_labels / len(target_train)) * 100
    validation_labels_proportion = (validation_labels / len(target_test)) * 100
    width = 0.3
    #Plot the training and validation labels 
    plt.figure(figsize=(12,8))
    plt.bar(digit_classes-width/2, training_labels_proportion, label="Training Set", color="#b34242", width=width, edgecolor='black', linewidth=2)
    plt.bar(digit_classes+width/2, validation_labels_proportion, label="Validation Set", color="#2330e0", width=width, edgecolor='black', linewidth=2)
    plt.title("Class Distribution in Training and Validation Sets")
    plt.xlabel("Classes in Digits"); plt.xlim(-1, 10)
    plt.ylabel("Percentage of Samples"); plt.ylim(0, 15)
    ticks = np.arange(len(digit_classes))
    #(ticks, labels)
    plt.xticks(ticks, digit_classes)
    plt.legend(loc=0)
    return standardized_train, standardized_test, target_train, target_test

#Part B Configure SVM using SVC ----
 #SVC Takes (n_samples,n_features) -> Training and (n_samples) -> Class labels
def SVM(standardized_train, target_train):
    C = [0.01, 0.1, 1, 10, 100]
    kernel_types = ['linear', 'rbf','poly']
    kernel_and_scores = {}
    for types in kernel_types:
        f1_scores= []
        for num in C:
            classifier=svm.SVC(kernel=types, C=num)
            scores = cross_validate(classifier, standardized_train, target_train, scoring='f1_macro', cv=10)
            average_f1_score = np.mean(scores['test_score'])
            f1_scores.append(average_f1_score)
            # Debug: print(f"For Kernel Type: {types} and C value {num} has average fl score of {average_f1_score}")
        kernel_and_scores[types] = f1_scores

    print("Linear score ->",kernel_and_scores['linear'])
    print("----------------")
    print("rbf score ->",kernel_and_scores['rbf'])
    print("----------------")
    print("poly score ->",kernel_and_scores['poly'])
    print("linear best score", max(kernel_and_scores['linear']))
    print("rbf best score", max(kernel_and_scores['rbf']))
    print("poly best score", max(kernel_and_scores['poly']))

    width = 0.3
    C_index = np.arange(len(C))
    num_kernel = len(kernel_and_scores)
    plt.figure(figsize=(12,12))
    #Enumerate over the kernel and scores dictionary, calculate an offset between the repsective kernel types
    for i, (kernel,scores) in enumerate(kernel_and_scores.items()):
        offset = (i-(num_kernel - 1)/2) * width
        #Plot the average f1_score of the kernel types accounting for an offset so they do not stack on top of eachother.
        plt.bar(C_index + offset, scores, width=width, label=kernel)
    plt.xlabel("Values of C")
    plt.ylabel("Average F measure")
    plt.title("Average F Measure vs Values of C ")
    plt.xticks(C_index, C)
    plt.ylim(0,1.1)
    plt.legend()

def NN(standardized_train, target_train):
    k = [1,2,5,10,50]
    knn_f1_score =[]
    for num in k:
        classifier = KNeighborsClassifier(n_neighbors=num)
        scores = cross_validate(classifier, standardized_train, target_train, scoring='f1_macro', cv=10)
        average_f1_score = np.mean(scores['test_score'])
        knn_f1_score.append(average_f1_score)
        print(f"For k-value {num}, score is {average_f1_score}")
    best_score = max(knn_f1_score)
    best_index = knn_f1_score.index(best_score)
    best_k = k[best_index]
    print(standardized_train.shape, target_train.shape)
    print(f"Best K value is {best_k} with the corresponding value {best_score}")
    width = 0.3
    k_index = np.arange(len(k))
    plt.figure(figsize=(12,8))
    plt.bar(k_index, knn_f1_score, width=width, label=k)
    plt.xlabel("Values of K")
    plt.ylabel("Average F measure")
    plt.title("Average F Measure vs Values of K ")
    plt.xticks(k_index, k)
    plt.ylim(.92, .98)
def MLP(standardized_train, target_train):
    #Holds the comination of our neurons and our layers that we need to evaluate
    neuron_combinations = [
        (10,),
        (50,),
        (100,),
        (10,50),
        (10,100),
        (10,100,50)
    ]
    mlp_scores =[]
    for combination in neuron_combinations:
        #Explained in the doc but we choose max_iter which is the max number of iterations to 1000 so all MLP configs converge so its fair. 
        classifier = MLPClassifier(hidden_layer_sizes=combination, random_state=1 , max_iter=1000)
        scores = cross_validate(classifier, standardized_train, target_train, scoring='f1_macro', cv=10)
        average_f1_score = np.mean(scores['test_score'])
        mlp_scores.append(average_f1_score)
    print(mlp_scores)

    width = 0.3
    n_index = np.arange(len(neuron_combinations))
    plt.figure(figsize=(12,8))
    plt.bar(n_index, mlp_scores, width=width, label=neuron_combinations)
    plt.xlabel("MLP Variant")
    plt.ylabel("Average F measure")
    plt.title("Average F Measure vs MLP Variant ")
    plt.xticks(n_index, neuron_combinations)
    plt.ylim(.94,.99)
#Each classifier will be trained manually instead of doing a loop.
def CompareClassifiers(standardized_train, standardized_test, target_train, target_test):
    #C-SVM, initialize the classifier and fit the training data subset
    svm_classifier = svm.SVC(C=100, kernel='poly').fit(standardized_train, target_train)
    #Predict using the standardized_test subset
    svm_prediction = svm_classifier.predict(standardized_test)
    #Compute confusion matrix (y_true, y_predicted) and accuracy score
    svm_confusion_matrix= confusion_matrix(target_test, svm_prediction )
    svm_accuracy = accuracy_score(target_test, svm_prediction)
    #Compute the average class precision, average class recall, and average class F-measure all MACRO.
    svm_average_precision, svm_average_class_recall, svm_average_f_measure, _ = precision_recall_fscore_support(target_test, svm_prediction, average='macro')

    #K nearest neighbor
    knn_classifier = KNeighborsClassifier(n_neighbors = 5).fit(standardized_train, target_train)
    knn_prediction = knn_classifier.predict(standardized_test)
    knn_confusion_matrix = confusion_matrix(target_test, knn_prediction)
    knn_accuracy = accuracy_score(target_test, knn_prediction)
    knn_average_precision, knn_average_class_recall, knn_average_f_measure, _ = precision_recall_fscore_support(target_test, knn_prediction, average='macro')

    #MLP
    mlp_classifer = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=1).fit(standardized_train, target_train)
    mlp_prediction = mlp_classifer.predict(standardized_test)
    mlp_confusion_matrix = confusion_matrix(target_test, mlp_prediction)
    mlp_accuracy = accuracy_score(target_test, mlp_prediction)
    mlp_average_precision, mlp_average_class_recall, mlp_average_f_measure, _ = precision_recall_fscore_support(target_test, mlp_prediction, average='macro')

    #LDA
    lda_classifier = LinearDiscriminantAnalysis().fit(standardized_train, target_train)
    lda_prediction = lda_classifier.predict(standardized_test)
    lda_confusion_matrix = confusion_matrix(target_test, lda_prediction)
    lda_accuracy = accuracy_score(target_test, lda_prediction)
    lda_average_precision, lda_average_class_recall, lda_average_f_measure, _ = precision_recall_fscore_support(target_test, lda_prediction, average='macro')

    #Naive Bayes (Gaussian)
    nb_classifier = GaussianNB().fit(standardized_train, target_train)
    nb_prediction = nb_classifier.predict(standardized_test)
    nb_confusion_matrix = confusion_matrix(target_test, nb_prediction)
    nb_accuracy = accuracy_score(target_test, nb_prediction)
    nb_average_precision, nb_average_class_recall, nb_average_f_measure, _ = precision_recall_fscore_support(target_test, nb_prediction, average='macro')

    #Print the scores
    print(f" SVM average precision is: {svm_average_precision}, Average class recall:  {svm_average_class_recall}, Average F-Measure: {svm_average_f_measure}")
    print(f" KNN average precision is: {knn_average_precision}, Average class recall:  {knn_average_class_recall}, Average F-Measure: {knn_average_f_measure}")
    print(f" MLP average precision is: {mlp_average_precision}, Average class recall:  {mlp_average_class_recall}, Average F-Measure: {mlp_average_f_measure}")
    print(f" LDA average precision is: {lda_average_precision}, Average class recall:  {lda_average_class_recall}, Average F-Measure: {lda_average_f_measure}")
    print(f" Naive Bayes average precision is: {nb_average_precision}, Average class recall:  {nb_average_class_recall}, Average F-Measure: {nb_average_f_measure}")
    print("---------------------------------------------------------------------")
    print("Accuracy score for each confusion matrix")
    print(f"Confusion Matrix Score for SVM: {svm_accuracy}")
    print(f"Confusion Matrix Score for KNN: {knn_accuracy}")
    print(f"Confusion Matrix Score for MLP: {mlp_accuracy}")
    print(f"Confusion Matrix Score for LDA: {lda_accuracy}")
    print(f"Confusion Matrix Score for NB: {nb_accuracy}")

    #Display the confusions matrices and bar graphs for each classifier
    fig, axes = plt.subplots(2,3, figsize=(12,10))
    confusion_matrices = [
    ConfusionMatrixDisplay(confusion_matrix=svm_confusion_matrix),
    ConfusionMatrixDisplay(confusion_matrix=knn_confusion_matrix),
    ConfusionMatrixDisplay(confusion_matrix=mlp_confusion_matrix),
    ConfusionMatrixDisplay(confusion_matrix=lda_confusion_matrix),
    ConfusionMatrixDisplay(confusion_matrix=nb_confusion_matrix)
    ]
    names = ['SVM', 'KNN', 'MLP', 'LDA', 'NB']
    for i, (matrices, name) in enumerate(zip(confusion_matrices, names)):
        matrices.plot(ax=axes.flatten()[i],colorbar=False, cmap='Blues')
        axes.flatten()[i].set_title(name)
    axes.flatten()[5].axis('off')
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    #Lists holding our metrics
    metrics = {
    "Average Class Precisions": [svm_average_precision, knn_average_precision, mlp_average_precision, lda_average_precision, nb_average_precision],
    "Average Class Recall": [svm_average_class_recall, knn_average_class_recall, mlp_average_class_recall, lda_average_class_recall, nb_average_class_recall],
    "Average Class F-measure": [svm_average_f_measure, knn_average_f_measure, mlp_average_f_measure, lda_average_f_measure, nb_average_f_measure],
    }
    #plot the metrics for each classifier.
    width=0.3
    fig,axes = plt.subplots(1,3, figsize=(12,8))
    classifier_index = np.arange(len(names))
    for axis, (metric, scores) in zip(axes, metrics.items()):
        bar_plots = axis.bar(classifier_index, scores, width=width)
        axis.set_ylim(0.72, .99)
        axis.set_ylabel(metric)
        axis.set_xticks(classifier_index)
        axis.set_xticklabels(names)
        axis.set_title(metric)
    plt.tight_layout()
#Dunder main
if __name__ == '__main__':
    standardized_train, standardized_test, target_train, target_test = digits()
    print("Part b. --------------------")
    SVM(standardized_train, target_train)
    print("Part C. ------------")
    NN(standardized_train, target_train)
    print("Part D. ---------------")
    MLP(standardized_train, target_train)
    print("Part E.---------------------")
    CompareClassifiers(standardized_train, standardized_test, target_train, target_test)
    plt.show()