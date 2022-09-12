import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from pathlib import Path
import skimage
from skimage.io import imread
from skimage.transform import resize
import seaborn as sns


def preprocess_data(X):
    # TODO: Preprocess your data here. Also use this to determine your best features
    return X


def k_nearest(X, y):
    # TODO: Initialize KNN model, and train
    quit()


def logistic_regression(X, y):
    # TODO: Initialize Logistic Regression, and train
    quit()


def random_forest(X, y):
    # TODO: Initialize Random Forest, and train
    quit()


def xgboost(X,y):
    # TODO: Initialize XGBoost, and train
    quit()


def svm(X, y):
    # TODO: Initialize SVM, and train
    quit()


def mlp(X, y):
    # TODO: Initialize MLP, and train
    quit()


model_map = {
    'k_nearest': k_nearest,
    'logistic_regression': logistic_regression,
    'random_forest': random_forest,
    'xgboost': xgboost,
    'svm': svm,
    'mlp': mlp
}


def load_image_files(container_path, dimension=(30, 30)):

    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "Your own dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    # return in the exact same format as the built-in datasets
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


def score_metrics(model, X_test, y_test):
    # TODO: Score the model using accuracy, precision and recall
    
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test,y_pred,average='macro')# average 
    recall = metrics.recall_score(y_test,y_pred,average='macro') # average 

    return precision, accuracy, recall


def compare_models(classifiers,models, X_test, y_test):
    # TODO: Draw Bar Graph(s) showing accuracy, precision and recall

    scores = []
    prec =[]
    acc = []
    r = []

    for model in models:
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111)
        precision, accuracy, recall = score_metrics(model, X_test, y_test)
        scores.append([precision, accuracy, recall])
        prec.append(precision)
        acc.append(accuracy)
        r.append(recall)
        fig.subtitle(classifiers[models.index(model)])
        ax.bar(['precision', 'accuracy', 'recall'], scores[models.index(model)])
    plt.show()
    # TODO: report on best model for precision, accuracy and recall
    

    print("Best precision:",str(round(max(prec),3)),"Classifier:",classifiers[prec.index(max(prec))])
    print("Best precision:",str(round(max(acc),3)),"Classifier:",classifiers[acc.index(max(acc))])
    print("Best precision:",str(round(max(r),3)),"Classifier:",classifiers[r.index(max(r))])

def visualize_data(X, y):
    # TODO: visualize the data
    quit()

# Entry Point of Program
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='./images')
    p.add_argument('--classifiers', type=str)
    args = p.parse_args()
    print("Welcome to the multiple model classifier. I see that you have chosen", args.classifiers, "as your model(s) of choice.")

    # Load Dataset
    cats_dogs = load_image_files(args.dataset)
    X = cats_dogs.data
    y = cats_dogs.target

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    # Visualize Data
    visualize_data(X_train, y_train)

    # Preprocess Data
    X_train_new = preprocess_data(X_train)

    # Splits the command-line arguments into separate classifiers
    classifiers = args.classifiers.split(',')

    # Iterate over selected classifiers and create a model based on the choice of classifier
    selected_models = []
    for cl in classifiers:
        model = model_map[cl]
        trained_model = model(X_train_new, y_train)

        # Append a trained model to selected_models
        selected_models.append(trained_model)

    # Preprocess Test data
    X_test_new = preprocess_data(X_test)

    # If multiple models selected: make a bar graph comparing them. If not, just report on results
    if len(selected_models) > 1:
        compare_models(classifiers,selected_models, X_test_new, y_test)
    else:
        prec, acc, recall = score_metrics(selected_models[0], X_test_new, y_test)
        print("Accuracy for model", args.classifiers ,":", acc)
        print("Precision for model", args.classifiers ,":", prec)
        print("Recall for model", args.classifiers ,":", recall)
    