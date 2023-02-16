import os

import numpy as np
import pandas as pd


from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from matplotlib import pyplot as plt


from IPython.display import display

import pickle
import dill

from files.files_handler import get_abs_path


#TODO metti solo classificatori che hanno predict_proba

class Models:

    def __init__(self, X= None, labels = None, file_names= None, category_legend= None):
        self.category_legend = category_legend
        self.file_names = file_names
        self.X = X
        self.labels = labels
        self.classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="rbf", C=0.025, probability=True),
            NuSVC(nu=0.2,probability=True),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            GaussianNB(),
            LinearDiscriminantAnalysis()
        ]

        #self.best_classifier = None
        #self.mean = None
        #self.le


    def train_and_test(self, n_splits=3, test_size=0.2, random_state=23):
        # Logging for Visual Comparison
        log_cols=["Fold","Classifier", "Accuracy", "Log Loss"]
        index = 0
        log = pd.DataFrame(columns=log_cols)



        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

        for fold, (train_index, test_index) in enumerate(sss.split(self.X, self.labels)):
            print(f"Fold {fold}:")
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")

            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]

            for clf in self.classifiers:
                clf.fit(X_train, y_train)
                name = clf.__class__.__name__

                print("=" * 30)
                print(name)

                print('****Results****')
                train_predictions = clf.predict(X_test)
                acc = accuracy_score(y_test, train_predictions)
                print("Accuracy: {:.4%}".format(acc))

                train_predictions_proba = clf.predict_proba(X_test)
                ll = log_loss(y_test, train_predictions_proba)
                print("Log Loss: {}".format(ll))

                log_entry = pd.DataFrame([[fold, name, acc * 100, ll]], columns=log_cols, index=[index])
                log = pd.concat([log,log_entry])
                index +=1

                if name == 'RandomForestClassifier':
                    conf_matrix = confusion_matrix(y_true=y_test, y_pred=train_predictions)

                    fig, ax = plt.subplots(figsize=(12, 12))
                    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
                    ax.set_xticks(np.arange(0,self.category_legend[0].shape[0]),self.category_legend[0])
                    ax.set_yticks(np.arange(0,self.category_legend[0].shape[0]),self.category_legend[0])
                    for i in range(conf_matrix.shape[0]):
                      for j in range(conf_matrix.shape[1]):
                          ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
                    plt.xlabel('Predictions', fontsize=18)
                    plt.ylabel('Actuals', fontsize=18)
                    plt.title('Confusion Matrix', fontsize=18)
                    plt.show()

            print("=" * 30)

        display(log)
        mean = log.groupby('Classifier')[['Accuracy', "Log Loss"]].mean()
        self.mean = mean.sort_values(by=['Accuracy'], ascending=False)
        display(self.mean)

    def train(self):
        for clf in self.classifiers:
            clf.fit(self.X, self.labels)
            name = clf.__class__.__name__

            print("=" * 30)
            print(name + ' trained with all the data')
            print("=" * 30)

            if name == 'DecisionTreeClassifier':
                path = str(get_abs_path('graphs', 'tree'))

                from sklearn.tree import export_graphviz
                # Export as dot file
                export_graphviz(clf, out_file=path+'.dot',
                                feature_names=np.arange(0, self.X[0].shape[0]),
                                class_names=self.category_legend[0],
                                rounded=True, proportion=False,
                                precision=2, filled=True)

                # Convert to png using system command (requires Graphviz)
                from subprocess import call
                call(['dot', '-Tpng', path+'.dot', '-o', path+'.png', '-Gdpi=600'])

                #os.system('dot -Tpng tree.dot -o tree.png')

                # Display in jupyter notebook
                from IPython.display import Image
                Image(filename=path+'.png')

            if name == 'RandomForestClassifier':
                # Extract single tree
                estimator = clf.estimators_[5]
                path = str(get_abs_path('graphs','r_tree'))

                from sklearn.tree import export_graphviz
                # Export as dot file
                export_graphviz(estimator, out_file=path+'.dot',
                                feature_names=np.arange(0, self.X[0].shape[0]),
                                class_names=self.category_legend[0],
                                rounded=True, proportion=False,
                                precision=2, filled=True)

                # Convert to png using system command (requires Graphviz)
                from subprocess import call
                call(['dot', '-Tpng', path+'.dot', '-o', path+'.png', '-Gdpi=600'])

                # Display in jupyter notebook
                from IPython.display import Image
                Image(filename=path+'.png')


    def train_with_subset(self,train_index):
        X_train = self.X[train_index]
        y_train = self.labels[train_index]
        for clf in self.classifiers:
            clf.fit(X_train, y_train)
            name = clf.__class__.__name__

            print("=" * 30)
            print(name + ' trained with subset')
            print("=" * 30)

    def train_with_fold(self, fold, n_splits = 3,test_size=0.2, random_state = 23, use_best = True, n_best = 5):


        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

        for i, (train_index, _) in enumerate(sss.split(self.X, self.labels)):
            if i==fold:

                #print(f"Fold {i}:")
                #print(f"  Train: index={train_index}")
                #print(f"  Test:  index={test_index}")

                self.train_with_subset(train_index)

                # if use_best:
                #     return self.predict_with_best_n( x, n_best)
                # else:
                #     return self.predict(X_train, range(len(self.classifiers)))
                # print("=" * 30)

    def predict(self,c_indices, X):
        l =  len(X)
        if l == 1:
            log_cols = ["Classifier", "Prediction"]
            log = pd.DataFrame(columns=log_cols)

            for clf in np.array(self.classifiers,  dtype=object)[c_indices]:
                name = clf.__class__.__name__
                prediction = clf.predict(X)
                ii = int(prediction[0])
                predicted_category = self.category_legend[0][ii]

                log_entry = pd.DataFrame([[name, predicted_category]], columns=log_cols)
                log = pd.concat([log, log_entry])

            return log
        else:
            raise Exception("'predict' takes only an array of shape(1,) as X,"
                            "shape given: {} ".format(l))

    def predict_proba(self, c_indices, X):
        l = len(X)
        if l == 1:
            log_cols = ["Classifier", "Prediction","Probability"]
            log = pd.DataFrame(columns=log_cols)

            for clf in np.array(self.classifiers, dtype=object)[c_indices]:
                name = clf.__class__.__name__
                prediction = clf.predict_proba(X)

                highest_prob = 0
                best_pred_index = -1
                for index, prob in enumerate(prediction[0]):
                    if prob > highest_prob:
                        highest_prob = prob
                        best_pred_index = index
                if best_pred_index ==-1:
                    raise Exception("nessuna predizione provata highest_prob: {}".format(highest_prob))


                #ii = int(prediction[0])
                predicted_category = self.category_legend[0][best_pred_index]

                log_entry = pd.DataFrame([[name, predicted_category, highest_prob]], columns=log_cols)
                log = pd.concat([log, log_entry])

            return log
        else:
            raise Exception("'predict' takes only an array of shape(1,) as X,"
                            "shape given: {} ".format(l))


    def predict_with_best_n(self,x, n, predict_proba = False):
        #TODO
        #self.train()

        if not self.mean.empty:
            if n <= self.mean.shape[0]:
                best_classifiers = self.mean.head(n)
                best_c_names = best_classifiers.index.tolist()
                best_c_indices = self.get_ids_by_names(best_c_names)

                if predict_proba:
                    return self.predict_proba(best_c_indices, x)
                else:
                    return self.predict(best_c_indices, x)




            else:
                raise Exception("n biggest than number of classifiers in 'predict_with_best_n'")
        else:
            raise Exception("models not with yet with self.train_and_test()")

    # def save(self, name):
    #     """save class as self.name.txt"""
    #     # Step 2
    #     with open(f'{name}.pkl', 'wb') as file:
    #         # Step 3
    #         pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    # def load(self, name):
    #     """try load self.name.txt"""
    #     file = open(name + '.txt', 'r')
    #     dataPickle = file.read()
    #     file.close()
    #
    #     self.__dict__ = pickle.load(dataPickle)
    # TODO fun for wrong predictions file names

    #TODO metti i commenti del
    def predict_with_multiple(self, x, n):
        best = self.predict_with_best_n(x,n)
        print("="*30)
        print("Best classifiers:")
        display(best)
        print("="*30)
        best_cl = np.array(best["Classifier"].tolist())
        best_pred = np.array(best["Prediction"].tolist())

        classifiers = np.array(self.mean.head(n).index.tolist())
        accuracy = np.array(self.mean.head(n)['Accuracy'].tolist())

        total_accuracy = accuracy.sum()
        accuracy_w = accuracy/total_accuracy

        probabilistc_prediction = []
        for p, cl_name in enumerate(best_cl):
            # cl_index = None
            # for i, cl in enumerate(classifiers):
            #     if cl_name == cl:
            #         cl_index = i
            #
            #         break
            # if cl_index == None:
            #     raise Exception("Classifier not found in dataframe")

            prediction = best_pred[p]
            weight = accuracy_w[p]
            pred_already_present = False
            for pred in probabilistc_prediction:
                if prediction == pred[0]:
                    pred[1] += weight
                    pred_already_present = True
                    break
            if not pred_already_present:
                probabilistc_prediction.append([prediction,weight])


        best_pred_name = 'None'
        best_pred_w = 0

        if len(probabilistc_prediction)>0:
            for pred in probabilistc_prediction:
                debug = pred
                if best_pred_w < pred[1]:
                    best_pred_w = pred[1]
                    debug = pred[0]
                    best_pred_name = pred[0]
        else:
            raise Exception("Probabilistic prediction array is empty")
        if best_pred_name != 'None':
            return [best_pred_name, best_pred_w]
        else:
            raise Exception("Best prediction not found ")

    def predict_with_multiple_proba(self, x, n):

        best = self.predict_with_best_n(x,n, predict_proba=True)
        print("="*30)
        print("Best classifiers:")
        display(best)
        print("="*30)
        best_cl = np.array(best["Classifier"].tolist())
        best_pred = np.array(best["Prediction"].tolist())
        best_prob = np.array(best["Probability"].tolist())

        classifiers = np.array(self.mean.head(n).index.tolist())
        accuracy = np.array(self.mean.head(n)['Accuracy'].tolist())

        accuracy = accuracy * best_prob

        total_accuracy = accuracy.sum()
        accuracy_w = accuracy/total_accuracy

        probabilistc_prediction = []

        #per ogni classificatore dei migliori
        for p, cl_name in enumerate(best_cl):

            # estrai la relativa accuratezza e previsione
            prediction = best_pred[p]
            weight = accuracy_w[p]
            pred_already_present = False

            # e itera sull'array delle previsioni probabilistiche
            for pred in probabilistc_prediction:
                #se è gia presente somma il peso al peso della previsione attuale
                if prediction == pred[0]:
                    pred[1] += weight
                    pred_already_present = True
                    break
            #Altrimenti aggiungila all'array delle previsioni probabilistiche
            if not pred_already_present:
                probabilistc_prediction.append([prediction,weight])

        #Estrai il nome della previsione migliore e il suo peso
        best_pred_name = 'None'
        best_pred_w = 0

        if len(probabilistc_prediction)>0:
            for pred in probabilistc_prediction:
                debug = pred
                if best_pred_w < pred[1]:
                    best_pred_w = pred[1]
                    debug = pred[0]
                    best_pred_name = pred[0]
        else:
            raise Exception("Probabilistic prediction array is empty")
        if best_pred_name != 'None':
            return [best_pred_name, best_pred_w]
        else:
            raise Exception("Best prediction not found ")




    def _predict_with_name(self, x, cl_name: str):
        id = self.get_ids_by_names([cl_name])
        return self.predict(x, id, cl_name)



        #TODO 0
    #Finds the ides of the classifiers mantaining the order given in 'names'
    def get_ids_by_names(self, names):
        ids =[]

        for name in names:
            for i, clf in enumerate(self.classifiers):
                if name == clf.__class__.__name__.split('(', 1)[0]:
                    ids.append(i)

        return ids





#
#
# data = np.load('data.npy', allow_pickle=True)
#
#
# X = data[0]
# labels = data[1]
# file_names = data[2]
# category_legend = data[3]
#
#
# m = Models(X, labels, file_names, category_legend)
# m.train_and_test()
# m.train()
# display(m.predict_with_best_n([X[0]], 5))
# # name ='cl1'
# #
# # with open(f'{name}.pkl', 'wb') as file:
# #     # Step 3
# #     pickle.dump(m, file, pickle.HIGHEST_PROTOCOL)
# #
# # ofile = open("BinaryData", "wb")
# # dill.dump(m, ofile)
# # ofile.close()
#
# #my_pickled_object = pickle.dumps(m)  # Pickling the object
#
# #display(m.predict_with_fold([X[0]], 2))
# print(category_legend[0][0])
#
# exit(0)