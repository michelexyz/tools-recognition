import numpy as np
import pandas as pd


from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



from IPython.display import display

import pickle
import dill


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
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis()]

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

                train_predictions = clf.predict_proba(X_test)
                ll = log_loss(y_test, train_predictions)
                print("Log Loss: {}".format(ll))

                log_entry = pd.DataFrame([[fold, name, acc * 100, ll]], columns=log_cols, index=[index])
                log = pd.concat([log,log_entry])
                index +=1

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
    def train_with_subset(self,train_index):
        X_train = self.X[train_index]
        y_train = self.labels[train_index]
        for clf in self.classifiers:
            clf.fit(X_train, y_train)
            name = clf.__class__.__name__

            print("=" * 30)
            print(name + ' trained with subset')
            print("=" * 30)

    def predict_with_fold(self, x, fold, n_splits = 3,test_size=0.2, random_state = 23, use_best = True, n_best = 5):

        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

        for i, (train_index, _) in enumerate(sss.split(self.X, self.labels)):
            if i==fold:

                #print(f"Fold {i}:")
                #print(f"  Train: index={train_index}")
                #print(f"  Test:  index={test_index}")

                self.train_with_subset(train_index)

                if use_best:
                    return self.predict_with_best_n( x, n_best)
                else:
                    return self.predict(X_train, range(len(self.classifiers)))
                print("=" * 30)

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
            raise Exception("'preditc' takes only an array of shape(1,) as X,"
                            "shape given: {} ".format(l))


    def predict_with_best_n(self,x, n):
        #TODO
        #self.train()

        if not self.mean.empty:
            if n <= self.mean.shape[0]:
                best_classifiers = self.mean.head(n)
                best_c_names = best_classifiers.index.tolist()
                best_c_indices = self.get_ids_by_names(best_c_names)
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
    def predict_with_prob(self, x):
        n = 3
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
            cl_index = None
            for i, cl in enumerate(classifiers):
                if cl_name == cl:
                    cl_index = i

                    break
            if cl_index == None:
                raise Exception("Classifier not found in dataframe")
            prediction = best_pred[p]
            weight = accuracy_w[cl_index]
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


    def _predict_with_name(self, x, cl_name: str):
        id = self.get_ids_by_names([cl_name])
        return self.predict(x, id, cl_name)



        #TODO 0
    def get_ids_by_names(self, names):
        ids =[]
        for i, clf in enumerate(self.classifiers):
            for name in names:
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