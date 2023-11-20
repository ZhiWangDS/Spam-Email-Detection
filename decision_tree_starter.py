# You may want to install "gprof2dot"
import io
from collections import Counter

import numpy as np
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import pydot

eps = 1e-5  # a small number


class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None, flag = None, m=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes
        self.flag = flag

    @staticmethod 
    def entropy(y):

        if  len(y) ==0 :
            return 0     
       # uniques, counts = np.unique(y, return_counts=True)
        #p_l = counts[0]/len(y) ; p_r = counts[1]/len(y)
        p_l = len(np.where(y<0.5)[0])/len(y) ;p_r  = 1-p_l;
        H_s = - p_l* np.log2(p_l + eps) - p_r* np.log2(p_r+eps)
        return  H_s

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO: implement information gain function         
        # H(s) - H_after ; right X > T ; left X<= T
 
        H_s = DecisionTree.entropy(y)

        s_l = y[np.where(X < thresh)[0]]
        s_r = y[np.where(X >= thresh)[0]]

        H_sl = DecisionTree.entropy(s_l)
        H_sr = DecisionTree.entropy(s_r)

        H_after = (len(s_l)*H_sl + len(s_r)*H_sr ) / (len(s_l) +len(s_r))

        return H_s - H_after
     
       # return np.random.rand()
    @staticmethod 
    def gini(y):

        if  len(y) ==0 :
            return 0  

        p_l = len(np.where(y<0.5)[0])/len(y) ;p_r  = 1-p_l;

        return 1 - p_l**2 -p_r**2


    @staticmethod
    def gini_impurity(X, y, thresh):
        # TODO: implement gini impurity function

        H_s = DecisionTree.gini(y)

        s_l = y[np.where(X < thresh)[0]]
        s_r = y[np.where(X >= thresh)[0]]

        H_sl = DecisionTree.gini(s_l)
        H_sr = DecisionTree.gini(s_r)

        H_after = (len(s_l)*H_sl + len(s_r)*H_sr ) / (len(s_l) +len(s_r))

        return H_s - H_after
        #pass

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        new_m = int((X.shape[1])**0.5)
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                #np.linspace(np.min(X[:,new_m]) + eps, np.max(X[:, new_m]) - eps, num=10)
                for i in range(X.shape[1])
            ])

            if self.flag:

                feature_size = int((X.shape[1])**0.5)
                random_columns = np.random.choice(X.shape[1],feature_size ,replace=False)
                for i in random_columns:
                    # gains.append([self.gini_impurity(X[:, i], y, t) for t in thresh[i, :]])
                    gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            else:
                for i in range(X.shape[1]):
                # gains.append([self.gini_impurity(X[:, i], y, t) for t in thresh[i, :]])
                    gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self
    

    


    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200,max_depth = 7):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        #self.flag = flag
        self.decision_trees = [
            #sklearn.tree.DecisionTreeClassifier(random_state=i, **self.params)
            DecisionTree(max_depth=max_depth, feature_labels=features)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO: implement function     
        for tree in self.decision_trees:
            # subsample X,y by bootstrap

            random_indexs = np.random.choice(len(y),X.shape[0])
            sub_sample = X[random_indexs]; sub_y = y[random_indexs]
 
            tree.fit(sub_sample,sub_y)
            
            #pass

    def predict(self, X):
        # TODO: implement function
        pred_hist = []

        for tree in self.decision_trees:
            pred = tree.predict(X)
            pred_hist.append(pred)

        pred_n = np.vstack(pred_hist)
        major_pred =stats.mode(pred_n).mode[0]
        return major_pred
        #pass


class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200,  flag = True):#m=1,
    #def __init__(self, max_depth,max_feature,num ):
        if params is None:
            params = {}
        # TODO: implement function
        # tree.DecisionTree --> max_feature
        self.n = n    
        self.flag =flag

        self.decision_trees =[DecisionTree(feature_labels=features) for i in range(self.n)]

     
       # pass

# https://www.scitepress.org/papers/2014/47390/47390.pdf

class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        # TODO: implement function
        return self

    def predict(self, X):
        # TODO: implement function
        pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == ''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == '-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)


if __name__ == "__main__":
    dataset = "titanic" #  "spam" 
    params = {
       # "max_depth": 7, #5
       # "random_state": 6,
        #"min_samples_leaf": 10,
        "flag": True
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = './dataset/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None, encoding=None)
        path_test = './dataset/titanic/titanic_test_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None, encoding=None)
        y = data[1:, -1]  # label = survived
        class_names = ["Died", "Survived"]
        labeled_idx = np.where(y != '')[0]

        y = np.array(y[labeled_idx])
        y = y.astype(float).astype(int)


        print("\n\nPart (b): preprocessing the titanic dataset") 
        X, onehot_features = preprocess(data[1:, :-1], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, :-1]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = './dataset/spam/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features:", features)
    print("Train/test size:", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree/RandomForest
    print("\n\nPart (a-b): simplified decision tree")
    #dt = DecisionTree(max_depth=3, feature_labels=features)
    dt = RandomForest(flag = True)
    dt.fit(X, y)
    print("Predictions", dt.predict(Z)[:100])

    y_pred = dt.predict(Z)

    y_test = y_pred.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv(dataset +'_submission.csv', index_label='Id')


    print("\n\nPart (c): sklearn's decision tree")
    clf = sklearn.tree.DecisionTreeClassifier( **params)
    #clf = RandomForest(**params)
    clf.fit(X, y)
    evaluate(clf)
    out = io.StringIO()

    # You may want to install "gprof2dot"
    sklearn.tree.export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
    graph = pydot.graph_from_dot_data(out.getvalue())
    pydot.graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)

    # TODO: implement and evaluate!


