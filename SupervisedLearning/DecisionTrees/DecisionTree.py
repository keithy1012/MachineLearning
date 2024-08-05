import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Node ():
    def __init__(self, feature_index = None, threshold= None, left= None, right= None, information_gain= None, value= None):
        
        self.feature_index = feature_index # Ex: "X1"
        self.threshold = threshold # Ex: "<10"
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.value = value

class DecisionTree():
    def __init__(self, minimum_sample_splits, max_depth):
        self.root = None
        self.minimum_sample_splits = minimum_sample_splits
        self.max_depth = max_depth

    def build_tree(self, dataset, current_depth):
        X,y = dataset[:, :-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        # For nodes
        if current_depth <= self.max_depth and num_samples > self.minimum_sample_splits:
            best_split = self.get_best_split(dataset, num_samples, num_features)

            if best_split["information_gain"] > 0:
                left = self.build_tree(best_split["left"], current_depth+1)
                right = self.build_tree(best_split["right"], current_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], left, right, best_split["information_gain"])
        # For leaf nodes (stopping conditions met)
        leaf_value = self.calculate_leaf_value(y)
        return Node(value = leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")

        for feature in range(num_features):
            feature_values = dataset[:, feature]
            possible_threshold = np.unique(feature_values)
            for threshold in possible_threshold:
                left, right = self.split(dataset, feature, threshold)
                if len(left) > 0 and len(right) > 0:
                    y,left_y, right_y = dataset[:, -1], left[:, -1], right[:, -1]
                    current_info_gain = self.information_gain(y, left_y, right_y)
                    if (current_info_gain > max_info_gain):
                        best_split["feature_index"] = feature
                        best_split["threshold"] = threshold
                        best_split["left"] = left
                        best_split["right"] = right
                        best_split["information_gain"] = current_info_gain
                        max_info_gain = current_info_gain
        return best_split

    # Splits the dataset based on the feature index and threshold
    def split(self, dataset, feature, threshold):
        left = np.array([row for row in dataset if row[feature] <= threshold])
        right = np.array([row for row in dataset if row[feature] > threshold])
        return left, right
    
    def information_gain(self, parent, left, right):
        left_weight = len(left) / len(parent)
        right_weight = len(right) / len(parent)

        gain = self.entropy(parent) - (left_weight * self.entropy(left) + right_weight * self.entropy(right))
        return gain
    
    def entropy(self, y):
        labels = np.unique(y)
        entropy = 0
        for label in labels:
            probability_of_label = len(y[y==label]) / len(y)
            entropy += -probability_of_label * np.log(probability_of_label)
        return entropy
    
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, y):
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset, 0)

    def predict(self, X):
        predictions = [self.make_predictions(x, self.root) for x in X]
        return predictions
    
    def make_predictions(self, x, tree):
        if tree.value!= None:
            return tree.value
        
        feature_values = x[tree.feature_index]
        if feature_values <= tree.threshold:
            return self.make_predictions(x, tree.left)
        else:
            return self.make_predictions(x, tree.right)

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.information_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv("iris.csv", skiprows=1, header=None, names=col_names)
data.head(10)
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
classifier = DecisionTree(minimum_sample_splits=3, max_depth=3)
classifier.fit(X_train,Y_train)
classifier.print_tree()

Y_pred = classifier.predict(X_test) 

accuracy_score(Y_test, Y_pred)