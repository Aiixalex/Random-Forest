from typing import Optional, Sequence, Mapping
import numpy as np
import pandas as pd
import random


class Node(object):
    def __init__(self, node_size: int, node_class: str, depth: int, single_class: bool = False):
        # Every node is a leaf unless you set its 'children'
        self.is_leaf = True
        # Each 'decision node' has a name. It should be the feature name
        self.name = None
        # All children of a 'decision node'. Note that only decision nodes have children
        self.children = {}
        # Whether corresponding feature of this node is numerical or not. Only for decision nodes.
        self.is_numerical = None
        # Threshold value for numerical decision nodes. If the value of a specific data is greater than this threshold,
        # it falls under the 'ge' child. Other than that it goes under 'l'. Please check the implementation of
        # get_child_node for a better understanding.
        self.threshold = None
        # The class of a node. It determines the class of the data in this node. In this assignment it should be set as
        # the mode of the classes of data in this node.
        self.node_class = node_class
        # Number of data samples in this node
        self.size = node_size
        # Depth of a node
        self.depth = depth
        # Boolean variable indicating if all the data of this node belongs to only one class. This is condition that you
        # want to be aware of so you stop expanding the tree.
        self.single_class = single_class

    def set_children(self, children):
        self.is_leaf = False
        self.children = children

    def get_child_node(self, feature_value) -> 'Node':
        if not self.is_numerical:
            return self.children[feature_value]
        else:
            if feature_value >= self.threshold:
                return self.children['ge']  # ge stands for greater equal
            else:
                return self.children['l']  # l stands for less than


class RandomForest(object):
    def __init__(self, n_classifiers: int,
                 criterion: Optional['str'] = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = None,
                 max_features: Optional[int] = None):
        """
        :param n_classifiers:
            number of trees to generated in the forest
        :param criterion:
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “entropy” for the information gain.
        :param max_depth:
            The maximum depth of the trees.
        :param min_samples_split:
            The minimum number of samples required to be at a leaf node
        :param max_features:
            The number of features to consider for each tree.
        """
        self.n_classifiers = n_classifiers
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.criterion_func = self.entropy if criterion == 'entropy' else self.gini

    def fit(self, X: pd.DataFrame, y_col: str) -> float:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of training dataset
        """
        features = self.process_features(X, y_col)
        for i in range(self.n_classifiers):
            new_df = pd.DataFrame(data = X)
            for row_idx in range(len(X)):
                new_df.iloc[row_idx] = X.iloc[random.randrange(len(X))]

            random.shuffle(features)
            features_partition = features[0:self.max_features]

            new_tree = self.generate_tree(new_df, y_col, features_partition)
            self.trees.append(new_tree)

        return self.evaluate(X, y_col)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        :param X: data
        :return: aggregated predictions of all trees on X. Use voting mechanism for aggregation.
        """
        predictions = []
        for index, row in X.iterrows():
            node_classes = []
            for node in self.trees:
                while not node.is_leaf():
                    feature_name = node.name
                    node = node.get_child_node(row[feature_name])

                node_classes.append(node.node_class)

            prediction = max(set(node_classes), key = node_classes.count)
            predictions.append(prediction)

        return np.array(predictions)

    def evaluate(self, X: pd.DataFrame, y_col: str) -> int:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of predictions on X
        """
        preds = self.predict(X)
        acc = sum(preds == X[y_col]) / len(preds)
        return acc

    def generate_tree(self, X: pd.DataFrame, y_col: str, features: Sequence[Mapping]) -> Node:
        """
        Method to generate a decision tree. This method uses self.split_tree() method to split a node.
        :param X:
        :param y_col:
        :param features:
        :return: root of the tree
        """
        root = Node(X.shape[0], X[y_col].mode(), 0)
        self.split_node(root, X, y_col, features)
        return root

    def split_node(self, node: Node, X: pd.DataFrame, y_col: str, features: Sequence[Mapping]) -> None:
        """
        This is probably the most important function you will implement. This function takes a node, uses criterion to
        find the best feature to slit it, and splits it into child nodes. I recommend to use revursive programming to
        implement this function but you are of course free to take any programming approach you want to implement it.
        :param node:
        :param X:
        :param y_col:
        :param features:
        :return:
        """
        min_gini_score = 1
        best_feature = None
        best_threshold = None
        for feature in features:
            if feature['dtype'] == 'int64':
                thresholds = np.unique([np.percentile(X[feature['name']], q) for q in np.linspace(0, 100, 100)])
                for threshold in thresholds:
                    gini_score = self.gini(X, feature, y_col, threshold)
                    if gini_score < min_gini_score:
                        min_gini_score = gini_score
                        best_feature = feature
                        best_threshold = threshold
            elif feature['dtype'] == 'object':
                gini_score = self.gini(X, feature, y_col, None)
                if gini_score < min_gini_score:
                    min_gini_score = gini_score
                    best_feature = feature

        node.name = best_feature['name']
        children = {}
        if best_feature['dtype'] == 'object':
            node.is_numerical = False

            for feature_value in X[best_feature['name']].unique():
                X_subset = X.loc[X[best_feature['name']] == feature_value]
                is_single_class = (len(X_subset[y_col].value_counts()) == 1)
                children[feature_value] = Node(len(X_subset), X_subset[y_col].mode(), node.depth + 1, is_single_class)

                if not is_single_class & (node.depth <= (self.max_depth - 1)) & (len(X_subset) >= self.min_samples_split):
                    self.split_node(children[feature_value], X_subset, y_col, features)

        elif best_feature['dtype'] == 'int64':
            node.is_numerical = True
            node.threshold = best_threshold

            X_subsets = {}
            X_subsets['l'] = X.loc[X[best_feature['name']] < best_threshold]
            X_subsets['ge'] = X.loc[X[best_feature['name']] >= best_threshold]
            for key in X_subsets.keys():
                X_subset = X_subsets[key]
                is_single_class = (len(X_subset[y_col].value_counts()) == 1)
                children[key] = Node(len(X_subset), X_subset[y_col].mode(), node.depth + 1, is_single_class)

                if not is_single_class & (node.depth <= (self.max_depth - 1)) & (len(X_subset) >= self.min_samples_split):
                    self.split_node(children[key], X_subset, y_col, features)

        node.set_children(children)

    def gini(self, X: pd.DataFrame, feature: Mapping, y_col: str, threshold: int) -> float:
        """
        Returns gini index of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """

        gini_score = 0

        size_dataset = X.shape[0]
        num_of_label_vals = len(X[y_col].unique())

        if feature['dtype'] == 'object':
            for feature_value in X[feature['name']].unique():
                X_subset = X.loc[X[feature['name']] == feature_value]

                gini = 1

                for label_val in X[y_col].unique():
                    gini -= ((len(X_subset.loc[X[y_col] == label_val]) / len(X_subset)) ** 2)

                gini_score += len(X_subset) / size_dataset * gini

        elif feature['dtype'] == 'int64':
            
            X_subsets = {}
            X_subsets['l'] = X.loc[X[feature['name']] < threshold]
            X_subsets['ge'] = X.loc[X[feature['name']] >= threshold]

            for key in X_subsets.keys():
                X_subset = X_subsets[key]
                gini = 1
                for label_val in X[y_col].unique():
                    gini -= ((len(X_subset.loc[X[y_col] == label_val]) / len(X_subset)) ** 2)

                gini_score += len(X_subset) / size_dataset * gini

        return gini_score


    def entropy(self, X: pd.DataFrame, feature: Mapping, y_col: str) -> float:
        """
        Returns gini index of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """
        pass

    def process_features(self, X: pd.DataFrame, y_col: str) -> Sequence[Mapping]:
        """
        :param X: data
        :param y_col: name of the label column in X
        :return:
        """
        features = []
        for n, t in X.dtypes.items():
            if n == y_col:
                continue
            f = {'name': n, 'dtype': t}
            features.append(f)
        return features
