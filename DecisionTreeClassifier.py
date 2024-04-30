import numpy as np
import pandas as pd


class DecisionTreeClassifier:
    def __init__(self, threshold_info_gains=0.02):
        """Initialize the Decision Tree Classifier with a threshold for information gains."""
        self.tree = None
        self.threshold_info_gains = threshold_info_gains

    def entropy(self, target_col):
        """
        Calculate the entropy of the target column.

        Parameters:
            target_col (array-like): Target column with class labels.

        Returns:
            float: The entropy value.
        """
        counts = np.bincount(target_col)  # Count frequency of each class in the target column
        probabilities = counts / np.sum(counts)  # Calculate probabilities for each class
        entropy = -np.sum(
            probabilities * np.log2(probabilities + 1e-9))  # Calculate entropy, adding epsilon to avoid log2(0)
        return entropy

    def info_gain(self, data, attribute, target):
        """
        Calculate the information gain for a specific attribute relative to the target.

        Parameters:
            data (DataFrame): The dataset including the target column.
            attribute (str): The feature for which information gain is calculated.
            target (str): The target column name.

        Returns:
            float: The information gain.
        """
        total_entropy = self.entropy(data[target])  # Calculate total entropy of the target column
        values = data[attribute].astype('category').cat.codes  # Convert attribute to categorical codes
        counts = np.bincount(values, minlength=len(data[attribute].cat.categories))  # Count frequency of each category
        weighted_entropy = 0
        for val in np.unique(values):  # Iterate over unique category codes
            subset = data[
                data[attribute] == data[attribute].cat.categories[val]]  # Data subset where attribute equals category
            prob = counts[val] / np.sum(counts)  # Probability of this category
            weighted_entropy += prob * self.entropy(subset[target])  # Calculate weighted entropy
        information_gain = total_entropy - weighted_entropy  # Information gain is the reduction in entropy
        return information_gain

    def build_tree(self, data, features, target):
        """
        Recursively build the decision tree.

        Parameters:
            data (DataFrame): The dataset used for building the tree.
            features (list): List of feature names.
            target (str): The target variable name.

        Returns:
            dict: A nested dictionary representing the decision tree.
        """
        if len(np.unique(data[target])) == 1:
            # If all entries are of the same class, return a leaf node
            return {'attribute': None, 'classification': data[target].iloc[0]}

        info_gains = np.array(
            [self.info_gain(data, feature, target) for feature in features])  # Calculate info gains for all features
        if np.all(info_gains < self.threshold_info_gains):
            # If no significant info gains, return a leaf node with the majority class
            return {'attribute': None, 'classification': data[target].mode()[0]}

        best_feature = features[np.argmax(info_gains)]  # Feature with the highest info gain
        tree = {'attribute': best_feature, 'children': {}}  # Start a new node
        remaining_features = [feature for feature in features if
                              feature != best_feature]  # Remaining features for further splits

        for value in data[best_feature].unique():  # For each unique value in the best feature
            subtree = self.build_tree(data[data[best_feature] == value], remaining_features,
                                      target)  # Recursively build subtree
            tree['children'][value] = subtree  # Attach subtree

        return tree

    def fit(self, X, y):
        """
        Fit the decision tree model using the training data.

        Parameters:
            X (DataFrame): Feature data.
            y (Series): Target data.
        """
        data = pd.concat([X, y], axis=1)  # Combine features and target into one DataFrame
        self.tree = self.build_tree(data, list(X.columns), y.name)  # Build the tree

    def traverse_tree(self, instance, node):
        """
        Traverse the tree to classify a given instance.

        Parameters:
            instance (Series): The data instance to classify.
            node (dict): The current node of the tree.

        Returns:
            The classification of the instance.
        """
        while node['attribute'] is not None:  # Traverse until a leaf node is reached
            if instance[node['attribute']] not in node['children']:
                return None  # Return None if the value is not found in the tree
            node = node['children'][instance[node['attribute']]]  # Move to the corresponding child node
        return node['classification']  # Return the classification from the leaf node

    def predict(self, X):
        """
        Predict the class labels for the given input.

        Parameters:
            X (DataFrame): The input features.

        Returns:
            array: Predicted class labels.
        """
        return np.array(
            [self.traverse_tree(row, self.tree) for _, row in X.iterrows()])  # Apply the traverse function to each row

    def print_tree(self, node=None, level=0):
        """
        Print the decision tree structure.

        Parameters:
            node (dict, optional): The current node to print. Defaults to the root node.
            level (int, optional): The current level in the tree for indentation purposes.
        """
        if node is None:
            node = self.tree
        indent = "  " * level
        if node['attribute'] is None:
            print(f"{indent}|--(Leaf, class: {node['classification']})")  # Print the leaf node
        else:
            for key, value in node['children'].items():
                print(f"{indent}|--{node['attribute']}={key}")  # Print the attribute and its value
                self.print_tree(value, level + 1)  # Recursively print each subtree

    def calculate_accuracy(self, X, y):
        """
        Calculate the accuracy of the classifier on the given test data and labels.

        Parameters:
            X (DataFrame): The test features.
            y (Series): The true labels.

        Returns:
            float: The accuracy percentage.
        """
        predictions = self.predict(X)  # Get predictions for the test data
        return np.mean(predictions == y)  # Calculate the accuracy as the mean of correct predictions
