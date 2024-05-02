import numpy as np
import pandas as pd


class DecisionTreeClassifier:
    def __init__(self, threshold_info_gains=0.02):
        """
        Initialize the Decision Tree Classifier with a threshold for information gains.

        Parameters:
            threshold_info_gains (float): The threshold for considering information gains significant.
        """
        self.tree = None  # Initialize the tree as None, will be built by fitting the model
        self.threshold_info_gains = threshold_info_gains  # Threshold for information gain

    def entropy(self, target_col):
        """
        Calculate the entropy of a target column.

        Parameters:
            target_col (Series): A pandas Series representing the target variable.

        Returns:
            float: The entropy of the target column.
        """
        elements, counts = np.unique(target_col, return_counts=True)  # Get unique elements and their frequency
        entropy = np.sum([
            (-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
            for i in range(len(elements))
        ])  # Calculate entropy for each unique element and sum
        return entropy

    def info_gain(self, data, attribute, target):
        """
        Calculate the information gain of an attribute relative to the target.

        Parameters:
            data (DataFrame): The entire dataset.
            attribute (str): The attribute whose info gain is to be calculated.
            target (str): The target variable in the dataset.

        Returns:
            float: The information gain from splitting on the specified attribute.
        """
        total_entropy = self.entropy(data[target])  # Calculate the total entropy of the target variable
        vals, counts = np.unique(data[attribute],
                                 return_counts=True)  # Get unique values and their frequency in the attribute
        weighted_entropy = np.sum([
            (counts[i] / np.sum(counts)) * self.entropy(data.where(data[attribute] == vals[i]).dropna()[target])
            for i in range(len(vals))
        ])  # Calculate weighted entropy for each unique value
        information_gain = total_entropy - weighted_entropy
        return information_gain

    def build_tree(self, data, features, target):
        """
        Recursively build the decision tree based on the data.

        Parameters:
            data (DataFrame): The data used to build the tree.
            features (list of str): The list of features to consider for splits.
            target (str): The target variable in the dataset.

        Returns:
            dict: A dictionary representation of the decision tree.
        """
        if len(np.unique(data[target])) == 1:
            # If all values of target are the same, return a leaf node with that classification
            return {'attribute': None, 'classification': data[target].unique()[0]}

        info_gains = [self.info_gain(data, feature, target) for feature in
                      features]  # Calculate info gains for all features
        if not info_gains or all(abs(gain) < self.threshold_info_gains for gain in info_gains):
            # If no significant info gains or no features left, return a leaf node with the majority class
            return {'attribute': None, 'classification': data[target].mode()[0]}

        best_feature_index = np.argmax(info_gains)  # Index of the feature with the highest info gain
        best_feature = features[best_feature_index]  # Name of the feature with the highest info gain
        tree = {'attribute': best_feature, 'children': {}}  # Initialize node with the best feature

        remaining_features = [feature for feature in features if
                              feature != best_feature]  # Exclude the best feature from remaining features

        for value in np.unique(data[best_feature]):
            # For each unique value of the best feature, build a subtree
            sub_data = data.where(
                data[best_feature] == value).dropna()  # Filter data for the value and drop missing values
            subtree = self.build_tree(sub_data, remaining_features, target)  # Recursively build a subtree
            tree['children'][value] = subtree  # Attach the subtree

        return tree

    def fit(self, X, y):
        """
        Fit the decision tree model using the training data.

        Parameters:
            X (DataFrame): The feature data.
            y (Series): The target data.
        """
        data = pd.concat([X, y], axis=1)  # Combine features and target into one DataFrame
        self.tree = self.build_tree(data, list(X.columns), y.name)  # Build the decision tree

    def traverse_tree(self, instance, node):
        """
        Recursively traverse the decision tree to classify an instance.

        Parameters:
            instance (Series): An instance to classify.
            node (dict): A node in the decision tree.

        Returns:
            The classification of the instance.
        """
        if node['attribute'] is None:
            return node['classification']  # Return classification if a leaf node is reached
        value = instance[node['attribute']]  # Get the attribute value for the instance
        if value not in node['children']:
            return None  # Return None if the attribute value is not in the tree (missing path)
        child = node['children'][value]  # Get the child node for the attribute value
        return self.traverse_tree(instance, child)  # Recursively traverse to the child node

    def predict(self, X):
        """
        Predict the class labels for a given input dataset.

        Parameters:
            X (DataFrame): The input features to predict.

        Returns:
            array: Predicted class labels.
        """
        return np.array([self.traverse_tree(row, self.tree) for _, row in
                         X.iterrows()])  # Apply traversal to each row in the dataset

    def print_tree(self, node=None, level=-1):
        """
        Print the decision tree in a readable format.

        Parameters:
            node (dict, optional): The current node to print. Defaults to the root node.
            level (int, optional): Current depth in the tree to manage indentation.
        """
        if node is None:
            node = self.tree
        attribute = node['attribute']
        level += 1  # Increment the level to increase indentation

        for key, value in node['children'].items():
            if value['attribute'] is None:
                print("\t" * level, "|", attribute, "=", key, ":", value['classification'])  # Print leaf node
            else:
                print("\t" * level, "|", attribute, "=", key)  # Print attribute and value
                self.print_tree(value, level)  # Recursively print subtree

    def calculate_accuracy(self, X, y):
        """
        Calculate the accuracy of the classifier on the test data.

        Parameters:
            X (DataFrame): The test features.
            y (Series): The true labels.

        Returns:
            float: The accuracy percentage.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)  # Calculate the accuracy as the mean of correct predictions

    def write_to_file(self, file, node=None, level=-1):
        """
        Write the decision tree to a file in a readable format.

        Parameters:
            file (file object): The file to write to.
            node (dict, optional): The current node to write. Defaults to the root node.
            level (int, optional): Current depth in the tree to manage indentation.
        """
        if node is None:
            node = self.tree
        attribute = node['attribute']
        level += 1

        for key, value in node['children'].items():
            if value['attribute'] is None:
                file.write("\t" * level + "| " + attribute + " = " + str(key) + ": " + value[
                    'classification'] + "\n")  # Write leaf node to file
            else:
                file.write(
                    "\t" * level + "| " + attribute + " = " + str(key) + "\n")
                self.write_to_file(file, value, level)
