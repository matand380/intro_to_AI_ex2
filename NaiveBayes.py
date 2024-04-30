import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        self.class_probs = {}  # Dictionary to store class probabilities
        self.feature_probs = {}  # Dictionary to store feature probabilities for each class

    def fit(self, train_data, train_target):
        """Fit the Naive Bayes model to the training data."""
        # Calculate class probabilities
        class_counts = train_target.value_counts()
        self.class_probs = (class_counts / class_counts.sum()).to_dict()

        # Calculate feature probabilities for each class
        for cls in self.class_probs:
            self.feature_probs[cls] = {}
            data_cls = train_data[train_target == cls]
            for feature in train_data.columns:
                feature_counts = data_cls[feature].value_counts()
                self.feature_probs[cls][feature] = (feature_counts / feature_counts.sum()).to_dict()

    def predict(self, test_data):
        """Predict the class for each instance in the test data."""
        predictions = []
        for _, instance in test_data.iterrows():
            class_probs = self.class_probs.copy()  # Start with the prior probabilities

            for cls in class_probs:
                for feature in test_data.columns:
                    feature_value = instance[feature]
                    class_probs[cls] *= self.feature_probs[cls].get(feature, {}).get(feature_value, 1e-10)  # Small smoothing factor

            predicted_class = max(class_probs, key=class_probs.get)  # Class with the highest probability
            predictions.append(predicted_class)

        return predictions

    def accuracy(self, predicted, target):
        """Calculate the accuracy of the predictions."""
        return np.mean(predicted == target)  # Mean of correct predictions as accuracy

