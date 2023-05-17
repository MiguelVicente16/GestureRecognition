import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
import shared_func

"""
This class is a classifier using DTW as the distance measure between pairs of time series data.
"""
class DTW:

    def __init__(self, n_neighbors):
        """
        Initializes the KNN_DTW classifier.

        Args:
            n_neighbors (int): The number of neighbors to consider for classification.
        """
        self.n_neighbors = n_neighbors
        self.data = None
        self.labels = None

    def fit(self, data, labels):
        """
        Fits the DTW classifier with the training dataset and corresponding labels.

        Args:
            dataset (Dataset): An instance of the Dataset class.
            labels (list): The labels corresponding to each sequence in the dataset.
        """
        self.data = np.array(data, dtype='object')
        self.labels = np.array(labels)

    def predict(self, test_data):
        """
        Predicts the class of the test data.

        Args:
            test_data (ndarray): The test data for classification.

        Returns:
            ndarray: The predicted class labels for the test data.
        """
        dist_matrix = self.compute_distance_matrix(test_data)

        indexes = dist_matrix.argsort()[:, :self.n_neighbors]
        labels = self.labels[indexes]
        predictions = mode(labels, axis=1, keepdims=True)[0]
        return predictions

    def compute_distance_matrix(self, test_data):
        """
        Computes the distance matrix between the test data and the training data.

        Args:
            test_data (ndarray): The test data for classification.

        Returns:
            ndarray: The distance matrix between the test data and training data.
        """
        dist_matrix = np.zeros((test_data.shape[0], len(self.data)))

        # Precompute the distance matrix
        for i in range(len(test_data)):
            for j in range(len(self.data)):
                dist_matrix[i, j] = self.dtw_distance(test_data[i], self.data[j])

        return dist_matrix

    def dtw_distance(self,time_serie1, time_serie2):
        """
        Computes the DTW distance between two time series x and y.

        Args:
            x (list): The first time series.
            y (list): The second time series.

        Returns:
            float: The DTW distance between x and y.
        """
        l1, l2 = len(time_serie1), len(time_serie2)
        cost_matrix = np.zeros((l1 + 1, l2 + 1))
        cost_matrix[1:, 0] = np.inf
        cost_matrix[0, 1:] = np.inf
        cost_matrix[0, 0] = 0

        for i in range(1, l1 + 1):
            for j in range(1, l2 + 1):
                cost = self.distance(time_serie1[i - 1], time_serie2[j - 1])
                cost_matrix[i, j] = cost + min(
                    cost_matrix[i - 1, j],
                    cost_matrix[i, j - 1],
                    cost_matrix[i - 1, j - 1]
                )

        return np.sqrt(cost_matrix[-1, -1]) / (l1 + l2)
    
    def distance(self, x, y):
        """
        This function computes the euclidean distance between two vectors
        """
        dist = 0.0
        for i in range(len(x)):
            diff = (x[i] - y[i])
            dist += diff * diff
        return dist

"""
Perform testing on a specific user's data using a given model.

Args:
    user_id (int): The ID of the user.
    dataset (ndarray): The dataset containing all the time series data.
    labels (ndarray): The corresponding labels for the dataset.
    model (object): The classification model.
    LIMIT (int, optional): The maximum number of instances per user. Defaults to 100.

Returns:
    tuple: A tuple containing the accuracy score and predictions for the test data.
"""
def test(user_id, data, labels, model, cross_validation_mode,LIMIT):
    # Split the dataset into train and test sets using Leave One Out Cross Validation
    train_set,train_labels,test_labels,test_set = shared_func.Leave_One_Out(user_id, data, labels,cross_validation_mode, LIMIT)

    # Fit the model to the train set and make predictions on the test set
    model.fit(train_set, train_labels)
    predictions = model.predict(test_set)

    # Calculate the accuracy score and return it along with the predictions
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy, predictions


"""
Perform validation on the dataset using a given model.

Args:
    dataset (Dataset): An instance of the Dataset class.
    labels (list): The labels corresponding to each sequence in the dataset.
    model (object): The classification model.
    LIMIT (int, optional): The maximum number of instances per user. Defaults to 100.

Returns:
    tuple: A tuple containing the list of accuracy scores and predictions for each user.
"""
def validation(dataset, labels, model, cross_validation_mode, LIMIT):
    data = dataset
    labels = np.array(labels)
    accuracies = []
    predictions = []

    for user_id in range(10):
        accuracy, prediction = test(user_id, data, labels, model, cross_validation_mode,LIMIT)
        accuracies.append(accuracy)
        predictions.append(prediction)
        print("The user score {}: {}".format(user_id+1, accuracies[-1]))

    return accuracies, predictions


""""""""""""""""""""
#      Launch     #
""""""""""""""""""""
if __name__ == '__main__':
  # user independent cross_validation_mode=1
  # user dependent cross_validation_mode=0
  cross_validation_mode = 1

  dataset = shared_func.Dataset()
  
  model = DTW(n_neighbors=3)
  data = shared_func.PCA_variance(dataset.data)
  data = np.array(dataset.data, dtype=object)

  accuracies, predictions = validation(data, dataset.labels, model, cross_validation_mode, LIMIT=100)
  # plot the confusion matrix of the 3-NN model 
  shared_func.plot_conf_mat(np.array(dataset.labels), predictions, LIMIT=100)