import numpy as np
import csv
from dollarpy import Point
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import os

#######################
### Aux functions ###
#######################

class Dataset:
    def __init__(self, domain, std):
        """
        Initializes the Dataset class.

        Args:
            domain (int): The domain of the dataset.
            std (int): Control variable to standarize the data or not
        """
        # Initialize the dataset structure
        self.dataset = [[[] for _ in range(10)] for _ in range(10)]  # A 2D list to store user data by subject and digit
        self.data = []  # List to store all user data
        self.labels = []  # List to store corresponding class labels
        domain4 = ['Cone', 'Cuboid', 'Cylinder','CylindricalPipe','Hemisphere','Pyramid','RectangularPipe','Sphere','Tetrahedron','Toroid']

        for subject_id in range(1, 11):
            for digit in range(0, 10):
                for repetition_nr in range(1, 11):
                    if domain == 1:
                        filepath = "Datasets_CSV/Domain{}_csv/Subject{}-{}-{}.csv".format(domain, subject_id, digit,
                                                                                     repetition_nr)
                    else:
                        filepath = "Datasets_CSV/Domain{}_csv/Subject{}-{}-{}.csv".format(domain, subject_id, domain4[digit],
                                                                                     repetition_nr)
                    try:
                        with open(filepath, "r") as f:
                            reader = csv.reader(f)
                            # Skip header row
                            next(reader)
                            lines = [line for line in reader]

                            class_id = digit  # The class ID is the digit itself
                            user_id = subject_id  # The user ID is the subject ID
                            user_data = []  # List to store data for a specific user

                            # Process each line of the CSV file
                            for row in lines:
                                if row:
                                    float_row = [float(val) for val in row[0:3]]  # Convert string values to floats
                                    user_data.append(float_row)

                            # Add the user data to the dataset structure and append to the overall data and labels lists
                            self.dataset[user_id - 1][class_id].append(user_data)
                            self.data.append(user_data)
                            self.labels.append(class_id)
                            if std == 0:
                                self.data = self.standardize_data_columns()
                    except IOError as e:
                        print("Unable to read dataset file {}!\n".format(filepath))
    
    def standardize_data_columns(self):
        """
        Standardizes the columns of each matrix in the data array.
        """
        if len(self.data) == 0:
           return

        # Standardize each matrix in the data array
        for j,matrix in enumerate(self.data):
            matrix = np.array(matrix)  # Convert to NumPy array for easier manipulation

            # Compute the mean and standard deviation for each column
            column_means = np.mean(matrix, axis=0)
            column_std = np.std(matrix, axis=0)

            # Standardize each column of the matrix
            
            for i in range(3):
                column_means = np.mean(matrix[:, i], axis=0)
                column_std = np.std(matrix[:, i], axis=0)
                matrix[:, i] = (matrix[:, i] - column_means) / column_std
            self.data[j] = matrix
            
        return self.data


def split_data(data, labels):
    """
    Split the data and labels into training and test sets.

    Args:
        data (list): The data to be split.
        labels (list): The corresponding labels.

    Returns:
        tuple: A tuple containing the training set, training labels, test set, and test labels.
    """
    train_set = []  # List to store training set samples
    train_labels = []  # List to store training set labels
    test_set = []  # List to store test set samples
    test_labels = []  # List to store test set labels

    for user_id in range(10):
        for i in range(10):
            start = (user_id * 100) + (i * 10)
            end = (user_id * 100) + (i * 10 + 7)
            # Select 7 samples for the training set
            train_samples = data[start:end]
            train_set.extend(train_samples)
            train_labels.extend(labels[start:end])

            start = (user_id * 100) + (i * 10 + 7)
            end = (user_id * 100) + (i * 10 + 10)
            # Select 3 samples for the test set
            test_samples = data[start:end]
            test_set.extend(test_samples)
            test_labels.extend(labels[start:end])

    train_set = np.array(train_set, dtype=object)
    train_labels = np.array(train_labels)
    test_set = np.array(test_set, dtype=object)
    test_labels = np.array(test_labels)
    
    return train_set, train_labels, test_set, test_labels

def PCA_variance(data):
    """
    Perform Principal Component Analysis (PCA) on the given data and calculate the variance ratios.

    Args:
        data (list): The data to perform PCA on.

    Returns:
        list: A list of transformed datasets after PCA.
    """
    pca_variances = []  # List to store the variance ratios after PCA
    final_dataset = []  # List to store the transformed datasets after PCA

    for user_data in data:
        pca = PCA(n_components=2)  # Create a PCA object with 2 components
        new_data = pca.fit_transform(user_data)  # Perform PCA on the user_data
        twod_data = [Point(*row) for row in new_data]  # Convert the transformed data to a list of Point objects
        final_dataset.append(twod_data)  # Add the transformed data to the final_dataset list
        pca_variances.append(pca.explained_variance_ratio_)  # Get the variance ratios and append to pca_variances list

    print("The average explained variance ratio is over all the dataset: ", np.mean(pca_variances, axis=0))

    return final_dataset  # Return the transformed datasets after PCA


def Leave_One_Out(user_id, dataset, labels, cross_validation_mode, LIMIT):
    """
    Perform leave-one-out cross-validation on the dataset.

    Args:
        user_id (int): The ID of the user to leave out.
        dataset (ndarray): The dataset.
        labels (ndarray): The corresponding class labels.
        cross_validation_mode (int): The cross-validation mode.
        LIMIT (int): The limit of instances per user.

    Returns:
        tuple: A tuple containing the train set, train labels, test labels, and test set.
    """
    if cross_validation_mode == 1:
        # User-dependent cross-validation
        indexes = range(LIMIT * user_id, LIMIT * user_id + LIMIT)
        train_set = np.delete(dataset, indexes)  # Remove the test set instances from the dataset
        train_labels = np.delete(labels, indexes)  # Remove the test set labels
        test_labels = labels[indexes]  # Get the test set labels
        test_set = dataset[indexes]  # Get the test set instances
    else:
        # User-independent cross-validation
        subset_size = 10  # Size of each subset
        subset_indexes = []
        for subset_index in range(LIMIT // subset_size):
            start_index = user_id * LIMIT + subset_index * subset_size
            end_index = start_index + subset_size
            subset_indexes.extend(range(start_index, end_index))
        train_set = np.delete(dataset, subset_indexes, axis=0)  # Remove the test set instances from the dataset
        train_labels = np.delete(labels, subset_indexes)  # Remove the test set labels
        test_labels = labels[subset_indexes]  # Get the test set labels
        test_set = dataset[subset_indexes]  # Get the test set instances
    
    return train_set, train_labels, test_labels, test_set


def plot_conf_mat(true_labels, pred_labels, domain, nr_model, cross_validation_mode):
    """
    Plot and save the confusion matrix.

    Args:
        true_labels (array-like): The true class labels.
        pred_labels (array-like): The predicted class labels.
        domain (int): The domain.
        nr_model (int): The model number.
        cross_validation_mode (int): The cross-validation mode.

    Returns:
        None
    """
    print("The confusion matrix of the model:")
    conf_mat = confusion_matrix(true_labels, pred_labels)
    df_cm = pd.DataFrame(conf_mat, index=range(10), columns=range(10))
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix of the model")

    # Get the current directory
    current_dir = os.getcwd()

    file_name = "confusion_matrix_{}_{}_{}.png".format(domain, nr_model, cross_validation_mode)

    if current_dir and file_name:
        save_path = os.path.join(current_dir, file_name)  # Create the full save path
        plt.savefig(save_path)  # Save the figure to the specified path
        print(f"Figure saved at {save_path}")
    else:
        plt.show()  # If save path is not available, display the figure

    return

