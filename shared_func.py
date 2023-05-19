import numpy as np
import csv
from dollarpy import Point
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

#######################
### Aux functions ###
#######################

class Dataset:
    def __init__(self, domain, nr_model):
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
                            if nr_model == 0:
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
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []

    for user_id in range(10):

        for i in range(10):
            start = (user_id * 100) + (i * 10)
            end = (user_id * 100) + (i * 10 + 8)
            # Select 7 samples for the training set
            train_samples = data[start:end]
            
            train_set.extend(train_samples)
            train_labels.extend(labels[start:end])

            start = (user_id * 100) + (i * 10 + 8)
            end = (user_id * 100) + (i * 10 + 10)

            # Select 3 samples for the test set
            test_samples = data[start:end]
           
            test_set.extend(test_samples)
            test_labels.extend(labels[start:end])

    train_set = np.array(train_set, dtype = object)
    train_labels = np.array(train_labels)
    test_set = np.array(test_set,  dtype = object)
    test_labels = np.array(test_labels)
    return train_set, train_labels, test_set, test_labels

def PCA_variance(data): 
    pca_variances = []
    final_dataset = []
    for user_data in data:
        pca = PCA(n_components=2)
        new_data = pca.fit_transform(user_data)
        twod_data = [Point(*row) for row in new_data]
        final_dataset.append(twod_data)
        pca_variances.append(pca.explained_variance_ratio_)
    #data = np.array(final_dataset, dtype='object')
    print("The average explained variance ratio is over all the dataset: ", np.mean(pca_variances, axis=0))
    return final_dataset


def Leave_One_Out(user_id, dataset, labels, cross_validation_mode, LIMIT):
    # Split the dataset into train and test sets
    if cross_validation_mode == 1:
        indexes = range(LIMIT * user_id, LIMIT * user_id + LIMIT)
        train_set = np.delete(dataset, indexes)
        train_labels = np.delete(labels, indexes)
        test_labels = labels[indexes]
        test_set = dataset[indexes]
    else:
        subset_size = 10
        subset_indexes = []
        for subset_index in range(LIMIT // subset_size):
            start_index = user_id * LIMIT + subset_index * subset_size
            end_index = start_index + subset_size
            subset_indexes.extend(range(start_index, end_index))
        train_set = np.delete(dataset, subset_indexes, axis=0)
        train_labels = np.delete(labels, subset_indexes)
        test_labels = labels[subset_indexes]
        test_set = dataset[subset_indexes]
    
    return train_set,train_labels,test_labels,test_set

def plot_conf_mat(true_labels, pred_labels):
    print("The confusion matrix of the model:")
    indexes = range(0, len(pred_labels))
    conf_mat = confusion_matrix(true_labels[indexes], pred_labels[indexes])
    df_cm = pd.DataFrame(conf_mat, index=range(10), columns=range(10))
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix of the model")
    plt.show()
    return
