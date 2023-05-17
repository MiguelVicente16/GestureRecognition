import numpy as np
from sklearn.metrics import accuracy_score
import shared_func
from dollarpy import Recognizer, Template, Point


def validation(data, labels, cross_validation_mode, LIMIT):
  labels = np.array(labels)
  accuracies = []
  predictions_per_users = []
  for user_id in range(10):
    train_set,train_labels,test_labels,test_set = shared_func.Leave_One_Out(user_id, data, labels,cross_validation_mode,LIMIT)
    templates = []
    for i,d in enumerate(train_set):
      templates.append(Template(str(labels[i]), d))
    recognizer = Recognizer(templates)
    predictions=[]
    for t in test_set:
      result = recognizer.recognize(t)
      predictions.append(int(result[0]))
    acc = accuracy_score(test_labels, predictions)
    print("The user score {}: {}".format(user_id+1, acc))
    accuracies.append(acc)
    predictions_per_users.append(predictions)
  return accuracies, predictions_per_users



""""""""""""""""""""
#      Launch     #
""""""""""""""""""""
if __name__ == '__main__':
     
    # user independent cross_validation_mode=1
    # user dependent cross_validation_mode=0
    cross_validation_mode = 1
    
    dataset = shared_func.Dataset() 
    data = shared_func.PCA_variance(dataset.data)
    accuracies2, predictions = validation(data, dataset.labels, cross_validation_mode, LIMIT=100)
    # plot the confusion matrix of the 3-NN model 
    shared_func.plot_conf_mat(np.array(dataset.labels), predictions, LIMIT=100)