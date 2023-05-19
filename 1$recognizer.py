import numpy as np
from sklearn.metrics import accuracy_score
import shared_func
from dollarpy import Recognizer, Template
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score




class OneDollar:
   
  def __init__(self):

    self.recognizer = None
    return

  def fit(self,train_set,labels):
    templates = []
    for i,d in enumerate(train_set):
      templates.append(Template(str(labels[i]), d))
    self.recognizer = Recognizer(templates)

  def predict(self,test_set):
    predictions=[]
    for t in test_set:
      result = self.recognizer.recognize(t)
      predictions.append(int(result[0]))
    return predictions
  

def test(train_set, train_labels, test_set, model):

  model.fit(train_set,train_labels)
  predictions = model.predict(test_set)

  return predictions


def validation(data, labels, cross_validation_mode, model, LIMIT):
  labels = np.array(labels)
  accuracies = []
  predictions_per_users = []
  for user_id in range(10):
    train_set,train_labels,test_labels,test_set = shared_func.Leave_One_Out(user_id, data, labels,cross_validation_mode,LIMIT)
    predictions = test(train_set, labels, test_set, model)
    acc = accuracy_score(test_labels, predictions)
    print("The user score {}: {}".format(user_id+1, acc))
    accuracies.append(acc)
    predictions_per_users.append(predictions)
  return 


def model_test(model, test_set_data, test_set_labels):
    
    predictions = model.predict(test_set_data)
    accuracy = accuracy_score(test_set_labels, predictions)
    print("The model score is: {}".format(accuracy))
    return accuracy, predictions


""""""""""""""""""""
#      Launch     #
""""""""""""""""""""
if __name__ == '__main__':
     
  # user independent cross_validation_mode=1
  # user dependent cross_validation_mode=0
  cross_validation_mode = 1

  domain = 4
  nr_model = 1
  dataset = shared_func.Dataset(domain, nr_model) 

  model = OneDollar()

  data = shared_func.PCA_variance(dataset.data)

  train_set_data, train_set_labels, test_set_data, test_set_labels = shared_func.split_data(data, dataset.labels)

  validation(train_set_data, train_set_labels, cross_validation_mode, model ,LIMIT=80)

  predictions = model_test(model,test_set_data,train_set_labels)

  shared_func.plot_conf_mat(test_set_labels, np.array(predictions))
  print("Model $P dollar measure from the confusion matrixes")
  print("The precision", precision_score(train_set_labels, predictions, average=None))
  print("The f1-score", f1_score(train_set_labels, predictions, average=None))
  print("The recall", recall_score(train_set_labels, predictions, average=None))
  print()
     