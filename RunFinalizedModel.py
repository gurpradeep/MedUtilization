from pandas import read_csv
from pandas import set_option
from numpy import set_printoptions
from sklearn.model_selection import train_test_split
from pickle import dump
from pickle import load

#Loading dataset from File
filename = "MedicareProcessed.csv"
names = ['Gender', 'Entity Code', 'State Label', 'Provider Label', 'Place of Service', 'HCPCS Code', 'HCPCS Drug Indicator', 'Number of Services', 'Number of Medicare Beneficiaries', 'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 'Average Medicare Payment Amount']
dataset = read_csv(filename, names=names)
array = dataset.values
X = array[:,0:11]
Y = array[:,11]
validation_size = 0.33
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)
#Run the model
loaded_model = load(open('finalized_model.sav', 'rb'))
#print(loaded_model.predict(X_validation))
result = loaded_model.score(X_validation, Y_validation)
print(result)
