from pandas import read_csv
from pandas import set_option
from numpy import set_printoptions
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from pickle import dump
from pickle import load

#Loading dataset from File
filename = "MedicareProcessed.csv"
names = ['Gender', 'Entity Code', 'State Label', 'Provider Label', 'Place of Service', 'HCPCS Code', 'HCPCS Drug Indicator', 'Number of Services', 'Number of Medicare Beneficiaries', 'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 'Average Medicare Payment Amount']
dataset = read_csv(filename, names=names)
array = dataset.values
X = array[:,0:11]
Y = array[:,11]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)
num_folds = 10
scoring = 'neg_mean_squared_error'
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
Errored = mean_squared_error(Y_validation, predictions)
print (Errored)

#Save the model
OutputModelFilename='finalized_model.sav'
dump(model, open(OutputModelFilename,'wb'))



