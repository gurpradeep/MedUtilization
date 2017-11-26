from pandas import read_csv
from pandas import set_option
from pandas import to_numeric
from pandas.tools.plotting import scatter_matrix
from matplotlib import pyplot
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

#Loading dataset from File
filename = "MedicareProcessed.csv"
#############################################
#names = ['National Provider Identifier', 'Organization Name', 'Credentials', 'Gender', 'Entity Code', 'City', 'Zip Code', 'State Code', 'Country Code', 'Provider Type', 'Medicare Participation', 'Place of Service', 'HCPCS Code', 'HCPCS Description', 'HCPCS Drug Indicator', 'Number of Services', 'Number of Medicare Beneficiaries', 'Number of Medicare Beneficiary/Day Services', 'Average Medicare Allowed Amount', 'Standard Deviation of Medicare Allowed Amount', 'Average Submitted Charge Amount', 'Standard Deviation of Submitted Charge Amount', 'Average Medicare Payment Amount', 'Standard Deviation of Medicare Payment Amount']
names = ['Gender', 'Entity Code', 'State Label', 'Provider Label', 'Place of Service', 'HCPCS Code', 'HCPCS Drug Indicator', 'Number of Services', 'Number of Medicare Beneficiaries', 'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 'Average Medicare Payment Amount']
dataset = read_csv(filename, names=names)
#print(dataset.ix[:5, :10])
#corelations = dataset.corr(method='pearson').iloc[:-1,-1]
set_option('display.width', 10000)
set_option('precision', 3)
#print(dataset.head(5))
#description = dataset.describe()
#print(description)

#corelations = dataset.corr(method='pearson').iloc[:-1,-1]
#correlations = dataset.corr(method='pearson')
#print (correlations)

#Skew
#skew = dataset.skew()
#print (skew)

##################visualization##########################
#dataset.hist()
#fig = pyplot.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(correlations, vmin=-1, vmax=1)
#fig.colorbar(cax)
#dataset.plot(kind='density', subplots=True, layout=(5,5), sharex=False)
#Axes = scatter_matrix(dataset)
#[pyplot.setp(item.yaxis.get_majorticklabels(), 'size', 5) for item in Axes.ravel()]
##x ticklabels
#[pyplot.setp(item.xaxis.get_majorticklabels(), 'size', 5) for item in Axes.ravel()]
##y labels
#[pyplot.setp(item.yaxis.get_label(), 'size', 5) for item in Axes.ravel()]
##x labels
#[pyplot.setp(item.xaxis.get_label(), 'size', 5) for item in Axes.ravel()]
#pyplot.show()

####Feature Selection###########################
array = dataset.values
X = array[:,0:11]
Y = array[:,11]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)
num_folds = 10
scoring = 'neg_mean_squared_error'
# ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB',AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF',RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET',ExtraTreesRegressor())])))
modelresults =[]
modelnames=[]
# evaluate each model in turn
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=7)
    cv_results = cross_val_score(model,X_train, Y_train, cv=kfold, scoring=scoring)
    modelresults.append(cv_results)
    modelnames.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(modelresults)
ax.set_xticklabels(modelnames)
pyplot.show()