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

#####Loading dataset from File#####################
filename = "MedicareProcessed.csv"
#names = ['National Provider Identifier', 'Organization Name', 'Credentials', 'Gender', 'Entity Code', 'City', 'Zip Code', 'State Code', 'Country Code', 'Provider Type', 'Medicare Participation', 'Place of Service', 'HCPCS Code', 'HCPCS Description', 'HCPCS Drug Indicator', 'Number of Services', 'Number of Medicare Beneficiaries', 'Number of Medicare Beneficiary/Day Services', 'Average Medicare Allowed Amount', 'Standard Deviation of Medicare Allowed Amount', 'Average Submitted Charge Amount', 'Standard Deviation of Submitted Charge Amount', 'Average Medicare Payment Amount', 'Standard Deviation of Medicare Payment Amount']
names = ['Gender', 'Entity Code', 'State Label', 'Provider Label', 'Place of Service', 'HCPCS Code', 'HCPCS Drug Indicator', 'Number of Services', 'Number of Medicare Beneficiaries', 'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 'Average Medicare Payment Amount']
dataset = read_csv(filename, names=names)

####Correlarions between datasets##################
#print(dataset.ix[:5, :10])
#corelations = dataset.corr(method='pearson').iloc[:-1,-1]
#print(dataset.head(5))
#description = dataset.describe()
#print(description)
#corelations = dataset.corr(method='pearson').iloc[:-1,-1]
#correlations = dataset.corr(method='pearson')
#print (correlations)

####Skews between datasets##################
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

#####Evaluate using a train and a test set########
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
#scoring = 'neg_mean_absolute_error'
#scoring = 'neg_mean_squared_error'
scoring = 'r2'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean(), results.std())

