import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import metrics 
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import ipywidgets as widgets
import warnings
from IPython.display import display
from time import strptime
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
#######This is an interface for the AI model###
PATH='./'
FILENAME='kenya_climate_data.csv'
data=pd.read_csv(PATH+FILENAME)
data.shape
data.info
data.describe().T
data.isnull().sum()
data.rename(str.strip,axis='columns',inplace=True)
data=data.rename(columns={'Rainfall - (MM)':'Rainfall','Month Average':'Month'})
data['day']=1

data=data.replace({'Jan Average':1,'Feb Average':2,'Mar Average':3,'Apr Average':4, 'May Average':5, 'Jun Average':6,
'Jul Average':7, 'Aug Average':8, 'Sep Average':9, 'Oct Average':10, 'Nov Average':11, 'Dec Average':12})
data['Year']=pd.to_datetime(data[['Year','Month','day']])
data=data.drop(columns=['Month','day'])
data=data.rename(columns={'Year':'datetime'})
print(data)
X = data['datetime'].astype(np.int64).values.reshape(-1, 1) / 10**9
y = data['Rainfall'].values.reshape(-1,1)
print('Shape of X:', X.shape)
print('Shape of y:', y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.scatter(X, y)
reg = LinearRegression()
reg.fit(X, y)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = reg.predict(X_range)

plt.plot(X_range, y_pred, color='red')


plt.xlabel('Date')
plt.ylabel('Rainfall (mm)')
plt.title('Monthly Rainfall in Kenya')
print(data)

last_date = data['datetime'].max()
future_dates = pd.date_range(last_date, periods=60, freq='M')


future_X = future_dates.astype(np.int64).values.reshape(-1, 1) / 10**9

future_y = model.predict(future_X)

plt.figure(figsize=(10,6))
plt.scatter(X, y)
plt.plot(X_range, y_pred, color='red')
plt.plot(future_X, future_y, color='green')
plt.xlabel('Date')
plt.ylabel('Rainfall (mm)')
plt.title('Monthly Rainfall in Kenya')
plt.show()