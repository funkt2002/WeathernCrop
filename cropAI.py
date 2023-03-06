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


PATH='./'
FILENAME='Crops.csv'
data=pd.read_csv(PATH+FILENAME)
print(PATH+FILENAME )
crop=data['label'].unique()
data['label'].value_counts()
data.rename(columns={'N':'nitrogen','P':'phosphorus','K':'potassium','label':'crop'}, inplace=True)
data.head()

features = ['nitrogen','phosphorus','potassium','temperature','humidity','ph','rainfall']
target = ['crop']
X = data[features]
y = data[target]

print(crop)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.33)
mmscaler=MinMaxScaler()
X_train=mmscaler.fit_transform(X_train)
X_test=mmscaler.transform(X_test)
y_train = LabelEncoder().fit_transform(np.asarray(y_train).ravel())
y_test = LabelEncoder().fit_transform(np.asarray(y_test).ravel())
for ii, col in enumerate(features):
  print('{} (min,max): \t \t {:.2f} {:.2f}'.format(col,X_train[:,ii].min(),X_train[:,ii].max()))

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print('Training Accuracy: {:.2f}%, Test Accuracy: {:.2f}%'.format(metrics.accuracy_score(y_train,model.predict(X_train))*100,metrics.accuracy_score(y_test,model.predict(X_test))*100))

def get_predictions(x1,x2,x3,x4,x5,x6,x7):
    feature = mmscaler.transform(np.asarray([x1,x2,x3,x4,x5,x6,x7]).reshape((1,-1)))
    croptoplant = crop[model.predict(feature).item()]
    print('{} should grow very well under these conditions'.format(croptoplant.upper()))

N_input = widgets.IntSlider(value=50, min=0, max=150, step=1, description='N:')
P_input = widgets.IntSlider(value=50, min=0, max=150, step=1, description='P:')
K_input = widgets.IntSlider(value=50, min=0, max=150, step=1, description='K:')
temp_input = widgets.IntSlider(value=20, min=0, max=50, step=1, description='Temperature:')
hum_input = widgets.IntSlider(value=20, min=0, max=100, step=1, description='Humidity:')
ph_input = widgets.FloatSlider(value=5.5, min=0, max=14, step=0.1, description='pH:')
rain_input = widgets.FloatSlider(value=50, min=0, max=300, step=0.1, description='Rainfall:')


def on_button_clicked(b):
    get_predictions(N_input.value, P_input.value, K_input.value, 
                    temp_input.value, hum_input.value, ph_input.value, 
                    rain_input.value)

button = widgets.Button(description='Get Crop Recommendation')
button.on_click(on_button_clicked)
display(N_input, P_input, K_input, temp_input, hum_input, ph_input, rain_input, button)