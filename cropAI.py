from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn import metrics 
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import warnings


warnings.filterwarnings('ignore')
%matplotlib inline


PATH = './'
FILENAME = 'Crops.csv'
data = read_csv(PATH + FILENAME)

# Prepare data
data.rename(columns={'N':'nitrogen','P':'phosphorus','K':'potassium','label':'crop'}, inplace=True)
features = ['nitrogen','phosphorus','potassium','temperature','humidity','ph','rainfall']
target = ['crop']
X = data[features]
y = data[target]
crop = data['crop'].unique()
print(crop)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


le = LabelEncoder()
y_train = le.fit_transform(np.asarray(y_train).ravel())
y_test = le.fit_transform(np.asarray(y_test).ravel())


model = RandomForestClassifier()
model.fit(X_train, y_train)


y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_accuracy = metrics.accuracy_score(y_train, y_pred_train) * 100
test_accuracy = metrics.accuracy_score(y_test, y_pred_test) * 100
print(f'Training Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
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