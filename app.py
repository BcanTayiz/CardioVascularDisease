import streamlit as st 
import pandas as pd
import numpy as np
import re
import seaborn as sns 
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import datetime
import random
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score
import streamlit_theme as stt
import streamlit.components.v1 as components
from datetime import date


random.seed(10)
stt.set_theme({'primary': '#1b3388'})
st.title("Cardiovascular Disease Alert")

st.write("Created by Barış Can Tayiz")

components.html("""
<div style="background-color:black;height:10px;border-radius:10px;margin-bottom:0px;">
</div><hr>""")

st.header("Variables")
st.write(
"""* Age | Objective Feature | age | int (days) \n
* Height | Objective Feature | height | int (cm) | \n
* Weight | Objective Feature | weight | float (kg) | \n
* Gender | Objective Feature | gender | categorical code | \n
* Systolic blood pressure | Examination Feature | ap_hi | int | \n
* Diastolic blood pressure | Examination Feature | ap_lo | int | \n
* Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal | \n
* Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal | \n
* Smoking | Subjective Feature | smoke | binary |\n
* Alcohol intake | Subjective Feature | alco | binary | \n
* Physical activity | Subjective Feature | active | binary | \n
* Presence or absence of cardiovascular disease | Target Variable | cardio | binary |"""
)

components.html("""
<div style="background-color:black;height:10px;border-radius:10px;margin-bottom:0px;">
</div><hr>""")

data = pd.read_csv('cardio_train.csv',sep=";")

data = data.drop('id',axis=1)

st.write(data.describe())

components.html("""
<div style="background-color:black;height:10px;border-radius:10px;margin-bottom:0px;">
</div><hr>""")

classifier_types = ("Random Forests","XGBClassifier","Neural Net")

classifier_name = st.sidebar.selectbox("Select Regressor",classifier_types)

st.header("Correlation Values of parameters")

cm = sns.light_palette("coral", as_cmap=True) 

st.write(data.corr().style.background_gradient(cmap=cm).set_precision(2))

X = data.drop('cardio',axis=1)
y = data['cardio']

x = X #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)


X = X.apply(pd.to_numeric)
y = y.apply(pd.to_numeric)






def parameter_ui(classifier_name):
    params = dict()
    if classifier_name == classifier_types[0]:
        n_estimators = st.sidebar.slider("n_estimators",11,99)
        criterion = st.sidebar.selectbox("criterion",("gini", "entropy"))
        max_depth = st.sidebar.slider("max_depth",1,100)
        min_samples_split = st.sidebar.slider("min_samples_split",2,100)
        min_samples_leaf = st.sidebar.slider("min_samples_leaf",1,100)
        max_features = st.sidebar.selectbox("criterion",("auto", "sqrt", "log2"))
        bootstrap = st.sidebar.selectbox("bootstrap",("True","False"))
        oob_score = st.sidebar.selectbox("oob_score",("True","False"))
        class_weight = st.sidebar.selectbox("class_weight",("balanced","balanced_subsample"))

        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        params["max_depth"] = max_depth
        params["min_samples_split"] = min_samples_split
        params["min_samples_leaf"] = min_samples_leaf
        params["max_features"] = max_features
        params["bootstrap"] = bootstrap
        params["oob_score"] = oob_score
        params["class_weight"] = class_weight
          

    elif classifier_name == classifier_types[1]:
        booster  = st.sidebar.selectbox("booster",("gbtree", "gblinear","dart"))
        verbosity  = st.sidebar.selectbox("verbosity",(0,1,2,3))
        nthread  = st.sidebar.slider("nthread ",1,100)
        eta = st.sidebar.slider("eta ",0,100,1)
        gamma  = st.sidebar.slider("gamma ",0,100)
        max_depth  = st.sidebar.slider("max_depth ",1,100)
        min_child_weight  = st.sidebar.slider("min_child_weight ",0,100)
        max_delta_step  = st.sidebar.slider("max_delta_step ",0,100)

        eta = eta / 100

        params["booster"] = booster
        params["verbosity"] = verbosity
        params["nthread"] = nthread
        params["eta"] = eta
        params["gamma"] = gamma
        params["max_depth"] = max_depth
        params["min_child_weight"] = min_child_weight
        params["max_delta_step"] = max_delta_step

    elif classifier_name == classifier_types[2]:
        hidden_layer_sizes = st.sidebar.slider("hidden_layer_sizes",1,50)
        activation = st.sidebar.selectbox("activation",("identity", "logistic","tanh","relu"))
        solver = st.sidebar.selectbox("solver",("lbfgs", "sgd", "adam"))
        alpha = st.sidebar.slider("alpha",0,100,1)
        batch_size = st.sidebar.slider("min_samples_leaf",1,100)
        learning_rate = st.sidebar.selectbox("learning_rate",("constant", "invscaling", "adaptive"))
        max_iter = st.sidebar.slider("max_iter",1,100)
        shuffle = st.sidebar.selectbox("shuffle",("True", "False"))

        alpha = alpha / 100

        params["hidden_layer_sizes"] = hidden_layer_sizes
        params["activation"] = activation
        params["solver"] = solver
        params["alpha"] = alpha
        params["batch_size"] = batch_size
        params["learning_rate"] = learning_rate
        params["max_iter"] = max_iter

    return params


params = parameter_ui(classifier_name)

def get_classifier(classifier_name,params):
    if classifier_name == classifier_types[0]:
         clf = RandomForestClassifier(n_estimators=params["n_estimators"], criterion=params["criterion"],max_depth=params["max_depth"]
         ,min_samples_split = params["min_samples_split"],min_samples_leaf = params["min_samples_leaf"],max_features = params["max_features"],
         oob_score = params['oob_score'],class_weight = params['class_weight'] )
    elif classifier_name == classifier_types[1]:
        clf =  XGBClassifier(booster=params["booster"], verbosity=params["verbosity"],nthread=params["nthread"],eta=params["eta"],gamma=params["gamma"],max_depth=params["max_depth"],
        min_child_weight = params["min_child_weight"],max_delta_step = params["max_delta_step"])
    elif classifier_name == classifier_types[2]:
        clf = MLPClassifier(hidden_layer_sizes=params["hidden_layer_sizes"],activation=params["activation"],solver=params["solver"],
        alpha=params["alpha"],batch_size=params["batch_size"],learning_rate=params["learning_rate"],max_iter = params["max_iter"])

    return clf

clf = get_classifier(classifier_name,params)

st.write(clf)   

#Classifier
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 42)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#MODEL IMPORTANCE
try:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.header("Feature Importance Figure")
    importances = clf.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importances):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    from matplotlib.pyplot import figure
    figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    pyplot.bar([data.iloc[:,x].name for x in range(len(importances))], importances)

    st.pyplot()
except:
    pass



st.header("Balanced Accuracy Score")
st.write("Balanced Accuracy Score shows the related accuracy level of predicted and real values. It should be close to 1 for best training results")
st.write(balanced_accuracy_score(y_test, y_pred))


st.header("Get your Cardiovascular Condition")

birthDay = st.text_input("Enter your Birthday as yyyy-mm-dd: ")
gender = st.text_input("Enter your gender, 1 women; 2 men: ")
height =st.text_input("Enter your height, as cm: ")
weight = st.text_input("Enter your weight, as kg: ")
ap_hi = st.text_input("Enter your Systolic blood pressure ap_hi: ")
ap_lo = st.text_input("Enter your Diastolic blood pressure ap_lo: ")
chol = st.text_input("Enter your Cholesterol level, 1: normal; 2: above normal;3: well above normal :  ")
gluc = st.text_input("Enter your Glucose level, 1: normal; 2: above normal;3: well above normal : ")
smoke = st.text_input("Enter your smoke, yes 1; no 0: ")
alco = st.text_input("Enter your alco, yes 1; no 0: ")
active = st.text_input("Enter your physical activity, yes 1; no 0: ")


today = pd.to_datetime(date.today())
birthDay = pd.to_datetime(birthDay)
days = int((today-birthDay).days)


entry_prediction = [int(days),int(gender),int(height),int(weight),int(ap_hi),
int(ap_lo),int(chol),int(gluc),int(smoke),int(alco),int(active)]




if len(entry_prediction) == 11 :


    df = pd.DataFrame(columns = x.columns)
    df = df.append(x)
    df.loc[x.shape[0]+1] = entry_prediction



    min_max_scaler = preprocessing.MinMaxScaler()
    df_scaled = min_max_scaler.fit_transform(df.tail(2))
    entryPred_df = pd.DataFrame(df_scaled)

 
    prediction = clf.predict_proba((entryPred_df.astype(float)))
    st.write(prediction)
    st.header("Probable {0} {1}".format("Cardiovascular Disease",'%{:,.2f}'.format(prediction[1][1])))








