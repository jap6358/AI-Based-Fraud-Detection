import pandas as pd
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import sklearn.metrics as mt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


all_performances = pd.DataFrame()
list_clf_name = []
list_pred = []
list_model = []

def fit_model(model, X_train, y_train):
    X_model = model.fit(X_train,y_train)
    return X_model

def add_list(name, model, y_pred):
    global list_clf_name, list_pred, list_model, list_x_test
    list_clf_name.append(name)
    list_model.append(model)
    list_pred.append(y_pred)
def add_all_performances(name, precision, recall, f1_score, AUC):
    global all_performances
    models = pd.DataFrame([[name, precision, recall, f1_score, AUC]],
                         columns=["model_name","precision", "recall", "f1_score", "AUC"])
    
    all_performances = pd.concat([all_performances, models], ignore_index=True)
    all_performances = all_performances.drop_duplicates()
      
    
def calculate_scores(X_train, X_test, y_train, y_test, y_pred, name, model):
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = mt.precision_score(y_test,y_pred)
    recall = mt.recall_score(y_test,y_pred)
    f1_score= mt.f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    
    add_list(name, model, y_pred)
    add_all_performances(name, precision, recall, f1_score, AUC)
    #print(all_performances.sort_values(by=['f1_score'], ascending=False))
    

def model_performance(model, X_train, X_test, y_train, y_test, technique_name):

    name= model.__class__.__name__+"_"+technique_name
    x_model = fit_model(model, X_train, y_train)
    y_pred = x_model.predict(X_test)
    print("***** "+ name +" DONE *****")

    calculate_scores(X_train, X_test, y_train, y_test, y_pred, name, model)
    
    
def display_all_confusion_matrices(y_test):
    column = 4
    row = int(all_performances["model_name"].count()/4)
    f, ax = plt.subplots(row, column, figsize=(20,10), sharey='row')
    ax = ax.flatten()

    for i in range(column*row):
        cf_matrix = confusion_matrix(y_test, list_pred[i])
        disp = ConfusionMatrixDisplay(cf_matrix)
        disp.plot(ax=ax[i], xticks_rotation=45)
        disp.ax_.set_title(list_clf_name[i]+"\nAccuracy:{accuracy:.4f}\nAUC:{auc:.4f}"
                           .format(accuracy= accuracy_score(y_test, list_pred[i]),auc= roc_auc_score(y_test, list_pred[i])),
                             fontsize=14)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i!=0:
            disp.ax_.set_ylabel('')


    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust()
    f.colorbar(disp.im_)
    plt.show()
if __name__ == "__main__":

    print("Loading dataset...")

    # Load dataset
    data = pd.read_csv("creditcard/creditcard.csv")

    # Split features and target
    X = data.drop("Class", axis=1)
    y = data["Class"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    from sklearn.linear_model import LogisticRegression

    print("Training models...")


    lr_model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model_performance(lr_model, X_train, X_test, y_train, y_test, "Basic")


    rf_model = RandomForestClassifier(
         n_estimators=100,
         random_state=42,
         class_weight='balanced')

    model_performance(rf_model, X_train, X_test, y_train, y_test, "Ensemble")


    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    xgb_model = XGBClassifier(
          n_estimators=100,
          max_depth=5,
          learning_rate=0.1,
          scale_pos_weight=scale_pos_weight,
          eval_metric='logloss',
          use_label_encoder=False)
    model_performance(xgb_model, X_train, X_test, y_train, y_test, "Boosting")
    print("\nModel Performance Summary:")
    print(all_performances)
