#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install xgboost


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import plot_roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
warnings.filterwarnings("ignore")
data= pd.read_csv("C:\\Users\\gebruiker\\Desktop\\10057 Data Scientist Case\\Case_Data_Scientist_Dataset.csv", delimiter=";")


# In[ ]:


data.head()


# In[ ]:


data.to_excel("C:\\Users\\gebruiker\\Desktop\\10057 Data Scientist Case\\Case_Data_Scientist_Dataset3.xlsx", index=False)


# In[ ]:


data.columns.values


# In[ ]:


data.info()


# In[ ]:


data['MultipleLines'].replace({'No phone service': 'No'}, inplace=True)
data['OnlineSecurity'].replace({'No internet service': 'No'}, inplace=True)
data['OnlineBackup'].replace({'No internet service': 'No'}, inplace=True)
data['DeviceProtection'].replace({'No internet service': 'No'}, inplace=True)
data['TechSupport'].replace({'No internet service': 'No'}, inplace=True)
data['StreamingTV'].replace({'No internet service': 'No'}, inplace=True)
data['StreamingMovies'].replace({'No internet service': 'No'}, inplace=True)


# In[ ]:


data['MultipleLines'].replace({'No phone service': 'No'}, inplace=True)


# In[ ]:


data['MonthlyCharges'] = data['MonthlyCharges'].replace({',': '.'}, regex=True)
data['MonthlyCharges'] = pd.to_numeric(data['MonthlyCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].replace({',': '.'}, regex=True)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


print('Number of duplicated values in training dataset: ', data.duplicated().sum())
data.drop_duplicates(inplace=True)


# In[ ]:


data = data[(data['TotalCharges'] >= data['MonthlyCharges'])  ]


# In[ ]:


data["gender"].fillna(data["gender"].mode()[0], inplace=True)


# In[ ]:


data["SeniorCitizen"].fillna(data["SeniorCitizen"].max(), inplace=True)


# In[ ]:


data['SeniorCitizen'].replace({1: 'Yes', 0: 'No'}, inplace=True)


# In[ ]:


data["Contract"].fillna(data["Contract"].mode()[0], inplace=True)


# In[ ]:


data["TotalCharges"] = data.groupby("InternetService")['TotalCharges'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


data.isnull().sum()


# In[ ]:


sns.boxplot(data['tenure'], color='skyblue')
plt.title("Box Plot before outlier removing")
plt.show()
def drop_outliers(df, field_name):
    q1 = df[field_name].quantile(0.25)
    q3 = df[field_name].quantile(0.75)
    iqr = q3 - q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    print(df[(df[field_name] < Lower_tail) | (df[field_name] > Upper_tail)].shape)
    df.drop(df[(df[field_name] < Lower_tail) | (df[field_name] > Upper_tail)].index, inplace=True)
drop_outliers(data, 'tenure')
sns.boxplot(data['tenure'], color='blue')
plt.title("Box Plot after outlier removing")
plt.show()


# In[ ]:


sns.boxplot(data['MonthlyCharges'], color='lightgreen')
plt.title("Box Plot before outlier removing")
plt.show()
def drop_outliers(df, field_name):
    q1 = df[field_name].quantile(0.25)
    q3 = df[field_name].quantile(0.75)
    iqr = q3 - q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    print(df[(df[field_name] < Lower_tail) | (df[field_name] > Upper_tail)].shape)
    df.drop(df[(df[field_name] < Lower_tail) | (df[field_name] > Upper_tail)].index, inplace=True)
drop_outliers(data, 'MonthlyCharges')
sns.boxplot(data['MonthlyCharges'], color='green')
plt.title("Box Plot after outlier removing")
plt.show()


# In[ ]:


sns.boxplot(data['TotalCharges'], color='pink')
plt.title("Box Plot before outlier removing")
plt.show()
def drop_outliers(df, field_name):
    q1 = df[field_name].quantile(0.25)
    q3 = df[field_name].quantile(0.75)
    iqr = q3 - q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    df.drop(df[(df[field_name] < Lower_tail) | (df[field_name] > Upper_tail)].index, inplace=True)
drop_outliers(data, 'TotalCharges')
sns.boxplot(data['TotalCharges'], color='red')
plt.title("Box Plot after outlier removing")
plt.show()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data['MultipleLines'].replace({'No Phone Service': 'No'}, inplace=True)


# In[ ]:


custom_colors = ['#ff9999', '#66b3ff', '#99ff99', '#c2c2f0', '#ffb3e6']
service_columns = data.columns[data.columns.get_loc("PhoneService"):data.columns.get_loc("StreamingMovies")+1]
num_rows = (len(service_columns) + 1) // 2
fig, axes = plt.subplots(num_rows, 2, figsize=(16, 4 * num_rows))
axes = axes.flatten()
for i, column in enumerate(service_columns):
    print(data[column].unique())
    usage_counts = data[column].value_counts()
    axes[i].pie(usage_counts, labels=usage_counts.index, autopct='%1.1f%%', startangle=90, colors=custom_colors)
    axes[i].set_title(f"{column} Usage")
for i in range(len(service_columns), len(axes)):
    fig.delaxes(axes[i])
plt.suptitle("Service Usage Distribution", fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:


categorical_columns = [col for col in data.columns if data[col].dtype == 'object' and col not in ['Churn', 'customerID']]
color_discrete_map = {'Yes': 'red', 'No': 'blue'}
num_rows = -(-len(categorical_columns) // 2)
fig = make_subplots(rows=num_rows, cols=2, subplot_titles=categorical_columns)
for i, col in enumerate(categorical_columns):
    row = i // 2 + 1
    col_index = i % 2 + 1
    counts = data.groupby(['Churn', col]).size().reset_index(name='counts')
    for churn_val in data['Churn'].unique():
        subset = counts[counts['Churn'] == churn_val]
        show_legend = i == 0 
        trace = go.Bar(
            x=subset[col],
            y=subset['counts'],
            name=f'Churn: {churn_val}',
            marker_color=color_discrete_map.get(churn_val, 'black'),
            text=subset['counts'],
            textposition='auto',
            showlegend=show_legend)
        fig.add_trace(trace, row=row, col=col_index)
    #annotation
    churned_counts = data[data['Churn'] == 'Yes'][col].value_counts()
    total_churned = sum(churned_counts)
    for category, count in churned_counts.items():
        percentage = count / total_churned * 100
        fig.add_annotation(
            go.layout.Annotation(
                text=f"{percentage:.2f}%",
                xref=f"x{col_index}",
                yref=f"y{col_index}",
                x=category, 
                y=-120, 
                showarrow=False,
                font=dict(size=10),
                xanchor='center'),
            row=row,
            col=col_index)
fig.update_layout(
    title_text='Categorical Variables vs Churn',
    barmode='group',
    height=650 * num_rows,
    template='plotly_white',
    legend=dict(x=1.02, y=0.5, font=dict(size=8)))
fig.show()


# In[ ]:


data['internetService']=data['InternetService'].copy()


# In[ ]:


data.head()


# In[ ]:


for i, column in enumerate(service_columns):
    data[column] = data[column].replace({'Yes': 1, 'DSL' : 1 , 'Fiber optic' : 1 , 'No': 0 })

# Calculate the average number of services consumers use
data['TotalServices'] = data[['PhoneService', 'MultipleLines', 'InternetService',
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']].sum(axis=1)
print(f"Average number of average_services use:")
print(data['TotalServices'].describe())
data = data.drop('TotalServices', axis=1)


# In[ ]:


data.head()


# In[ ]:


senior_citizens_df = data[data['SeniorCitizen'] == "Yes"]
service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
service_usage = {}
for column in service_columns:
    service_usage[column] = senior_citizens_df[column].sum()
sorted_service_usage = sorted(service_usage.items(), key=lambda x: x[1], reverse=True)
for service, count in sorted_service_usage:
    print(f"{service}: {count}")
services = [item[0] for item in sorted_service_usage]
counts = [item[1] for item in sorted_service_usage]
plt.figure(figsize=(10, 8))
bars = plt.barh(services, counts, color='green')
plt.xlabel('Number of Senior Citizens')
plt.ylabel('Services')
plt.title('Popularity of Services Among Senior Citizens')
plt.gca().invert_yaxis()  # To display the most popular service at the top
plt.tight_layout()
#annotation
for bar, count in zip(bars, counts):
    plt.text(bar.get_width() + 0.1,  # Place the text just after the end of the bar
             bar.get_y() + bar.get_height() / 2, 
             f'{count}', 
             va='center', 
             color='black',
             fontsize=10)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))
churn_counts = data["Churn"].value_counts()
colors = ['#0000FF', '#1e90ff']
explode = [0.2, 0]
plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode, shadow=True)
plt.axis('equal')
plt.title('Churn Distribution')
plt.show()
#The dataset is unbalanced 


# In[ ]:


plt.figure(figsize=(10, 8))
sns.set(font_scale=1.5)
sns.kdeplot(
    data=data, x="tenure", hue="Churn",
    fill=True, common_norm=False, palette="Paired",
    alpha=0.7, linewidth=0)
plt.xlabel('Tenure')
plt.ylabel('Density')
plt.show()


# In[ ]:


plt.figure(figsize=(22, 6))
mean, median = data["MonthlyCharges"].mean(), data["MonthlyCharges"].median()
graph = sns.histplot(data["MonthlyCharges"])
graph.axvline(mean, c='red', label='mean')
graph.axvline(median, c='green', label='median')
plt.legend()
print(f"mean of MonthlyCharges: {mean:.2f}")
print(f"median of MonthlyCharges: {median:.2f}")
plt.show()


# In[ ]:


plt.figure(figsize=(40,10))
plt.subplot(121)
sns.histplot(data=data, x="TotalCharges", hue="Churn", multiple="stack")
plt.show()


# In[ ]:


data = data.drop('customerID', axis=1)


# In[ ]:


data = data.drop('InternetService', axis=1)


# In[ ]:


data.head()


# In[ ]:


resp = (data['Churn'] == 'Yes').sum()  # Sum of churned customers
total = data.shape[0]  # Total number of customers
percent = round((resp / total) * 100, 2)  # Percentage of churned customers

print(resp, 'churned out of', total, 'customers.')
print('Churned: ' + str(percent) + '%')


# In[ ]:


data['gender'].replace({'Male': 1 , 'Female': 0}, inplace=True)
data['SeniorCitizen'].replace({'Yes': 1 , 'No': 0}, inplace=True)
data['Partner'].replace({'Yes': 1 , 'No': 0}, inplace=True)
data['Dependents'].replace({'Yes': 1 , 'No': 0}, inplace=True)
data['PaperlessBilling'].replace({'Yes': 1 , 'No': 0}, inplace=True)


# In[ ]:


data.head()


# In[ ]:


numeric_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
categorical_cols = ['Contract', 'PaymentMethod', 'internetService']
scaler = StandardScaler()
encoder = LabelEncoder()
scaled_numeric = pd.DataFrame(scaler.fit_transform(data[numeric_cols]), columns=numeric_cols)
scaled_numeric.index = data.index
encoded_categorical = data[categorical_cols].apply(encoder.fit_transform)
encoded_categorical.index = data.index 
untouched_cols = data.columns.difference(numeric_cols + categorical_cols)
untouched_data = data[untouched_cols]
scaled_ds = pd.concat([scaled_numeric, encoded_categorical, untouched_data], axis=1)
scaled_ds.head()


# In[ ]:


dataa=data.copy()


# In[ ]:


scaled_ds.info()


# In[ ]:


scaled_ds['Churn'] = scaled_ds['Churn'].map({'Yes': 1, 'No': 0})


# In[ ]:


cor = scaled_ds.corr()
response_corr = cor.loc[['Churn'], :]
response_corr_sorted = response_corr.sort_values(by='Churn', axis=1)
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(data=response_corr_sorted, orient='h', palette='Set2')
for i, v in enumerate(response_corr_sorted.values.flatten()):
    bar_plot.text(v + 0.01, i, str(round(v, 2)), color='black', va='center')
plt.show()


# In[ ]:


X= scaled_ds.drop('Churn', axis=1).values
y =scaled_ds['Churn'].values


# In[ ]:


print('Preprocessed Data:')
print(scaled_ds.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[ ]:


X_train.shape, y_train.shape, X_test.shape,  y_test.shape


# In[ ]:


sm = SMOTE()
X_bal, y_bal = sm.fit_resample(X_train, y_train)


# In[ ]:


y_test.shape


# In[ ]:


X_bal.shape, y_bal.shape, X_test.shape,  y_test.shape


# In[ ]:


print(X_bal, y_bal, X_test,  y_test)


# In[ ]:


resp = (y_bal == 1).sum()  
total = y_bal.shape[0]  
percent = round((resp / total) * 100, 2) 

print(resp, 'churned out of', total, 'total')
print('Churned: ' + str(percent) + '%')


# In[ ]:


print(X_bal.shape)  
print(y_bal.shape)


# In[ ]:


y_bal


# In[ ]:


print(X_bal.shape)  # should print (n_samples, n_features)
print(y_bal.shape)  # should print (n_samples,)


# In[ ]:


classifier_xgb = XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000)
classifier_xgb.fit(X_bal, y_bal)
prediction = classifier_xgb.predict(X_test)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cross_val_score_mean = cross_val_score(classifier_xgb, X_bal, y_bal, cv=cv, scoring='roc_auc').mean()
print("Cross Validation Score : ", '{0:.2%}'.format(cross_val_score_mean))
roc_auc = roc_auc_score(y_test, prediction)
print("ROC_AUC Score : ", '{0:.2%}'.format(roc_auc))
acdt=accuracy_score(y_test, classifier_xgb.predict(X_test))
print("accuracy_score:" , '{0:.2%}'.format(acdt))


# In[ ]:


plot_roc_curve(classifier_xgb, X_test, y_test)
plt.title('ROC_AUC_Plot')
plt.show()
cm = confusion_matrix(y_test,classifier_xgb.predict(X_test))
names = ['True Neg','False Pos','False Neg','True Pos']
counts = [value for value in cm.flatten()]
percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
print(classification_report(y_test,classifier_xgb.predict(X_test)))


# In[ ]:


classifier_rf = RandomForestClassifier(max_depth = 4,random_state = 0)
classifier_rf.fit(X_bal, y_bal)
prediction = classifier_rf.predict(X_test)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cross_val_score_mean = cross_val_score(classifier_rf, X_bal, y_bal, cv=cv, scoring='roc_auc').mean()
print("Cross Validation Score : ", '{0:.2%}'.format(cross_val_score_mean))

roc_auc = roc_auc_score(y_test, prediction)
print("ROC_AUC Score : ", '{0:.2%}'.format(roc_auc))
acdt=accuracy_score(y_test, classifier_rf.predict(X_test))
print("accuracy_score:" , '{0:.2%}'.format(acdt))


# In[ ]:


# Plot ROC_AUC_Plot
plot_roc_curve(classifier_rf, X_test, y_test)
plt.title('ROC_AUC_Plot')
plt.show()

cm = confusion_matrix(y_test,classifier_rf.predict(X_test))
names = ['True Neg','False Pos','False Neg','True Pos']
counts = [value for value in cm.flatten()]
percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
print(classification_report(y_test,classifier_rf.predict(X_test)))


# In[ ]:


model_svm = SVC(kernel='linear') 
model_svm.fit(X_bal, y_bal)
prediction = model_svm.predict(X_test)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cross_val_score_mean = cross_val_score(model_svm, X_bal, y_bal, cv=cv, scoring='roc_auc').mean()
print("Cross Validation Score : ", '{0:.2%}'.format(cross_val_score_mean))

roc_auc = roc_auc_score(y_test, prediction)
print("ROC_AUC Score : ", '{0:.2%}'.format(roc_auc))
acdt=accuracy_score(y_test, model_svm.predict(X_test))
print("accuracy_score:" , '{0:.2%}'.format(acdt))


# In[ ]:


plot_roc_curve(model_svm, X_test, y_test)
plt.title('ROC_AUC_Plot')
plt.show()

cm = confusion_matrix(y_test,model_svm.predict(X_test))
names = ['True Neg','False Pos','False Neg','True Pos']
counts = [value for value in cm.flatten()]
percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
print(classification_report(y_test,model_svm.predict(X_test)))


# In[ ]:


model_lg = LogisticRegression(max_iter=120,random_state=0, n_jobs=20)
model_lg.fit(X_bal, y_bal)
prediction = model_lg.predict(X_test)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cross_val_score_mean = cross_val_score(model_lg, X_bal, y_bal, cv=cv, scoring='roc_auc').mean()
print("Cross Validation Score : ", '{0:.2%}'.format(cross_val_score_mean))

roc_auc = roc_auc_score(y_test, prediction)
print("ROC_AUC Score : ", '{0:.2%}'.format(roc_auc))
acdt=accuracy_score(y_test, model_lg.predict(X_test))
print("accuracy_score:" , '{0:.2%}'.format(acdt))


# In[ ]:


plot_roc_curve(model_lg, X_test, y_test)
plt.title('ROC_AUC_Plot')
plt.show()

cm = confusion_matrix(y_test,model_lg.predict(X_test))
names = ['True Neg','False Pos','False Neg','True Pos']
counts = [value for value in cm.flatten()]
percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
print(classification_report(y_test,model_lg.predict(X_test)))  


# In[ ]:


model_kn = KNeighborsClassifier(n_neighbors=9, leaf_size=20)
model_kn.fit(X_bal, y_bal)
prediction = model_kn.predict(X_test)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cross_val_score_mean = cross_val_score(model_kn, X_bal, y_bal, cv=cv, scoring='roc_auc').mean()
print("Cross Validation Score : ", '{0:.2%}'.format(cross_val_score_mean))
roc_auc = roc_auc_score(y_test, prediction)
print("ROC_AUC Score : ", '{0:.2%}'.format(roc_auc))
acdt=accuracy_score(y_test, model_kn.predict(X_test))
print("accuracy_score:" , '{0:.2%}'.format(acdt))


# In[ ]:


plot_roc_curve(model_kn, X_test, y_test)
plt.title('ROC_AUC_Plot')
plt.show()
cm = confusion_matrix(y_test,model_kn.predict(X_test))
names = ['True Neg','False Pos','False Neg','True Pos']
counts = [value for value in cm.flatten()]
percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
print(classification_report(y_test,model_kn.predict(X_test)))


# In[ ]:


model_dt = DecisionTreeClassifier(random_state = 1000,max_depth = 4,min_samples_leaf = 1)
model_dt.fit(X_bal, y_bal)
prediction = model_dt.predict(X_test)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cross_val_score_mean = cross_val_score(model_dt, X_bal, y_bal, cv=cv, scoring='roc_auc').mean()
print("Cross Validation Score : ", '{0:.2%}'.format(cross_val_score_mean))

roc_auc = roc_auc_score(y_test, prediction)
print("ROC_AUC Score : ", '{0:.2%}'.format(roc_auc))
acdt=accuracy_score(y_test, model_dt.predict(X_test))
print("accuracy_score:" , '{0:.2%}'.format(acdt))   


# In[ ]:


plot_roc_curve(model_dt, X_test, y_test)
plt.title('ROC_AUC_Plot')
plt.show()

cm = confusion_matrix(y_test,model_dt.predict(X_test))
names = ['True Neg','False Pos','False Neg','True Pos']
counts = [value for value in cm.flatten()]
percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
print(classification_report(y_test,model_dt.predict(X_test)))


# In[ ]:


model_names = ['Gradient Boosting ', 'Random Forest ', 'SVM', 'Logistic Regression ', 'KNN', 'Decision Tree ']

accuracy = [79.11, 75.42, 72.94, 74.55, 69.42, 70.86]  
Roc_auc = [75.67, 76.60, 74.94, 75.34, 62.36, 73.07]  
F1 = [80.00, 77.00, 75.00, 76.00, 71.0, 73.0]  
result_df = pd.DataFrame({'Accuracy': accuracy, 'Roc_auc': Roc_auc, 'F1': F1}, index=model_names)
result_df = result_df.sort_values(by='Accuracy', ascending=False)
styled_df = result_df.style.background_gradient(cmap='Blues').set_precision(2).set_table_styles(
    [{
        'selector': 'th',
        'props': [('font-size', '15pt'), ('background-color', 'lightblue')]},
        {
        'selector': 'td',
        'props': [('font-size', '13pt')]}])
styled_df


# In[ ]:


numerical_cols = dataa.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols.remove('SeniorCitizen') 
categorical_cols = dataa.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')  
scaler = StandardScaler()
dataa[numerical_cols] = scaler.fit_transform(dataa[numerical_cols])
le = LabelEncoder()
dataa['Churn'] = le.fit_transform(dataa['Churn'])
dataa = pd.get_dummies(dataa, columns=categorical_cols)
X = dataa.drop('Churn', axis=1)
y = dataa['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


sm = SMOTE()
X_bal, y_bal = sm.fit_resample(X_train, y_train)


# In[ ]:


classifier_rf = RandomForestClassifier(max_depth = 4,random_state = 0)
classifier_rf.fit(X_bal, y_bal)
prediction = classifier_rf.predict(X_test)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cross_val_score_mean = cross_val_score(classifier_rf, X_bal, y_bal, cv=cv, scoring='roc_auc').mean()
print("Cross Validation Score : ", '{0:.2%}'.format(cross_val_score_mean))

roc_auc = roc_auc_score(y_test, prediction)
print("ROC_AUC Score : ", '{0:.2%}'.format(roc_auc))
acdt=accuracy_score(y_test, classifier_rf.predict(X_test))
print("accuracy_score:" , '{0:.2%}'.format(acdt))


# In[ ]:


importances = classifier_rf.feature_importances_
# Creating a Series with feature importances and feature names
feature_importances = pd.Series(importances, index=X_bal.columns.tolist()).sort_values(ascending=False)

# Plotting the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index, palette="rocket")
sns.despine()
plt.xlabel("Feature Importances")
plt.ylabel("Features")
plt.title("Most Important Features according to the Model with Randomforest")
plt.xlim(0, 1)
plt.yticks(fontsize=10)
plt.show()

