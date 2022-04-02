# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss  # returns the error rate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def classes_to_numeric(x):
    if x == 'good':
        return 1
    if x == 'bad':
        return 0
last_column = "status"
data_set_train = pd.read_csv("training.csv")
data_set_train[last_column] = data_set_train[last_column].apply(classes_to_numeric)  # change to numeric values

########################
data_set_test = pd.read_csv("test.csv")
data_set_test[last_column] = data_set_test[last_column].apply(classes_to_numeric)  # change to numeric values
########################

print("---------------------------------------Statistical Description------------------------------------------------------")
print(data_set_train.describe().to_string())
##################################################################################
corrMatrix1 = data_set_train.corr()
##to drow the correlation coeffetion between features using colors
# fig1, ax1 = plt.subplots(figsize=(100, 100))  # Sample figsize in inches
# sn.heatmap(corrMatrix1, annot=True, linewidths=.5, ax=ax1, linecolor='grey')  # seaborn
# plt.title("Correlation")  # Correlation plot
# plt.show()
# data_set_train[data_set_train.columns[1:]].corr()[data_set_train][:]
# print(corrMatrix1.to_string())

# for x in corrMatrix1:
#   s1 = str(corrMatrix1[x].sort_values(ascending=False))
#  print(f"----------The correlation {x}------------------------------------\n")
# print(s1)
s1 = corrMatrix1[last_column].sort_values(ascending=False)
############################################################################
print(f"----------The correlation {last_column}------------------------------------\n")

print(s1.to_string())
############################################################################
#chosing the target class of training and test data as y , and input classes as x
df1_data_train_set_edit = pd.DataFrame(data_set_train)
df1_data_train_set_edit = df1_data_train_set_edit.drop(
    columns=['sfh', 'submit_email', 'ratio_intErrors', 'ratio_intRedirection', 'ratio_nullHyperlinks', 'or', 'space',
             'iframe', 'port', 'right_clic', 'char_repeat'])

X_train = df1_data_train_set_edit.drop(columns=[last_column])
y_train = df1_data_train_set_edit[last_column]
####
df1_data_test_set_edit = pd.DataFrame(data_set_test)
df1_data_test_set_edit = df1_data_test_set_edit.drop(
    columns=['sfh', 'submit_email', 'ratio_intErrors', 'ratio_intRedirection', 'ratio_nullHyperlinks', 'or','space',
             'iframe','port','right_clic','char_repeat'])

X_test = df1_data_test_set_edit.drop(columns=[last_column])
y_test = df1_data_test_set_edit[last_column]
print("---------------------------------------Density Plot------------------------------------------------------")
X_train_KNN = X_train[['google_index','digits_url_ratio','domain_in_title','phish_hints',
                       'page_rank','www','hyperlinks']]
#Write code of density plot.....
# plt.title('Density Plot Of status ')
# plt.ylabel('status')
# plt.xlabel('oogle_index')
# plt.plot(y_train,X_train['google_index'])
# plt.legend()
# plt.show()
# plt.ylabel('status')
# plt.xlabel('page_rank')
# plt.plot(y_train,X_train['page_rank'])
# plt.legend()
# plt.show()
# plt.ylabel('status')
# plt.xlabel('All Other Features')
# plt.plot(y_train,X_train)
# plt.legend()
# plt.show()

##################################################################################

print("---------------------------------------Linear Regression------------------------------------------------------")

model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.score(X_test, y_test)  # the accuracy is: "R^2"
print(f'-Performance measure (Accuracy) for predictions using linearRegression: {predictions}')

print(
    "---------------------------------------Logistic Regression------------------------------------------------------")
##################################
model = LogisticRegression(solver='lbfgs', max_iter=1000)

model.fit(X_train, y_train)

yl_pred = model.predict(X_test)
error_rate = zero_one_loss(y_test, yl_pred)
print("Accuracy: ", metrics.accuracy_score(y_test, yl_pred))
print("Precision: ", metrics.precision_score(y_test, yl_pred))
print("Recall: ", metrics.recall_score(y_test, yl_pred))
print(f'Error Rate= {error_rate}')

print(f'*LogisticReg_classification_report:\n{metrics.classification_report(y_test, yl_pred)}')
print(f'*LogisticReg_confusion_matrix:\n{metrics.confusion_matrix(y_test, yl_pred)}')
print("------------------------------------KNN---------------------------------------------------------")
# write code of KNN ......
X_train_KNN = X_train[['google_index','digits_url_ratio','domain_in_title','phish_hints',
                       'page_rank','www','hyperlinks']]
X_test_KNN = X_test[['google_index','digits_url_ratio','domain_in_title','phish_hints',
                       'page_rank','www','hyperlinks']]
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_KNN, y_train)
yl_pred = knn.predict(X_test_KNN)
error_rate = zero_one_loss(y_test, yl_pred)
print("Accuracy: ", metrics.accuracy_score(y_test, yl_pred))
print("Precision: ", metrics.precision_score(y_test, yl_pred))
print("Recall: ", metrics.recall_score(y_test, yl_pred))
print(f'Error Rate= {error_rate}')
print(f'*KNN_classification_report:\n{metrics.classification_report(y_test, yl_pred)}')
print(f'*KNN_confusion_matrix:\n{metrics.confusion_matrix(y_test, yl_pred)}')
print("-----------------------------------------Decision tree----------------------------------------------------")

# write code of Decision ......

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)

print(f'Accuracy_DecisionTree: {acc}')

tree.export_graphviz(model, out_file='graph_decision_tree.dot'
                     , feature_names=X_train.columns,
                     class_names=['bad', 'good'],
                     label='all',
                     rounded=True,
                     filled=True)
print("-----------------------------------------Naive bayes----------------------------------------------------")
# write code of Naive ......
X_train_Naive = X_train[['google_index','digits_url_ratio','domain_in_title','phish_hints',
                       'page_rank','www','hyperlinks']]
X_test_Naive = X_test[['google_index','digits_url_ratio','domain_in_title','phish_hints',
                       'page_rank','www','hyperlinks']]
gnb = GaussianNB()
gnb.fit(X_train_Naive, y_train)
yl_pred = gnb.predict(X_test_Naive)
error_rate = zero_one_loss(y_test, yl_pred)
print("Accuracy: ", metrics.accuracy_score(y_test, yl_pred))
print("Precision: ", metrics.precision_score(y_test, yl_pred))
print("Recall: ", metrics.recall_score(y_test, yl_pred))
print(f'Error Rate= {error_rate}')

print(f'*Naive bayes_classification_report:\n{metrics.classification_report(y_test, yl_pred)}')
print(f'*Naive bayes_confusion_matrix:\n{metrics.confusion_matrix(y_test, yl_pred)}')



print("----------------------------------------- K-means ----------------------------------------------------")
# write code of K-means ......
X_train_Kmeans = X_train[['google_index','digits_url_ratio']]
X_test_Kmeans = X_test[['google_index','digits_url_ratio']]
kmeans = KMeans(2)
kmeans.fit(X_train_Kmeans, y_train)
yl_pred = kmeans.predict(X_test_Kmeans)
error_rate = zero_one_loss(y_test, yl_pred)
print("Accuracy: ", metrics.accuracy_score(y_test, yl_pred))
print("Precision: ", metrics.precision_score(y_test, yl_pred, average='micro'))
print("Recall: ", metrics.recall_score(y_test, yl_pred, average='micro'))
print(f'Error Rate= {error_rate}')

print(f'*K-means_classification_report:\n{metrics.classification_report(y_test, yl_pred)}')
print(f'*K-means_confusion_matrix:\n{metrics.confusion_matrix(y_test, yl_pred)}')


print("-----------------------------------------SVM----------------------------------------------------")

# write code of SVM ......
X_train_svm = X_train[['google_index','digits_url_ratio',
                       'page_rank','www']]
X_test_svm = X_test[['google_index','digits_url_ratio',
                       'page_rank','www']]
svc = SVC(degree=1)

svc.fit(X_train_svm, y_train)

# xx = np.linspace(x)

y_predict_svm = svc.predict(X_test_svm)
conf = metrics.confusion_matrix(y_test, y_predict_svm)

print(conf)
acc = metrics.accuracy_score(y_test, y_predict_svm)
print(acc)

print("--------")
all_report_svc = metrics.classification_report(y_test, y_predict_svm)
print(all_report_svc)
