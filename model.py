import pickle

import pandas as pd
from keras import layers, Sequential
from keras.regularizers import l2
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)
df = pd.concat([pd.read_csv("ateros.csv"), pd.read_csv("data_syn1.csv"), pd.read_csv("data_syn2.csv")])
print(df)
# label = LabelEncoder()
# for col in df.columns.values.tolist():
#     if pd.api.types.is_string_dtype(df[col].dtype):
#         df[col] = label.fit_transform(df[col])
#         le_name_mapping = dict(zip(label.classes_, label.transform(label.classes_)))
#         print(le_name_mapping)
print(df)
# df = df.sort_values(by=["HeartDisease"])
y = df["Progress"]
X = df.drop("Progress", axis=1)

# smote = SMOTE(random_state=42, k_neighbors=2)
# nearmiss = NearMiss(n_neighbors=2)

# X, y = nearmiss.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(y_train)
print(y_test)
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)


# print('Imbalanced: {}'.format(Counter(y_train)))
# print('Balanced: {}'.format(Counter(y_train_smote)))

print(X_train)
print(y_train)

# mean = X_train.mean(axis=0)
# std = X_train.std(axis=0)
# X_train = (X_train - mean) / std

# Define the model architecture
model = Sequential()
model.add(layers.Dense(64, input_shape=(len(X.columns.values),), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2)
model.save("my_model.h5")
# workers=10,                    use_multiprocessing=True)
print(model.summary())

y_pred = [round(float(x)) for x in model.predict(X_test)]
y_pred_train = [round(float(x)) for x in model.predict(X_train)]
fprnn, tprnn, threshnn = roc_curve(y_test, y_pred, pos_label=1)

cm_test = confusion_matrix(y_pred, y_test)
cm_train = confusion_matrix(y_pred_train, y_train)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print(cm_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp.plot()
plt.show()
print(cm_train)
print('Accuracy for training set for Neural network = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set forNeural network = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('modell loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#

def test(model):
    model.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = model.predict(X_test)
    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = model.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    fpr2, tpr2, thresh2 = roc_curve(y_test, y_pred, pos_label=1)

    print(cm_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    disp.plot()
    disp.ax_.set(xlabel=type(model).__name__ + ' Confusion Matrix')
    plt.show()
    print('Accuracy for training set for model {} = {}'.format(type(model).__name__,
                                                               (cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set for model {} = {}'.format(type(model).__name__,
                                                           (cm_test[0][0] + cm_test[1][1]) / len(y_test)))
    with open(type(model).__name__ + '.pkl', 'wb') as files:
        pickle.dump(model, files)
    return fpr2, tpr2



fpr1, tpr1 = test(svm.SVC(probability=True))
fpr2, tpr2 = test(GaussianNB())
fpr3, tpr3 = test(DecisionTreeClassifier())
fpr4, tpr4 = test(RandomForestClassifier(n_estimators=20))
fpr5, tpr5 = test(XGBClassifier(use_label_encoder=False))

plt.style.use('seaborn')

random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='SVC')
plt.plot(fpr2, tpr2, linestyle='--', color='green', label='Naive Bayes')
plt.plot(fpr3, tpr3, linestyle='--', color='yellow', label='Decision Tree')
plt.plot(fpr4, tpr4, linestyle='--', color='black', label='Random Forest')
plt.plot(fpr5, tpr5, linestyle='--', color='purple', label='XGB')
plt.plot(fprnn, tprnn, linestyle='--', color='pink', label='Neural Network')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC', dpi=300)
plt.show()

with open(type(model).__name__ + '.pkl', 'wb') as files:
    pickle.dump(model, files)



# X_train1 = X_train
# xx = pd.DataFrame(X_train)
# xx["pred"] = [round(float(x)) for x in model.predict(xx)]
# xx["ypred"] = y_train
# print(xx)
# print(len(xx[xx["ypred"]!=xx["pred"]]))
#
# X_train = xx[xx["ypred"]==xx["pred"]]
# y_train = xx["pred"]
# X_train = X_train.drop(["ypred", "pred"], axis=1)
# print(X_train)
# print(y_train)
# model = keras.Sequential()
# model.add(layers.Dense(17, input_dim=17, input_shape=(17,), activation="relu"))
# # model.add(Dense(17, activation='relu'))
# # model.add(Dense(17, activation='relu'))
# # model.add(Dropout(0.25))
# # model.add(Dense(17, activation='relu'))
# # model.add(Dense(17, activation='relu'))
# # model.add(Dropout(0.25))
# # model.add(Dense(17, activation='relu'))
# model.add(layers.Dense(17, activation='relu'))
# # model.add(layers.Dense(256, activation='relu'))
# # model.add(layers.Dense(512, activation='relu'))
# # model.add(layers.Dense(1024, activation='relu'))
#
# # model.add(Dropout(0.5))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(X_train, y_train, batch_size=1000, validation_split=0.2, epochs=200)
# # workers=10,                    use_multiprocessing=True)
# print(model.summary())
#
# # print(y_train)
# #
# # for idx, *row in X_train.itertuples():
# #     print(len(row))
# #     print([round(float(i)) for i in model.predict([row])][0])
# #     print(y_train.iloc[idx])
# #     if [round(float(i)) for i in model.predict([row])][0] == y_train.iloc[[idx]]:
# #         xx.append(X_train.iloc[idx])
# #         yy.append(y_train.iloc[idx])
# #
# # print(len(xx))
# # print(len(X_train))
# # print(len(yy))
#
# # disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
# # disp.plot()
# # plt.show()
