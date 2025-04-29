import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

def constructData(uncorrected, corrected):
    uncorrectedData = []
    for fileName in uncorrected:
        file = open(fileName, "r")
        reader = csv.reader(file)

        data = []
        for row in reader:
            data.append(row)

        # Populates an 2d array where each row is a cone and each column is a timestamp
        cones = []
        numCones = len(data[0])
        for i in range(0, 50):
            for j in range(0, numCones):
                if len(cones) < numCones:
                    cones.append([])
                cones[j].append(float(data[i][j]))
        
        uncorrectedData = uncorrectedData + cones

    correctedData = []
    for fileName in corrected:
        file = open(fileName, "r")
        reader = csv.reader(file)

        data = []
        for row in reader:
            data.append(row)

        # Populates an 2d array where each row is a cone and each column is a timestamp
        cones = []
        numCones = len(data[0])
        for i in range(0, 50):
            for j in range(0, numCones):
                if len(cones) < numCones:
                    cones.append([])
                cones[j].append(float(data[i][j]))
        
        correctedData = correctedData + cones
    
    features = np.array(uncorrectedData)
    label = []
    for i in range(0, len(uncorrectedData)):
        if uncorrectedData[i] == correctedData[i]:
            label.append(0) # one for no correction
        else: # 0 for needing correction
            label.append(1)
    return features, np.array(label)

np.set_printoptions(threshold=np.inf)
X, y = constructData(["Uncorrected_1.csv", "Uncorrected_2.csv", "Uncorrected_3.csv", "Uncorrected_4.csv", "Uncorrected_5.csv", "Uncorrected_6.csv", "Uncorrected_7.csv", "Uncorrected_8.csv", "Uncorrected_9.csv", "Uncorrected_10.csv"], ["Corrected_1.csv", "Corrected_2.csv", "Corrected_3.csv", "Corrected_4.csv", "Corrected_5.csv", "Corrected_6.csv", "Corrected_7.csv", "Corrected_8.csv", "Corrected_9.csv", "Corrected_10.csv"])



# mask = ~np.isnan(X).any(axis=1)
# X = X[mask]
# y = y[mask]

# y_true = y
# y_pred = np.zeros(X.shape[0])
# for i in range(0, X.shape[0]):
#     diff = 0
#     for j in range(X.shape[1] - 1, 1, -1):
#         diff = diff + X[i][j] - X[i][j-1]
#     if diff > 2:
#         y_pred[i] = 1


# print("Accuracy:", accuracy_score(y_true, y_pred))
# print(classification_report(y_true, y_pred))













# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# model = LogisticRegression(class_weight='balanced', max_iter=1000)  # increase max_iter just in case
# model.fit(X_train, y_train)

# y_pred_proba = model.predict_proba(X_test)[:, 1]  # get probabilities for class 1
# y_pred = model.predict(X_test)
# y_pred = (y_pred_proba > 0).astype(int)  # adjust the threshold to 0.3, for instance

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# cm = confusion_matrix(y_test, y_pred)
# print(cm)


