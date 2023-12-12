import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

#load dataset
df = pd.read_csv("output_cluster_kmeansv3.csv", index_col=0)

#drop less informative cloumns
df = df.drop(columns=['category_x','subcategory', 'name', 'brand', 'brand_url', 'codCountry',
                           'variation_0_color', 'variation_0_image',
                           'variation_1_color',
                           'variation_1_image', 'url', 'image_url', 'model', 'id', 'image_variation',
                           'color_variation', 'likes_count_x'])

print(df['cluster'].value_counts())


# Splitting into training (70%) and temporary (30%)
train_df, temp_df = train_test_split(df, stratify = df['cluster'], test_size=0.3, random_state=42)

# Splitting temp_df into testing (50%) and validation (50%)
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# create target variables for classification
y_train = train_df['cluster']
y_test = test_df['cluster']
y_val = val_df['cluster']

#drop target variable form training dataset
train_df = train_df.drop(['cluster'], axis=1)
test_df = test_df.drop(['cluster'], axis=1)
val_df = val_df.drop(['cluster'], axis=1)

#Normalise dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df)
X_test_scaled = scaler.transform(test_df)
X_val_scaled = scaler.transform(val_df)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
# Count the occurrences of each class in y_train_resampled
class_counts = np.bincount(y_train_resampled)

# If your class labels are different, adjust accordingly
for class_label, count in enumerate(class_counts):
    print(f"Class {class_label}: {count} samples")

#Get the optmim K value by cross validation score
cv_scores = []
cv_scores_std = []
k_range = range(1, 135, 5)
for i in k_range:
    clf = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(clf, X_train_resampled, y_train_resampled, scoring='accuracy', cv=KFold(n_splits=10, shuffle=True))
    # print(scores)
    cv_scores.append(scores.mean())
    cv_scores_std.append(scores.std())
# Plot the relationship
# plt.figure(figsize=(15,10))
plt.errorbar(k_range, cv_scores, yerr=cv_scores_std, marker='x', label='Accuracy')
plt.ylim([0.1, 1.1])
plt.xlabel('$K$')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

#find optimum K value by grid search
parameter_grid = {'n_neighbors': range(1, 135, 5)}
knn_clf = KNeighborsClassifier()
gs_knn = GridSearchCV(knn_clf, parameter_grid, scoring='accuracy', cv=KFold(n_splits=10, shuffle=True))
gs_knn.fit(X_train_resampled, y_train_resampled)
# print('Best K value: ', gs_knn.best_params_['n_neighbors'])
# print('The accuracy: %.4f\n' % gs_knn.best_score_)
# Got the statistics
cv_scores_means = gs_knn.cv_results_['mean_test_score']
cv_scores_stds = gs_knn.cv_results_['std_test_score']
# Plot the relationship
plt.figure(figsize=(15,10))
plt.errorbar(k_range, cv_scores_means, yerr=cv_scores_stds, marker='o', label='gs_knn Accuracy') # gs_knn
plt.errorbar(k_range, cv_scores, yerr=cv_scores_std, marker='x', label='manual Accuracy') # manual
plt.ylim([0.1, 1.1])
plt.xlabel('$K$')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

knn_classifier_resampled = KNeighborsClassifier(n_neighbors=11)  # You can adjust n_neighbors as needed
knn_classifier_resampled.fit(X_train_resampled, y_train_resampled)

# Predict on the validation set
y_val_pred = knn_classifier_resampled.predict(X_val_scaled)

# Predict on the test set
y_test_pred = knn_classifier_resampled.predict(X_test_scaled)

# Compute confusion matrices
confusion_matrix_val = confusion_matrix(y_val, y_val_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)

# Calculate precision
precision_validation = precision_score(y_val, y_val_pred, average='weighted')  # 'weighted' considers class imbalance
precision_test = precision_score(y_test, y_test_pred, average='weighted')  # 'weighted' considers class imbalance

# Calculate recall
recall_validation = recall_score(y_val, y_val_pred, average='weighted')  # 'weighted' considers class imbalance
recall_test = recall_score(y_test, y_test_pred, average='weighted')  # 'weighted' considers class imbalance

print("Precision for validation set:")
print(precision_validation)

print("Precision for Testing set:")
print(precision_test)

print("Recall for validation set")
print(recall_validation)

print("Recall for Testing set set")
print(recall_test)

print("Confusion Matrix for Validation Set:")
print(confusion_matrix_val)

print("\nConfusion Matrix for Test Set:")
print(confusion_matrix_test)