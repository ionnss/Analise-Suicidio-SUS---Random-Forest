import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load dataframe for prediction model
dataframe = pd.read_csv('df_atualizada_2.csv')
df = dataframe.drop('CAUSABAS', axis=1)

# Define dependent (y) and independent (X) variables
X = df.drop('suicidio', axis=1)
y = df['suicidio']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define simplified parameter grid for GridSearchCV
param_grid = {
    'max_depth': [5, 10, 15, None],
    'min_samples_leaf': [1, 2, 4]
}

# Create a base model with class weights
clf = RandomForestClassifier(random_state=42, class_weight={0: 1, 1: 10})

# Instantiate GridSearchCV with StratifiedKFold
cv = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                           cv=cv, n_jobs=-1, verbose=2)

# Fit GridSearchCV to the training data
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters
print(f"Best parameters: {grid_search.best_params_}")

# Make predictions on the test set using the best model
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# Evaluate performance
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Visualize cross-validation results
results = pd.DataFrame(grid_search.cv_results_)

# Create heatmap of results
scores_matrix = results.pivot(index='param_max_depth', columns='param_min_samples_leaf', values='mean_test_score')

plt.figure(figsize=(10, 8))
plt.title("GridSearchCV Results")
sns.heatmap(scores_matrix, annot=True, cmap="viridis")
plt.xlabel("min_samples_leaf")
plt.ylabel("max_depth")
plt.show()
