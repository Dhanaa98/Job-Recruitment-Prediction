# **Job Recruitment Prediction** 

## Introduction
This project involves analyzing recruitment data to build Classification predictive models for hiring decisions. The primary goal is to evaluate different machine learning algorithms and compare their performance based on various metrics.
In this project, Logistic regression, Random forest classifier, Decision tree classifier and Support vector machine models were implemented and their performance were compared.
This type of a predictive model can be crucial for an organization as it streamlines the recruitment process by identifying the most suitable candidates, reducing hiring costs, and improving overall hiring efficiency.

## Data Preprocessing
- The dataset was loaded and checked for missing values and duplicates.
- No missing values or duplicates were found.
- The data was split into features (X) and target (y).
- The dataset was further split into training and testing sets using an 80-20 split.

## Model Building and Evaluation
### 1. Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
```
### 2. Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(n_estimators=100)
rfc_model.fit(X_train, y_train)
```
### 3. Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
dtc_model = DecisionTreeClassifier()
dtc_model.fit(X_train, y_train)
```
### 4. Support Vector Machine
```python
from sklearn import svm
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)
```
## Model Comparison

![table](https://github.com/Dhanaa98/Job-Recruitment-Prediction/assets/171159250/b3e85921-a9bd-4f7c-a893-c1f1f6b3581e)

## Conclusion

- **Logistic Regression:** Achieved decent accuracy and moderate R2 scores but might benefit from more iterations or data scaling.
- **Random Forest Classifier:** Performed exceptionally well with high accuracy and R2 scores, indicating it captures the patterns in the data effectively.
- **Decision Tree Classifier:** Also showed high accuracy but might be prone to overfitting, as suggested by the perfect training accuracy.
- **Support Vector Machine:** Balanced performance with good accuracy and moderate R2 scores, proving to be a robust choice.
  
Overall, the **Random Forest Classifier** stands out as the best-performing model in this analysis, providing a good balance of accuracy and generalizability.  
Performance of these models can further be enhanced through feature engineering, hyperparameter tuning, and ensemble methods to optimize predictive accuracy and robustness.

For any questions or further assistance, please feel free to contact me.

Dhananjaya Mudunkotuwa  
dhananjayamudunkotuwa1998@gmail.com 
