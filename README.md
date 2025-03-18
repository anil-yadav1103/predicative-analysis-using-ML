# predicative-analysis-using-ML

COMPANY:CODTECH SOLUTION

NAME:BOMMANONI ANIL

INTERN ID:CT12PDW

DOMAIN:DATA ANALYSIS

DURATION:8 WEEKS

MENTOR:NEELA SANTHOSH

Predictive Analysis Using Machine Learning
Predictive analysis is a branch of data analytics that focuses on using historical data, statistical algorithms, and machine learning (ML) techniques to predict future outcomes. By analyzing patterns in data, predictive models provide valuable insights that help organizations make informed decisions, improve efficiency, and reduce risks.

In this project, we performed predictive analysis using machine learning classification models to predict outcomes based on a dataset. For demonstration, we used the popular Titanic dataset, which contains information about the passengers aboard the Titanic ship, including features such as age, gender, class, and survival status. Our goal was to build a model that could predict whether a passenger survived based on these features.

The first step in predictive analysis is data preprocessing and feature selection. This involves cleaning the dataset by handling missing values, selecting the most relevant features, and encoding categorical data. In our case, features like passenger class (pclass), age, gender (sex), fare, and embarkation town (embark_town) were selected. Categorical variables such as sex and embark_town were converted into numerical format using one-hot encoding.

After preparing the dataset, we split the data into training and testing sets using the train_test_split function from the sklearn library. The training set was used to train the machine learning model, while the testing set was used to evaluate its performance.

For the machine learning model, we used the Random Forest Classifier, an ensemble learning method known for its high accuracy and ability to handle both numerical and categorical data. It builds multiple decision trees and combines their outputs to improve prediction accuracy and prevent overfitting.

To improve the model's performance, we also applied feature scaling using StandardScaler, which helps normalize the data and ensures that all features contribute equally to the model training.

Once the model was trained, we evaluated its performance using metrics such as Accuracy Score, Classification Report, and Confusion Matrix. The accuracy score provided the percentage of correctly predicted outcomes, while the classification report offered a detailed breakdown of precision, recall, and F1-score for each class (survived or not survived). The confusion matrix visually showed the true positive, true negative, false positive, and false negative predictions.

For visualization and better understanding, we plotted the feature importance using the matplotlib library. This helped us identify which features had the most significant impact on the model’s predictions.

Tools and Libraries Used:
Python: Programming language for data processing and model building.
Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
Seaborn: For loading datasets and visualization.
Matplotlib: For plotting graphs and feature importance.
Scikit-learn (sklearn): For machine learning algorithms, model training, evaluation, and preprocessing.
RandomForestClassifier: The main algorithm used for classification tasks.
