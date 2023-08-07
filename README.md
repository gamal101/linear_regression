# linear_regression
# README

This code is a simple linear regression model that predicts salary based on years of experience. It uses the scikit-learn library to perform the regression analysis. Here's a breakdown of the code and its functionality:

1. Importing the necessary libraries:
   - `numpy` for numerical operations
   - `matplotlib.pyplot` for data visualization
   - `pandas` for data manipulation and analysis
   - `train_test_split` from `sklearn.model_selection` to split the dataset into training and testing sets
   - `LinearRegression` from `sklearn.linear_model` to create a linear regression model
   - `mean_squared_error` and `r2_score` from `sklearn.metrics` to evaluate the model's performance.

2. Loading the dataset:
   - The code reads the CSV file named "Salary_Data.csv" using `pd.read_csv()` from the pandas library.
   - The dataset is assumed to have two columns: years of experience (input) and salary (output).

3. Splitting the dataset:
   - The code splits the dataset into training and testing sets using `train_test_split()`.
   - It assigns 70% of the data to the training set and 30% to the testing set.

4. Creating and training the linear regression model:
   - The code creates a `LinearRegression` object.
   - It fits the model to the training data using `fit()`.

5. Predicting salaries:
   - The code uses the trained model to predict salaries for the test data using `predict()`.

6. Data visualization:
   - The code creates a scatter plot of the training data points using `plt.scatter()`.
   - It plots the regression line using `plt.plot()` with the training data.
   - It also creates a scatter plot of the test data points and plots the regression line for the test data.

7. Calculating the mean squared error (MSE):
   - The code calculates the MSE between the predicted salaries and the actual salaries for the test data.

8. Additional function `mm()`:
   - The code defines a function `mm()` that calculates the mean square difference between two arrays.

9. Predicting a salary for a given experience:
   - The code predicts a salary for an experience value of 8.8 years using the trained model.

10. Calculating the R2 score:
    - The code calculates the R2 score, which measures the proportion of the variance in the dependent variable that is predictable from the independent variable.

Note: Make sure to update the file path in `dataset = pd.read_csv("D:\Salary_Data.csv")` to the correct location of your "Salary_Data.csv" file.

This code provides a basic implementation of linear regression for salary prediction. Feel free to modify and build upon it for your specific requirements.
