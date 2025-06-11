# End-to-End E-commerce Analysis and Prediction

## Project Description

This project performs a complete, end-to-end analysis of the Olist E-commerce dataset. The primary goal is to understand the factors that drive customer satisfaction and to predict low review scores. A key feature of this project is that the core statistical tests and machine learning algorithms were built entirely from scratch to demonstrate a foundational understanding of their principles.

***
## Folder Descriptions

* **`data/`**: Contains the final, processed dataset (`olist_abt_processed_output.csv`) that is ready for analysis and modeling.
* **`sql/`**: Holds all the SQL scripts. This includes queries for initial data exploration and the main, complex query used to build the project's analytical base table (ABT).
* **`src/`**: Contains all the Python source code used for the project, from data extraction to final machine learning model implementation.

***
## Python File Explanations

* **`abttable.py`**: Connects to the MySQL database, runs the main SQL query to join multiple tables, and saves the initial, comprehensive dataset.
* **`data_preprocessing.py`**: Loads the raw dataset created by `abttable.py`, performs all data cleaning (handles missing values, duplicates, etc.), and saves the final processed file.
* **`probability_operations.py`**: Uses the processed data to calculate basic, joint, and conditional probabilities to explore relationships between features like delivery times and review scores.
* **`statisticaloperations.py`**: Performs various hypothesis tests (like T-tests and ANOVA) and calculates confidence intervals from scratch to statistically validate insights.
* **`machinelearningops.py`**: Contains the from-scratch implementations of several machine learning models, including Linear Regression, Logistic Regression, and Random Forest. It then trains and evaluates these models on the processed data.
* **`perceptron.py`**: A standalone script that implements the Perceptron algorithm from scratch, serving as a bonus demonstration of fundamental ML concepts.
