# Differential Evolution

A machine learning project implementing Differential Evolution optimization algorithm for feature selection and hyperparameter tuning on the Titanic dataset.

## 📋 Project Overview

This project demonstrates the application of Differential Evolution (DE), a population-based metaheuristic optimization algorithm, to improve machine learning model performance. The implementation uses the classic Titanic survival prediction dataset to showcase the optimization process.

## 🗂️ Repository Structure

```
Differential-Evolution/
├── Differential_Evolution.ipynb    # Main Jupyter notebook with implementation
├── Titanic/                         # Dataset folder
│   ├── train.csv                   # Training dataset
│   ├── test.csv                    # Test dataset
│   └── gender_submission.csv       # Sample submission file
└── README.md                        # Project documentation
```

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn
  - K-Nearest Neighbors (KNN)
  - Gaussian Naive Bayes
  - Decision Tree Classifier
- **Preprocessing**: StandardScaler, LabelEncoder, SimpleImputer

## 📊 Dataset

The project uses the Titanic dataset with the following features:
- **PassengerId**: Unique identifier for each passenger
- **Pclass**: Ticket class (1st, 2nd, 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- **Survived**: Target variable (0 = No, 1 = Yes)

## 🚀 Getting Started

### Prerequisites

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/qlbusalim/Differential-Evolution.git
cd Differential-Evolution
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Differential_Evolution.ipynb
```

3. Run the cells sequentially to:
   - Load and explore the data
   - Perform exploratory data analysis (EDA)
   - Preprocess the dataset
   - Apply Differential Evolution for optimization
   - Train and evaluate machine learning models

## 🔍 Features

- **Exploratory Data Analysis (EDA)**: Comprehensive visualization of numerical and categorical features
- **Data Preprocessing**: 
  - Handling missing values
  - Feature encoding with one-hot encoding
  - Train-validation split (80-20)
- **Machine Learning Models**: Multiple classifiers including KNN, Naive Bayes, and Decision Trees
- **Model Evaluation**: Accuracy score, confusion matrix, and classification reports

## 📈 Workflow

1. **Data Loading**: Import train and test datasets
2. **EDA**: Analyze distribution of features using histograms and count plots
3. **Preprocessing**: Handle missing values and encode categorical variables
4. **Train-Test Split**: Split data into training (80%) and validation (20%) sets
5. **Model Training**: Apply machine learning algorithms
6. **Optimization**: Use Differential Evolution for hyperparameter tuning
7. **Evaluation**: Assess model performance using various metrics

## 📝 License

This project is available for educational and research purposes.

## 👤 Author

**qlbusalim**

- GitHub: [@qlbusalim](https://github.com/qlbusalim)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## ⭐ Show Your Support

Give a ⭐️ if this project helped you learn about Differential Evolution and machine learning optimization!
```
