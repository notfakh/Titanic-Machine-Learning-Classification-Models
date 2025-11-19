# Titanic-Machine-Learning-Classification-Models
This project builds and evaluates multiple machine learning models to predict passenger survival on the Titanic dataset. It includes data preprocessing, feature encoding, model training, performance comparison, and visualization of results, including a decision tree plot.

## 🎯 Models Evaluated

- **Logistic Regression** - Binary classification baseline
- **Decision Tree Classifier** - Interpretable tree-based model with full visualization
- **Random Forest** - Ensemble of multiple decision trees
- **Support Vector Classifier (SVC)** - Linear kernel classifier
- **Gradient Boosting** - Advanced sequential ensemble method

## 🎬 Problem Statement

**Can we predict whether a passenger survived the Titanic disaster based on their characteristics?**

Features used for prediction:
- Age, Sex, Passenger Class
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Fare paid
- Port of embarkation

## 📊 Key Features

- ✅ Complete data preprocessing pipeline
- ✅ Handling missing values (Age, Embarked)
- ✅ Label encoding for categorical variables
- ✅ Feature standardization for SVM
- ✅ Performance metrics (Accuracy, Precision, Recall, F1-Score)
- ✅ Confusion matrices for each model
- ✅ Side-by-side accuracy vs F1-Score comparison
- ✅ **Full Decision Tree Visualization** - See the complete decision-making process!

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the Titanic dataset:
   - Download `Titanic-Dataset.csv` from [Kaggle](https://www.kaggle.com/c/titanic/data)
   - Place it in the project root directory

### Usage

Run the main script:
```bash
python titanic_model_comparison.py
```

The script will:
1. Load and preprocess the Titanic dataset
2. Handle missing values and encode categorical features
3. Train all 5 models
4. Display performance metrics
5. Generate comparison visualizations
6. Show a detailed decision tree diagram

## 📈 Sample Results

Based on the Titanic dataset (891 passengers):

| Model | Accuracy | F1-Score | Key Strength |
|-------|----------|----------|--------------|
| Logistic Regression | ~80% | ~0.78 | Fast, interpretable |
| Random Forest | ~82% | ~0.80 | Best overall performance |
| Gradient Boosting | ~81% | ~0.79 | Robust predictions |
| Decision Tree | ~78% | ~0.76 | Most interpretable |
| SVC | ~79% | ~0.77 | Good generalization |

*Note: Results may vary slightly due to random splits*

## 📚 Dataset Information

**Source:** Kaggle Titanic - Machine Learning from Disaster

**Dataset Details:**
- **891 passengers** in training set
- **11 features** including:
  - Survived (target): 0 = No, 1 = Yes
  - Pclass: Ticket class (1st, 2nd, 3rd)
  - Sex: Male or Female
  - Age: Age in years
  - SibSp: # of siblings/spouses aboard
  - Parch: # of parents/children aboard
  - Fare: Passenger fare
  - Embarked: Port (C = Cherbourg, Q = Queenstown, S = Southampton)

**Preprocessing Steps:**
1. Removed non-predictive columns (PassengerId, Name, Ticket, Cabin)
2. Filled missing Age values with median
3. Filled missing Embarked values with mode
4. Encoded Sex and Embarked as numerical values
5. Standardized features for SVM

## 🎨 Visualizations

### 1. Model Performance Comparison
Bar chart comparing Accuracy and F1-Score across all models

### 2. Decision Tree Visualization
Complete decision tree showing:
- Split conditions at each node
- Sample distribution
- Predicted classes with color coding
- Feature importance through tree structure

## 🔍 Data Preprocessing Pipeline

```python
# Missing Value Handling
- Age: Filled with median (handles outliers better)
- Embarked: Filled with mode (most common port)

# Feature Engineering
- Label Encoding: Sex (male/female) → (0/1)
- Label Encoding: Embarked (C/Q/S) → (0/1/2)
- Standardization: Applied for SVC model

# Feature Selection
- Removed: PassengerId, Name, Ticket, Cabin
- Kept: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
```

## 🛠️ Customization

### Adjust Train/Test Split

Modify line 37:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Change `test_size=0.2` to adjust the split ratio.

### Modify Model Hyperparameters

Edit the `models` dictionary (lines 45-51):
```python
models = {
    "Random Forest Classifier": RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=10,      # Maximum depth
        random_state=42
    ),
    # Customize other models...
}
```

### Adjust Decision Tree Depth

Control tree visualization complexity (line 99):
```python
dt_model = DecisionTreeClassifier(
    max_depth=5,      # Limit tree depth for cleaner visualization
    random_state=42
)
```

## 📊 Understanding the Results

### Confusion Matrix Interpretation
```
                 Predicted
                 No   Yes
Actual    No    [TN] [FP]
          Yes   [FN] [TP]
```

- **TN (True Negative):** Correctly predicted did not survive
- **TP (True Positive):** Correctly predicted survived
- **FP (False Positive):** Predicted survived, but didn't
- **FN (False Negative):** Predicted didn't survive, but did

### Decision Tree Reading Guide
- **Root node:** Top of tree, splits entire dataset
- **Internal nodes:** Decision points based on features
- **Leaf nodes:** Final predictions (orange = survived, blue = not survived)
- **Gini:** Measure of impurity (0 = pure, 0.5 = mixed)
- **Samples:** Number of passengers at that node
- **Value:** [not survived, survived] counts

## 🤝 Contributing

Contributions are welcome! Ideas for enhancement:

- Add cross-validation for more robust evaluation
- Implement hyperparameter tuning (GridSearchCV)
- Add feature importance analysis
- Include ROC curves and AUC scores
- Add neural network models
- Create interactive visualizations with Plotly
- Add SHAP values for model interpretability

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

FAKHRUL NASRI SUFI BIN SUFIAN
- GitHub: [@yourusername](https://github.com/notfakh) 
- LinkedIn: [Your Profile](https://www.linkedin.com/in/fakhrul-sufian-b51454363/)
- Email: fkhrlnasry@gmail.com

## 🙏 Acknowledgments

- Kaggle for providing the Titanic dataset
- Scikit-learn library contributors
- The data science community for tutorials and inspiration

## 📚 References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Decision Tree Visualization](https://scikit-learn.org/stable/modules/tree.html)

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please:
- Open an issue in this repository
- Contact me via email
- Connect on LinkedIn

---

⭐ If this project helped you learn about machine learning classification, please consider giving it a star!

## 🎓 Learning Outcomes

After working through this project, you'll understand:
- Data preprocessing and cleaning techniques
- Handling missing values in real-world datasets
- Encoding categorical variables
- Training multiple ML models
- Comparing model performance
- Interpreting decision trees
- Creating effective visualizations

**Perfect for:** Data science students, ML beginners, and anyone interested in the Titanic dataset!
