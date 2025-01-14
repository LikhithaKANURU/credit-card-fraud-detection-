# Credit Card Fraud Detection

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning models. The primary goal is to accurately distinguish between legitimate and fraudulent transactions to prevent financial losses and enhance security.

## Features
- Preprocessing of highly imbalanced datasets.
- Implementation of machine learning algorithms for classification.
- Performance evaluation using precision, recall, and F1-score.
- Visualization of data distributions and classification results.

## Dataset
- **Source**: The dataset used in this project is publicly available and contains credit card transactions, with labels indicating whether a transaction is fraudulent.
- **Key Characteristics**:
  - Highly imbalanced (fraud cases are rare).
  - Contains numerical features resulting from PCA transformation.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: 
  - Data Processing: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn, TensorFlow

## Implementation Steps
1. **Data Preprocessing**:
   - Handling missing values.
   - Scaling numerical features.
   - Addressing class imbalance using techniques like SMOTE.

2. **Exploratory Data Analysis (EDA)**:
   - Visualization of transaction amounts and time.
   - Analysis of correlations between features.
   - Example Visualizations:
     - ![Distribution of Transaction Amounts](images/transaction_amount_distribution.png)
     - ![Correlation Heatmap](images/correlation_heatmap.png)

3. **Model Development**:
   - Trained various classifiers (e.g., Logistic Regression, Random Forest, Neural Networks).
   - Evaluated models using metrics such as accuracy, precision, recall, and F1-score.

4. **Model Evaluation and Selection**:
   - Compared performance metrics to select the best-performing model.
   - Example Visualizations:
     - ![Confusion Matrix](images/confusion_matrix.png)
     - ![Precision-Recall Curve](images/precision_recall_curve.png)

## Results
- The model achieved high precision and recall, reducing false positives and negatives.
- Visualization of the confusion matrix demonstrated the model's effectiveness in identifying fraudulent transactions.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/LikhithaKANURU/credit-card-fraud-detection-.git
   ```
2. Navigate to the project directory:
   ```bash
   cd credit-card-fraud-detection
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook or Python script for training and evaluation.

## Future Improvements
- Incorporate advanced deep learning techniques for better accuracy.
- Test the model on real-world transaction data.
- Develop an API for real-time fraud detection.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request for any improvements.

## License
This project is licensed under the MIT License.

## Acknowledgements
- Special thanks to the creators of the dataset.
- Inspiration from the data science and machine learning community.

---

For more details, check the [GitHub Repository](https://github.com/LikhithaKANURU/credit-card-fraud-detection-.git).
