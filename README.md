# Bankruptcy Prediction Using Machine Learning
This project implements machine learning models to predict corporate bankruptcy based on financial ratios. The dataset, sourced from the **Emerging Markets Information Service (EMIS)**, consists of financial data from Polish companies spanning 2000-2013. 

Our models classify companies into **bankrupt** or **non-bankrupt** categories using financial indicators such as **profitability, liquidity, and leverage ratios**.

## 🚀 Project Overview
- **Objective:** Develop a predictive model to assess bankruptcy risk based on financial indicators.
- **Dataset:** [Polish Companies Bankruptcy Data (UCI)](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
- **ML Models Implemented:**
  - Logistic Regression
  - Hard-Margin Support Vector Machine (SVM)
  - Neural Networks (MLP)
  - Ensemble Learning (Combining multiple models)
- **Tech Stack:** Python, Scikit-Learn, TensorFlow, Pandas, NumPy, Matplotlib

## 📊 Dataset Description
The dataset includes **64 financial indicators**, covering:
- **Profitability**: Return on Assets (ROA), Operating Profit Rate
- **Liquidity**: Current Ratio, Quick Ratio
- **Leverage**: Debt Ratio, Interest Coverage Ratio
- **Other Factors**: Revenue Growth, Asset Turnover

## 🏆 Key Results
| Model                  | Accuracy | Recall | Precision | F1-score |
|------------------------|----------|--------|------------|----------|
| Logistic Regression   | 95.1%    | 0%    | 0%         | 0%      |
| Hard-Margin SVM       | 88%      | 18%   | 9%         | 12%     |
| Neural Networks (MLP) | 70%      | 99%   | 50%        | 9%      |
| Ensemble Model        | 47%      | 100%  | 4.7%       | 9%      |

## 🛠️ Implementation Steps
1. **Data Preprocessing**:
   - Feature engineering: Handling missing values, normalizing financial indicators
   - Class balancing through stratified sampling
2. **Model Development**:
   - Logistic Regression with Gradient Descent
   - Hard-Margin SVM for linear separation
   - Neural Network trained with TensorFlow and Keras
   - Ensemble Learning for improved performance
3. **Evaluation & Optimization**:
   - Metrics: Accuracy, Precision, Recall, F1-score
   - Hyperparameter tuning for optimization

## 📌 Key Challenges & Solutions
- **Class Imbalance**: Addressed using stratified sampling and cost-sensitive learning
- **Feature Correlation**: Reduced dimensionality by eliminating highly correlated features
- **Overfitting in Neural Networks**: Applied dropout layers and early stopping

## 🔗 Usage & Setup
### Install Dependencies
pip install -r requirements.txt
### Run the model
python train_model.py
### Predict Bankruptcy Risk
python predict.py --input sample_financial_data.csv

## 📌 Future Work
- Implement time-series forecasting to predict bankruptcy likelihood over multiple years
- Use Explainable AI (XAI) to analyze key financial ratios influencing bankruptcy risk

## 📜 References
- Polish Companies Bankruptcy Dataset - UCI: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data
- Financial Ratios Research Papers
