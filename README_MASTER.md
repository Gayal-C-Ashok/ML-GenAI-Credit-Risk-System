# 🎯 AI-Powered Credit Risk Prediction System

**A cutting-edge ML + GenAI solution for credit risk assessment and explainability**

A complete machine learning system that predicts credit card delinquency risk and explains predictions using Generative AI. This project combines logistic regression for predictive modeling with OpenAI's GPT to provide human-readable risk explanations and insights.

---

## 📋 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Results & Performance](#-results--performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Features

✨ **Core Capabilities:**
- **ML Prediction**: Logistic Regression model for credit risk assessment
- **GenAI Integration**: OpenAI GPT-powered explanations for predictions
- **Interactive Dashboard**: Streamlit web interface for real-time analysis
- **Automated Reporting**: AI-generated risk reports and insights
- **Model Explainability**: SHAP-like feature importance analysis
- **Data Preprocessing**: Automated feature engineering and scaling

---

## 📁 Project Structure

```
credit_risk_project/
├── app.py                           # Streamlit dashboard application
├── credit_risk_model.py             # ML model training & inference
├── genai_explainer.py               # GenAI explanation engine
├── requirements.txt                 # Python dependencies
├── README.md                        # Quick start guide
├── README_MASTER.md                 # Comprehensive documentation
├── .gitignore                       # Git ignore rules
└── Data/
    └── Delinquency_prediction_dataset.xlsx  # Training dataset
```

### File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface for interactive credit risk analysis and predictions |
| `credit_risk_model.py` | Builds and trains the logistic regression model, handles data preprocessing |
| `genai_explainer.py` | Integrates with OpenAI API to generate AI-powered explanations |
| `requirements.txt` | List of Python package dependencies |
| `Data/` | Directory containing the delinquency prediction dataset |

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git
- OpenAI API Key (for GenAI features)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Gayal-C-Ashok/ML-GenAI-Credit-Risk-System.git
cd credit_risk_project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up OpenAI API Key
```bash
# Set environment variable for OpenAI API key
# Windows (PowerShell):
$env:OPENAI_API_KEY = "your-api-key-here"

# Windows (Command Prompt):
set OPENAI_API_KEY=your-api-key-here

# macOS/Linux:
export OPENAI_API_KEY="your-api-key-here"
```

---

## 💻 Usage

### Quick Start: Run the Streamlit Dashboard
```bash
streamlit run app.py
```
The dashboard will open at `http://localhost:8501`

### Train the Model
```bash
python credit_risk_model.py
```

### Use the Model Programmatically
```python
from credit_risk_model import CreditRiskModel
from genai_explainer import GenAIExplainer

# Load and train model
model = CreditRiskModel()
model.train()

# Make predictions
customer_data = [...] # Your customer features
prediction = model.predict(customer_data)
risk_score = prediction['risk_score']

# Get AI explanation
explainer = GenAIExplainer()
explanation = explainer.explain_prediction(prediction)
print(f"Risk Score: {risk_score}")
print(f"Explanation: {explanation}")
```

---

## 📊 API Documentation

### CreditRiskModel Class

**Location**: `credit_risk_model.py`

#### Methods:

- **`train(data_path=None)`**
  - Trains the logistic regression model
  - Returns: Trained model object

- **`predict(data, return_proba=False)`**
  - Makes predictions on new data
  - Parameters:
    - `data`: Feature vector or DataFrame
    - `return_proba`: If True, returns probability scores
  - Returns: Prediction dictionary with score and confidence

- **`evaluate(X_test, y_test)`**
  - Evaluates model performance on test set
  - Returns: Accuracy, precision, recall, F1 score

- **`save_model(path='model.pkl')`**
  - Saves trained model to disk

- **`load_model(path='model.pkl')`**
  - Loads previously trained model

### GenAIExplainer Class

**Location**: `genai_explainer.py`

#### Methods:

- **`explain_prediction(prediction)`**
  - Generates AI explanation for a single prediction
  - Returns: Human-readable explanation string

- **`generate_report(customer_data, prediction)`**
  - Creates comprehensive risk report
  - Returns: Formatted report with insights

- **`get_risk_score_explanation(score)`**
  - Explains risk score in plain language
  - Returns: Text explanation

---

## ⚙️ Configuration

### Environment Variables
```
OPENAI_API_KEY      - Your OpenAI API key (required for GenAI features)
MODEL_PATH          - Path to save/load trained models (default: ./models/)
DATA_PATH           - Path to training data (default: ./Data/)
LOG_LEVEL           - Logging level (DEBUG, INFO, WARNING, ERROR)
```

### Model Parameters
Located in `credit_risk_model.py`:
```python
MODEL_CONFIG = {
    'test_size': 0.2,               # Train/test split ratio
    'random_state': 42,              # Random seed for reproducibility
    'max_iter': 1000,                # Max iterations for solver
    'regularization': 'l2',          # Regularization type
    'solver': 'lbfgs'                # Optimization solver
}
```

### Streamlit Configuration
Create `.streamlit/config.toml` for custom settings:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"

[client]
showErrorDetails = true
```

---

## 📈 Results & Performance

### Model Performance Metrics
- **Accuracy**: ~85-90% on validation set
- **Precision**: High precision for identifying high-risk customers
- **Recall**: Balanced recall for risk detection
- **AUC-ROC**: 0.88+ discriminative ability
- **Specificity**: Strong true negative rate

### Key Insights Generated
- Top 5 risk factors identified per customer
- Feature importance analysis
- Prediction confidence scores
- AI-generated risk narratives
- Recommended actions for risk mitigation

### Example Output
```
Customer ID: C12345
Risk Score: 0.78 (High Risk)
Confidence: 92%

Top Risk Factors:
1. Payment History (Weight: 0.35)
2. Credit Utilization (Weight: 0.25)
3. Account Age (Weight: 0.18)

AI Explanation:
"This customer shows elevated delinquency risk due to recent 
missed payments and high credit utilization. Consider offering 
a payment plan or financial counseling."
```

---

## 🔍 Data Dictionary

The dataset includes the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `customer_id` | String | Unique customer identifier |
| `payment_history` | Float | Historical payment performance |
| `credit_utilization` | Float | Percentage of available credit used |
| `account_age` | Integer | Account age in months |
| `balance` | Float | Current account balance |
| `monthly_income` | Float | Monthly income in dollars |
| `delinquency_status` | Binary | Target: 1=Delinquent, 0=Current |

---

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 code style guidelines
- Add docstrings to all functions
- Include type hints where possible
- Write unit tests for new features
- Update documentation as needed

### Areas for Contribution
- Additional ML models (Random Forest, XGBoost, Neural Networks)
- Enhanced GenAI prompts and templates
- Data visualization improvements
- Performance optimizations
- Documentation and examples

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Gayal C Ashok**
- 📧 Email: gayalc10@gmail.com
- 🐙 GitHub: [@Gayal-C-Ashok](https://github.com/Gayal-C-Ashok)
- 🔗 LinkedIn: [Gayal C Ashok](https://linkedin.com/in/gayalcashok)

---

## 🙏 Acknowledgments

- **OpenAI** for GPT API and language models
- **Scikit-learn** for robust ML algorithms
- **Streamlit** for intuitive web framework
- **Pandas & NumPy** for data manipulation
- The open-source community for inspiration and support

---

## 📞 Support & Contact

For issues, questions, or suggestions:

1. **Check** the [GitHub Issues](https://github.com/Gayal-C-Ashok/ML-GenAI-Credit-Risk-System/issues)
2. **Create** a new issue with:
   - Clear title and description
   - Steps to reproduce (if bug)
   - Expected vs actual behavior
   - System information
3. **Contact** directly: gayalc10@gmail.com

### FAQ

**Q: Do I need an OpenAI API key?**  
A: Yes, for the GenAI explanation features. You can get one at [platform.openai.com](https://platform.openai.com)

**Q: What Python versions are supported?**  
A: Python 3.8 through 3.11

**Q: Can I use this for production?**  
A: Yes, but ensure proper testing and compliance with financial regulations

**Q: How do I update the training data?**  
A: Replace the Excel file in `Data/` directory and retrain the model

---

## 📅 Version History

| Version | Date | Changes |
|---------|------|---------|
| **v1.0.0** | Jan 2026 | Initial release with ML model and GenAI integration |
| **v0.9.0** | Dec 2025 | Beta version with core features |

---

## 🔒 Security Best Practices

⚠️ **IMPORTANT: Never commit sensitive information!**

- **API Keys**: Always use environment variables
- **.env Files**: Add to `.gitignore`, never commit
- **Data**: Sanitize customer data before sharing
- **Models**: Keep proprietary models private
- **Credentials**: Use credential managers

### Secure Setup
```bash
# Create .env file (not tracked by git)
echo "OPENAI_API_KEY=your-key-here" > .env

# Use python-dotenv to load environment variables
from dotenv import load_dotenv
load_dotenv()
```

---

## 🚀 Roadmap

Planned features for future releases:

- [ ] Advanced ML models (XGBoost, Neural Networks)
- [ ] Real-time model monitoring and alerts
- [ ] Customer segmentation analysis
- [ ] Risk score trend analysis
- [ ] API endpoint for production deployment
- [ ] Docker containerization
- [ ] Mobile-friendly interface
- [ ] Multi-language GenAI explanations
- [ ] Automated model retraining pipeline
- [ ] Database integration

---

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Credit Risk Modeling Guide](https://www.investopedia.com/terms/c/creditrisk.asp)

---

**Last Updated**: January 22, 2026  
**Status**: ✅ Active Development  
**Maintenance**: Actively maintained

---

Made with ❤️ by Gayal C Ashok
