
---

```markdown
# 🚨 Credit Card Fraud Detection with AWS SageMaker

This project is part of an IBM Internship and aims to build and deploy a machine learning model to detect fraudulent credit card transactions using AWS cloud services.

---

## 📌 Overview

We use public datasets to simulate enterprise-level fraud detection systems with real-time capabilities. The solution is cloud-native, secure, and scalable — leveraging AWS SageMaker for model training and deployment.

---

## 🧠 Machine Learning Models

- ✅ **Random Forest Classifier** (Supervised)
- ✅ **Isolation Forest** (Unsupervised)

---

## 📁 Project Structure

```

IBM\_Project/
├── fraud\_detection.py         # Main training script
├── inference.py               # SageMaker inference handling
├── random\_forest\_model.pkl    # Trained model file
├── model.tar.gz               # Compressed model for deployment
├── requirements.txt           # Python dependencies
└── README.md                  # You're here!

````

---

## 📊 Dataset

- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Hosted on AWS S3:  
  `https://ibm-project-bucket-19.s3.us-east-1.amazonaws.com/creditcard.csv`

---

## ✅ Completed Milestones

### 1. Data Handling & Training
- Loaded dataset from S3
- Performed preprocessing and EDA
- Trained Isolation Forest and Random Forest
- Evaluated with metrics: Accuracy, Precision, Recall, F1-Score, MCC
- Saved best model as `random_forest_model.pkl`

### 2. Model Packaging
- Compressed model into `model.tar.gz`
- Uploaded to:  
  `s3://ibm-project-bucket-19/models/model.tar.gz`

### 3. GitHub Integration
- Repository: [github.com/mitul-2210/IBM_Project](https://github.com/mitul-2210/IBM_Project)
- Scripts and model artifacts uploaded

### 4. SageMaker Setup
- SageMaker Studio launched with appropriate IAM role
- S3 full access permissions configured
- Deployed model as an endpoint using `SKLearnModel`

```python
predictor = model.deploy(
    instance_type='ml.t2.medium',
    initial_instance_count=1,
    endpoint_name='fraud-detector-sagemaker-1'
)
````

### 5. Inference Script

* Created `inference.py` with:

  * `model_fn()`: Load model
  * `input_fn()`: Accept JSON
  * `predict_fn()`: Run prediction
  * `output_fn()`: Return output

---

## 🧪 Sample Inference

```python
sample = {
    "Time": 0.0,
    "V1": -1.3598071,
    "V2": -0.0727811,
    ...
    "Amount": 149.62
}

response = predictor.predict([sample])
print("Prediction:", response)
```

---

## 🛠 Requirements

```text
scikit-learn==0.23.2
pandas
numpy
matplotlib
seaborn
joblib
```

---

## 🚀 Next Steps

* [ ] Endpoint test with real-time transactions
* [ ] Setup AWS Lambda + API Gateway
* [ ] Integrate with monitoring (CloudWatch, Grafana)
* [ ] Add frontend/dashboard (optional)

---

## 📬 Author

* 👨‍💻 [Mitul Patel](https://github.com/mitul-2210)
* 🌐 GitHub Repo: [IBM\_Project](https://github.com/mitul-2210/IBM_Project)

---

```

Let me know if you want a version tailored for deployment notebooks or documentation styling with badges (like build passing, license, etc.)!
```
