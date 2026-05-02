# 🏠 Smart Property Advisor

A machine learning-powered web application that predicts residential property prices based on key housing and neighborhood features. Built with an end-to-end ML pipeline and deployed via Streamlit.

---

## 📌 Overview

Smart Property Advisor uses a trained regression model on the classic Boston Housing dataset to estimate property prices. Users can interactively adjust feature values through sliders and input fields and get an instant price prediction, making it a practical tool for data-driven real estate decision-making.

---

## 🚀 Features

- **Interactive Price Predictor** — Adjust 13 housing features in real time and get an estimated market value
- **Key Insights Panel** — Understand which factors (rooms, crime rate, pollution, income levels) most influence prices
- **Reset & Clear Controls** — Quickly reset all inputs to sensible defaults or clear them
- **Clean Streamlit UI** — Multi-page navigation with sidebar menu (Predictor, Insights, About)
- **End-to-End ML Pipeline** — Preprocessing and model inference bundled together via a `PredictPipeline` class

---

## 🧠 Input Features

| Feature | Description |
|---|---|
| `CRIM` | Per capita crime rate |
| `ZN` | Proportion of residential land zoned (%) |
| `INDUS` | Non-retail business area (%) |
| `CHAS` | Adjacent to Charles River (0/1) |
| `NOX` | Nitric oxide concentration (air pollution) |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of units built before 1940 (%) |
| `DIS` | Weighted distance to employment centers |
| `RAD` | Highway accessibility index |
| `TAX` | Property tax rate per $10,000 |
| `PTRATIO` | Pupil-teacher ratio |
| `B` | Population diversity index |
| `LSTAT` | Lower income population (%) |

---

## 🗂️ Project Structure

```
smart-property-advisor/
│
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
├── runtime.txt             # Python runtime version
│
├── src/
│   └── pipeline/
│       └── predict_pipeline.py   # CustomData & PredictPipeline classes
│
├── artifacts/              # Saved model and preprocessor files
├── catboost_info/          # CatBoost training logs
├── notebook/               # Jupyter notebooks for EDA & model training
└── .github/workflows/      # CI/CD workflows
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.x

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/AtharvaVSawant/smart-property-advisor.git
cd smart-property-advisor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

---

## 📦 Tech Stack

| Category | Tools |
|---|---|
| Frontend | Streamlit, streamlit-option-menu |
| ML / Modeling | Scikit-learn, CatBoost, XGBoost |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn, Plotly |
| Serialization | Dill |

---

## 📊 Model Pipeline

1. **Data Ingestion** — Load and split the Boston Housing dataset
2. **Preprocessing** — Feature scaling and transformation using Scikit-learn pipelines
3. **Model Training** — Multiple regressors evaluated; best model selected (CatBoost/XGBoost)
4. **Artifact Saving** — Preprocessor and model serialized with `dill`
5. **Prediction** — `PredictPipeline` loads artifacts and runs inference on new inputs

---

## 💡 Key Insights

- **More rooms → Higher price** — Room count is one of the strongest positive predictors
- **Higher crime rate → Lower price** — Crime significantly depresses property value
- **Air pollution → Negative impact** — NOx levels inversely affect prices
- **Lower income population → Lower price** — LSTAT is a strong negative indicator
- **Distance to jobs → Mixed impact** — Depends on transportation accessibility

---

## 👤 Author

**Atharva Sawant**
- 📧 atharvasawant3183@gmail.com
- 📞 +91 9653320569

---

## 📄 License

This project is open-source and available for portfolio and educational purposes.
