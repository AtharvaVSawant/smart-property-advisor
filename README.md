# 🏠 Smart Property Advisor

> An end-to-end Machine Learning web application that predicts residential property prices based on neighborhood and structural features — built with a production-style ML pipeline and deployed via Streamlit.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-FF4B4B?logo=streamlit)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.5-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

<!-- Replace the line below with your actual GIF once recorded -->
<!-- ![Demo](assets/demo.gif) -->

---

## 📌 Features

- 🎯 Predicts property prices in real-time using a trained ML model
- 🧪 Clean end-to-end pipeline: data ingestion → transformation → training → prediction
- 📊 Insights page explaining key drivers of property prices
- 🔄 Interactive sliders and inputs with Reset / Clear functionality
- ✅ Deployment-safe preprocessing with dynamic feature reindexing

---

## 🗂️ Project Structure

```
smart-property-advisor/
│
├── src/
│   ├── pipeline/
│   │   ├── predict_pipeline.py     # CustomData + PredictPipeline classes
│   │   └── train_pipeline.py       # Training orchestration
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── notebook/                       # EDA and model training notebooks
├── artifacts/                      # Saved model and preprocessor (auto-generated)
├── app.py                          # Streamlit application entry point
├── setup.py
├── requirements.txt
├── runtime.txt
└── .github/workflows/              # CI/CD pipeline
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AtharvaVSawant/smart-property-advisor.git
cd smart-property-advisor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (first time only)

```bash
python src/pipeline/train_pipeline.py
```

This will generate the trained model and preprocessor inside the `artifacts/` folder.

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🧠 Input Features

| Feature | Description |
|---|---|
| CRIM | Per-capita crime rate |
| ZN | % of residential land zoned for large lots |
| INDUS | % non-retail business acres |
| CHAS | Located near Charles River (0/1) |
| NOX | Nitric oxide concentration (air pollution) |
| RM | Average number of rooms per dwelling |
| AGE | % of homes built before 1940 |
| DIS | Weighted distance to employment centres |
| RAD | Highway accessibility index |
| TAX | Property tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio |
| B | Population diversity index |
| LSTAT | % lower-income population |

---

## 🤖 Model & Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| ML Models | CatBoost, XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Serialization | Dill |
| Visualization | Matplotlib, Seaborn, Plotly |
| CI/CD | GitHub Actions |

---

## 📊 Key Insights

- **More rooms → Higher price** (strongest positive feature)
- **Higher crime rate → Lower price**
- **Air pollution (NOx) → Negative impact**
- **Lower-income population % → Negative impact**
- **Distance to employment → Mixed, non-linear effect**

---

## 📬 Contact

**Atharva Sawant**
📧 [atharvasawant3183@gmail.com](mailto:atharvasawant3183@gmail.com)
🔗 [GitHub Profile](https://github.com/AtharvaVSawant)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
