# 🏥 Medical Cost Prediction

A full-stack machine learning application that predicts individual medical insurance charges using the **Medical Cost Personal Dataset**.

## Features

| Layer | Technology | Role |
|-------|-----------|------|
| Machine Learning | scikit-learn | Ridge, Random Forest, Gradient Boosting |
| Backend API | Flask + flask-cors | REST API serving predictions |
| Frontend | Streamlit + Plotly | Interactive UI with charts |
| Data Viz | Matplotlib, Seaborn, Plotly | EDA & result visualisations |

---

## Project Structure

```
medical_cost_prediction/
├── insurance.csv        # Dataset (Medical Cost Personal Dataset)
├── train_model.py       # EDA + model training → saves model.pkl
├── app.py               # Flask REST API  (port 5000)
├── streamlit_app.py     # Streamlit frontend (port 8501)
├── requirements.txt     # Python dependencies
├── plots/               # Auto-generated EDA plots (after training)
│   ├── charges_distribution.png
│   ├── charges_by_smoker.png
│   ├── bmi_vs_charges.png
│   ├── correlation_heatmap.png
│   └── feature_importance.png
└── model.pkl            # Saved model bundle (after training)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
This will:
- Perform exploratory data analysis and save plots to `./plots/`
- Train Ridge, Random Forest, and Gradient Boosting regressors
- Evaluate all models with 5-fold cross-validation
- Save the best model to `model.pkl`

### 3. Start the Flask API
```bash
python app.py
```
API will be available at `http://127.0.0.1:5000`

### 4. Launch the Streamlit frontend (new terminal)
```bash
streamlit run streamlit_app.py
```
App will open at `http://localhost:8501`

---

## API Endpoints

### `GET /health`
Returns API status and model metrics.

```json
{
  "status": "ok",
  "model": "GradientBoosting",
  "metrics": { "r2": 0.8734, "mae": 2540.12, "rmse": 4102.34, "cv_r2": 0.866 }
}
```

### `POST /predict`
Single prediction.

**Request body:**
```json
{
  "age": 35,
  "sex": "male",
  "bmi": 27.5,
  "children": 1,
  "smoker": "no",
  "region": "southeast"
}
```

**Response:**
```json
{
  "predicted_charge": 5842.31,
  "model_used": "GradientBoosting",
  "input": { ... }
}
```

### `POST /predict/batch`
Batch predictions — accepts a JSON array of input objects, returns array of predictions.

---

## Dataset

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | Age of the insured |
| `sex` | str | `male` / `female` |
| `bmi` | float | Body Mass Index |
| `children` | int | Number of dependents |
| `smoker` | str | `yes` / `no` |
| `region` | str | `northeast`, `northwest`, `southeast`, `southwest` |
| `charges` | float | **Target** — annual medical cost (USD) |

---

## Model Performance (typical results)

| Model | R² | MAE | RMSE | CV R² |
|-------|----|-----|------|-------|
| Ridge | ~0.79 | ~$4,200 | ~$6,100 | ~0.78 |
| Random Forest | ~0.87 | ~$2,600 | ~$4,200 | ~0.86 |
| Gradient Boosting | ~0.88 | ~$2,500 | ~$4,050 | ~0.87 |

---

## Streamlit App Tabs

1. **🔮 Predictor** — Input patient features, get instant cost prediction with risk gauge
2. **📊 Data Exploration** — Interactive Plotly charts exploring the dataset
3. **📈 Model Insights** — Model comparison metrics and EDA plots from training
4. **⚙️ Batch Predict** — Upload CSV for bulk predictions, download results
