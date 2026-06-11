# CNIT ML Backend — Deployment Guide

## Step 1: Train models in Google Colab
1. Upload `train_and_export.ipynb` to Google Colab
2. Upload your `child_nutrition_karnataka.csv` dataset
3. Run all cells — takes ~5 mins on free Colab
4. Download the `models.zip` file at the end

## Step 2: Set up repo for Render
1. Create a new GitHub repo: `cnit-ml-backend`
2. Upload all files from this folder
3. Unzip `models.zip` → place all `.pkl` files inside a `models/` folder
4. Push to GitHub

## Step 3: Deploy on Render
1. Go to render.com → New → Web Service
2. Connect your `cnit-ml-backend` GitHub repo
3. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn main:app --bind 0.0.0.0:$PORT --workers 2`
   - Instance Type: Free
4. Click Deploy
5. Your API URL will be: `https://cnit-ml-api.onrender.com`

## Step 4: Connect to your Next.js frontend
Add to your `.env.local`:
```
NEXT_PUBLIC_ML_API_URL=https://cnit-ml-api.onrender.com
```

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API is running |
| `/predict/risk` | POST | Risk level for one child |
| `/predict/deficiency` | POST | All 5 deficiencies for one child |
| `/predict/batch` | POST | Risk for many children (forecasting) |
| `/forecast` | GET | 6-month Prophet forecast |
| `/insights` | GET | Model accuracy + SHAP features |

## Sample Request — Risk Prediction
```json
POST /predict/risk
{
  "age_months": 24,
  "gender": 0,
  "district": "Kalaburagi",
  "mother_edu": 1,
  "income_level": 0,
  "scheme_enrolled": 0,
  "weight_kg": 8.5,
  "height_cm": 74.0,
  "vitamin_a": 1,
  "iron": 0,
  "underweight": 0,
  "wasting": 0,
  "stunting": 0
}
```
