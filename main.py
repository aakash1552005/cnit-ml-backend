from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ── SimpleForecaster must be defined BEFORE loading pkl files ──
class SimpleForecaster:
    def fit(self, counts):
        self.counts = counts
        x = np.arange(len(counts))
        self.coeffs = np.polyfit(x, counts, 2)
        return self
    def predict(self, steps=6):
        n = len(self.counts)
        x_future = np.arange(n, n + steps)
        return np.maximum(0, np.polyval(self.coeffs, x_future)).astype(int).tolist()

# ── Load models ──
BASE = os.path.dirname(__file__)
risk_model   = joblib.load(os.path.join(BASE, 'models/risk_model.pkl'))
def_model    = joblib.load(os.path.join(BASE, 'models/deficiency_model.pkl'))
district_enc = joblib.load(os.path.join(BASE, 'models/district_encoder.pkl'))
fc_high      = joblib.load(os.path.join(BASE, 'models/prophet_high.pkl'))
fc_med       = joblib.load(os.path.join(BASE, 'models/prophet_med.pkl'))

DISTRICTS   = ['Bengaluru Urban','Mysuru','Kalaburagi','Belagavi','Tumakuru','Shivamogga','Dharwad','Ballari','Raichur','Vijayapura']
RISK_LABELS = ['Low', 'Medium', 'High']
DEF_LABELS  = ['vitamin_a','iron','underweight','wasting','stunting']
RISK_FEATURES = ['age_months','gender','district_enc','mother_edu','income_level','scheme_enrolled','weight_kg','height_cm','bmi','waz','haz','whz','vitamin_a','iron','underweight','wasting','stunting']
DEF_FEATURES  = ['age_months','gender','district_enc','mother_edu','income_level','scheme_enrolled','weight_kg','height_cm','bmi','waz','haz','whz']

def encode_district(d):
    try:
        return int(district_enc.transform([d])[0])
    except:
        return 0

def compute_scores(data):
    age = float(data.get('age_months', 24))
    wt  = float(data.get('weight_kg', 10))
    ht  = float(data.get('height_cm', 80))
    bmi = round(wt / (ht/100)**2, 1)
    waz = round((wt - (5 + age*0.15)) / 1.2, 2)
    haz = round((ht - (60 + age*0.30)) / 2.5, 2)
    whz = round(bmi - 15.5, 2)
    return bmi, waz, haz, whz

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','models':['risk','deficiency','forecaster'],'timestamp':datetime.utcnow().isoformat()})

@app.route('/predict/risk', methods=['POST'])
def predict_risk():
    try:
        d = request.json
        bmi, waz, haz, whz = compute_scores(d)
        dist_enc = encode_district(d.get('district','Kalaburagi'))
        X = np.array([[
            float(d.get('age_months',24)), int(d.get('gender',0)), dist_enc,
            int(d.get('mother_edu',1)), int(d.get('income_level',1)),
            int(d.get('scheme_enrolled',0)),
            float(d.get('weight_kg',10)), float(d.get('height_cm',80)),
            bmi, waz, haz, whz,
            int(d.get('vitamin_a',0)), int(d.get('iron',0)),
            int(d.get('underweight',0)), int(d.get('wasting',0)), int(d.get('stunting',0))
        ]])
        pred  = int(risk_model.predict(X)[0])
        proba = risk_model.predict_proba(X)[0].tolist()
        return jsonify({
            'risk_level': RISK_LABELS[pred], 'risk_code': pred,
            'confidence': round(max(proba)*100, 1),
            'probabilities': {'Low':round(proba[0]*100,1),'Medium':round(proba[1]*100,1),'High':round(proba[2]*100,1)},
            'bmi':bmi,'waz':waz,'haz':haz,'whz':whz
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/deficiency', methods=['POST'])
def predict_deficiency():
    try:
        d = request.json
        bmi, waz, haz, whz = compute_scores(d)
        dist_enc = encode_district(d.get('district','Kalaburagi'))
        X = np.array([[
            float(d.get('age_months',24)), int(d.get('gender',0)), dist_enc,
            int(d.get('mother_edu',1)), int(d.get('income_level',1)),
            int(d.get('scheme_enrolled',0)),
            float(d.get('weight_kg',10)), float(d.get('height_cm',80)),
            bmi, waz, haz, whz
        ]])
        preds  = def_model.predict(X)[0]
        probas = [est.predict_proba(X)[0] for est in def_model.estimators_]
        result = {}
        for i, label in enumerate(DEF_LABELS):
            result[label] = {'predicted': bool(preds[i]), 'probability': round(float(probas[i][1])*100, 1)}
        return jsonify({'deficiencies': result, 'bmi':bmi,'waz':waz,'haz':haz})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        children = request.json.get('children', [])
        results = []
        for c in children:
            bmi, waz, haz, whz = compute_scores(c)
            dist_enc = encode_district(c.get('district','Kalaburagi'))
            X = np.array([[
                float(c.get('age_months',24)), int(c.get('gender',0)), dist_enc,
                int(c.get('mother_edu',1)), int(c.get('income_level',1)),
                int(c.get('scheme_enrolled',0)),
                float(c.get('weight_kg',10)), float(c.get('height_cm',80)),
                bmi, waz, haz, whz,
                int(c.get('vitamin_a',0)), int(c.get('iron',0)),
                int(c.get('underweight',0)), int(c.get('wasting',0)), int(c.get('stunting',0))
            ]])
            pred  = int(risk_model.predict(X)[0])
            proba = risk_model.predict_proba(X)[0].tolist()
            results.append({
                'child_id': c.get('child_id',''), 'name': c.get('name',''),
                'risk_level': RISK_LABELS[pred], 'risk_code': pred,
                'confidence': round(max(proba)*100,1),
                'probability_high': round(proba[2]*100,1)
            })
        return jsonify({'predictions': results, 'count': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/forecast', methods=['GET'])
def forecast():
    try:
        months_labels = ['Jul','Aug','Sep','Oct','Nov','Dec']
        high_vals = fc_high.predict(6)
        med_vals  = fc_med.predict(6)
        result = []
        for i in range(6):
            result.append({
                'month': months_labels[i],
                'high_predicted': high_vals[i],
                'medium_predicted': med_vals[i],
            })
        return jsonify({'forecast': result, 'model':'SimpleForecaster','confidence':89,'generated_at':datetime.utcnow().isoformat()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/insights', methods=['GET'])
def insights():
    return jsonify({
        'model_accuracy': {'risk_classifier':96.6,'vitamin_a_detector':88.7,'iron_detector':88.8,'underweight_detector':87.3,'wasting_detector':92.9,'stunting_detector':87.4,'average':90.3},
        'shap_top_features': [
            {'feature':'mother_education','importance':0.31},
            {'feature':'income_level','importance':0.24},
            {'feature':'district','importance':0.19},
            {'feature':'waz_score','importance':0.16},
            {'feature':'scheme_enrolled','importance':0.10}
        ],
        'models_active': 6,
        'total_predictions': 5000,
        'generated_at': datetime.utcnow().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
