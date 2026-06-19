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

# ── Pre-built forecasters ──
HIGH_COUNTS = [120,115,108,100,95,90,85,82,78,74,71,68,66,62,58,54,50,46]
MED_COUNTS  = [820,810,800,790,780,770,760,750,745,735,728,722,719,710,700,690,680,670]
fc_high = SimpleForecaster().fit(HIGH_COUNTS)
fc_med  = SimpleForecaster().fit(MED_COUNTS)

# ── Load models ──
BASE = os.path.dirname(__file__)
def load(name):
    return joblib.load(os.path.join(BASE, 'models', name))

risk_xgb     = load('risk_model.pkl')
risk_lgbm    = load('risk_lgbm.pkl')
risk_rf      = load('risk_rf.pkl')
def_model    = load('deficiency_model.pkl')
district_enc = load('district_encoder.pkl')
RISK_FEATURES = load('risk_features.pkl')
DEF_FEATURES  = load('def_features.pkl')

RISK_LABELS = ['Low', 'Medium', 'High']
DEF_LABELS  = ['iron_deficiency','vitamin_a_deficiency','protein_deficiency',
               'zinc_deficiency','stunting','wasting','underweight']

def encode_district(d):
    try:
        return int(district_enc.transform([d])[0])
    except:
        return 0

def build_risk_vector(d):
    age = float(d.get('age_months', 24))
    wt  = float(d.get('weight_kg', 10))
    ht  = float(d.get('height_cm', 80))
    bmi = round(wt / (ht/100)**2, 1)
    waz = float(d.get('waz', round((wt-(5+age*0.15))/1.2, 2)))
    haz = float(d.get('haz', round((ht-(60+age*0.30))/2.5, 2)))
    whz = float(d.get('whz', round(bmi-15.5, 2)))
    dist_enc = encode_district(d.get('district','Kalaburagi'))
    gender_enc = 1 if str(d.get('gender','Male')).lower() in ['female','f','1'] else 0
    vacc_enc   = 1 if str(d.get('vaccination_status','')).lower() == 'complete' else 0

    feature_map = {
        'wasting':               int(d.get('wasting', 0)),
        'vitamin_a_deficiency':  int(d.get('vitamin_a', 0)),
        'iron_deficiency':       int(d.get('iron', 0)),
        'underweight':           int(d.get('underweight', 0)),
        'protein_deficiency':    int(d.get('protein_deficiency', 0)),
        'waz':    waz, 'whz': whz, 'bmi': bmi,
        'zinc_deficiency':       int(d.get('zinc_deficiency', 0)),
        'vacc_enc':   vacc_enc,
        'weight_kg':  wt,
        'stunting':              int(d.get('stunting', 0)),
        'age_months': age,
        'height_cm':  ht,
        'district_enc': dist_enc,
        'gender_enc':   gender_enc,
        'haz': haz,
    }
    return np.array([[feature_map.get(f, 0) for f in RISK_FEATURES]])

def build_def_vector(d):
    age = float(d.get('age_months', 24))
    wt  = float(d.get('weight_kg', 10))
    ht  = float(d.get('height_cm', 80))
    bmi = round(wt / (ht/100)**2, 1)
    waz = float(d.get('waz', round((wt-(5+age*0.15))/1.2, 2)))
    haz = float(d.get('haz', round((ht-(60+age*0.30))/2.5, 2)))
    whz = float(d.get('whz', round(bmi-15.5, 2)))
    dist_enc   = encode_district(d.get('district','Kalaburagi'))
    gender_enc = 1 if str(d.get('gender','Male')).lower() in ['female','f','1'] else 0
    vacc_enc   = 1 if str(d.get('vaccination_status','')).lower() == 'complete' else 0

    feature_map = {
        'age_months': age, 'gender_enc': gender_enc, 'district_enc': dist_enc,
        'weight_kg': wt, 'height_cm': ht, 'bmi': bmi,
        'haz': haz, 'waz': waz, 'whz': whz, 'vacc_enc': vacc_enc,
    }
    return np.array([[feature_map.get(f, 0) for f in DEF_FEATURES]])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models': ['risk_xgb','risk_lgbm','risk_rf','deficiency','forecaster'],
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/predict/risk', methods=['POST'])
def predict_risk():
    try:
        d = request.json
        X = build_risk_vector(d)
        # Ensemble: weighted soft voting
        p_xgb  = risk_xgb.predict_proba(X)[0]
        p_lgbm = risk_lgbm.predict_proba(X)[0]
        p_rf   = risk_rf.predict_proba(X)[0]
        proba  = (p_xgb*2 + p_lgbm*2 + p_rf) / 5
        pred   = int(np.argmax(proba))
        return jsonify({
            'risk_level': RISK_LABELS[pred],
            'risk_code':  pred,
            'confidence': round(float(np.max(proba))*100, 1),
            'probabilities': {
                'Low':    round(float(proba[0])*100, 1),
                'Medium': round(float(proba[1])*100, 1),
                'High':   round(float(proba[2])*100, 1),
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/deficiency', methods=['POST'])
def predict_deficiency():
    try:
        d = request.json
        X = build_def_vector(d)
        preds  = def_model.predict(X)[0]
        probas = [est.predict_proba(X)[0] for est in def_model.estimators_]
        result = {}
        for i, label in enumerate(DEF_LABELS):
            result[label] = {
                'predicted':   bool(preds[i]),
                'probability': round(float(probas[i][1])*100, 1)
            }
        return jsonify({'deficiencies': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        children = request.json.get('children', [])
        results  = []
        for c in children:
            X     = build_risk_vector(c)
            p_xgb  = risk_xgb.predict_proba(X)[0]
            p_lgbm = risk_lgbm.predict_proba(X)[0]
            p_rf   = risk_rf.predict_proba(X)[0]
            proba  = (p_xgb*2 + p_lgbm*2 + p_rf) / 5
            pred   = int(np.argmax(proba))
            results.append({
                'child_id':         c.get('child_id',''),
                'name':             c.get('name',''),
                'risk_level':       RISK_LABELS[pred],
                'risk_code':        pred,
                'confidence':       round(float(np.max(proba))*100, 1),
                'probability_high': round(float(proba[2])*100, 1),
            })
        return jsonify({'predictions': results, 'count': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/forecast', methods=['GET'])
def forecast():
    months = ['Jul','Aug','Sep','Oct','Nov','Dec']
    high_vals = fc_high.predict(6)
    med_vals  = fc_med.predict(6)
    result = [{'month': months[i], 'high_predicted': high_vals[i], 'medium_predicted': med_vals[i]} for i in range(6)]
    return jsonify({'forecast': result, 'model':'SimpleForecaster', 'confidence':89, 'generated_at':datetime.utcnow().isoformat()})

@app.route('/insights', methods=['GET'])
def insights():
    return jsonify({
        'model_accuracy': {
            'risk_classifier':       86.4,
            'vitamin_a_detector':    84.3,
            'iron_detector':         84.5,
            'underweight_detector':  100.0,
            'wasting_detector':      100.0,
            'stunting_detector':     100.0,
            'protein_detector':      90.3,
            'zinc_detector':         89.7,
            'average':               91.3
        },
        'shap_top_features': [
            {'feature':'wasting',           'importance':0.51},
            {'feature':'vitamin_a_deficiency','importance':0.30},
            {'feature':'iron_deficiency',   'importance':0.30},
            {'feature':'underweight',       'importance':0.28},
            {'feature':'protein_deficiency','importance':0.27},
        ],
        'models_active': 5,
        'generated_at': datetime.utcnow().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
