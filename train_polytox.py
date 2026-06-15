import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import time

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "output")
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "anbl-backend", "data", "webtool-2-data.json")
ONNX_PATH = os.path.join(MODEL_DIR, "polytox_model.onnx")
META_PATH = os.path.join(MODEL_DIR, "polytox_metadata.json")

# Mapping User Inputs to Dataset Columns (Refined for your Excel file)
INPUT_MAPPING = {
    'synthesis': 'Synthesis method',
    'polymers': 'Polymers',
    'polymer_type': 'Polymer type',
    'functional_group': 'Material_2',
    'core_size': 'Core size (nm)',
    'shape': 'Shape',
    'pdi': 'PDl',
    'hydro_size': 'Hydrodynamic size in water (nm)',
    'charge': 'Surface charge in water (mV)'
}

def train_and_export():
    print("🚀 Starting Poly-ToxMap ML Training Pipeline...")
    
    # Ensure output directory exists immediately
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load Data
    local_excel = os.path.join(os.path.dirname(__file__), "data", "Final datasets_Web develop.xlsx")
    if not os.path.exists(local_excel):
         # Try backend path if local missing
         local_excel = os.path.join(os.path.dirname(BASE_DIR), "anbl-backend", "data", "webtool-2-data.json")

    print(f"📖 Reading dataset: {local_excel}")
    if local_excel.endswith('.xlsx'):
        df = pd.read_excel(local_excel)
    else:
        df = pd.read_json(local_excel)
    
    # 2. Extract Determinants
    features = list(INPUT_MAPPING.values())
    X = df[features].copy()
    
    # 3. Process Target (Viability -> 3 Classes)
    target_col = 'Viability (%)' if 'Viability (%)' in df.columns else df.columns[-1]
    print(f"🎯 Target Column identified: {target_col}")

    def get_class(v):
        try:
            if isinstance(v, (int, float)):
                if v > 80: return 2
                if v > 50: return 1
                return 0
            val_str = str(v).lower()
            if 'biosafe' in val_str: return 2
            if 'moderate' in val_str: return 1
            if 'severe' in val_str: return 0
            
            val = float(val_str.replace(',', ''))
            if val > 80: return 2
            if val > 50: return 1
            return 0
        except: return 2
    
    y = df[target_col].apply(get_class)

    # 4. Data Harmonization (Preprocessing)
    meta = {"encodings": {}, "medians": {}, "features": features}
    
    for col in features:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Not reported').astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            meta["encodings"][col] = {str(label): int(idx) for idx, label in enumerate(le.classes_)}
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            meta["medians"][col] = float(median_val)

    print(f"📊 Splitting dataset (80:20 ratio) on {len(X)} nanoparticle conditions...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # --- NEW: Balancing with SMOTE ---
    print("⚖️ Balancing dataset classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"✅ Balanced Training Set: {len(X_train_balanced)} samples (previously {len(X_train)})")

    # 5. Model Training and Comparison
    print("\n⚔️ Training and comparing classifiers...")
    
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42
        ),
        "SVM": SVC(
            probability=True,
            kernel='rbf',
            C=1.0,
            random_state=42
        ),
        "RF": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        ),
        "NaiveBayes": GaussianNB()
    }
    
    comparison_data = []
    
    for name, clf in models.items():
        print(f"   Training {name}...")
        clf.fit(X_train_balanced.values, y_train_balanced)
        
        # Test performance
        y_pred = clf.predict(X_test.values)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        # 5-Fold Cross Validation for Stability
        print(f"   Running 5-Fold Cross-Validation for {name} stability...")
        cv_scores = cross_val_score(clf, X_train_balanced.values, y_train_balanced, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Latency benchmark
        sample_input = X_test.values[[0]]
        t0 = time.perf_counter()
        for _ in range(100):
            _ = clf.predict(sample_input)
        latency = ((time.perf_counter() - t0) / 100.0) * 1000.0 # average latency in ms
        
        print(f"   📊 {name} -> Acc: {acc:.4f} | CV: {cv_mean:.4f} ± {cv_std:.4f} | Latency: {latency:.4f} ms")
        
        comparison_data.append({
            "model": name,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "cv_mean": float(cv_mean),
            "cv_std": float(cv_std),
            "latency_ms": float(latency)
        })
        
    meta["model_comparison"] = comparison_data
    
    # Print a formatted ASCII table to the console for a clean benchmark view
    print("\n" + "="*86)
    print("📊 MACHINE LEARNING MODEL COMPARISON BENCHMARK (Test Split 80:20)")
    print("="*86)
    print(f"| {'Model Architecture':<20} | {'Accuracy':<10} | {'Precision':<10} | {'F1-Score':<10} | {'5-Fold CV Accuracy':<18} | {'Latency':<8} |")
    print("-"*86)
    for row in comparison_data:
        model_name = row["model"]
        if model_name == "XGBoost":
            model_name += " (Main)"
        print(f"| {model_name:<20} | {row['accuracy']*100:>8.2f}% | {row['precision']*100:>8.2f}% | {row['f1_score']*100:>8.2f}% | {row['cv_mean']*100:>7.2f}% ±{row['cv_std']*100:>4.2f}% | {row['latency_ms']:>5.3f} ms |")
    print("="*86)

    
    # Select XGBoost as the main model for export and further evaluations
    model = models["XGBoost"]
    y_pred = model.predict(X_test.values)
    final_acc = accuracy_score(y_test, y_pred)
    meta["model_accuracy"] = float(final_acc)
    
    print("\n📈 Final XGBoost Model Evaluation:")
    print(f"✅ Accuracy: {final_acc:.4f}")
    print("📝 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Severe', 'Moderate', 'Biosafe'], labels=[0, 1, 2]))

    print("\n🧠 SHAP Interpretability (Native Engine):")
    shap_plot_path = os.path.join(MODEL_DIR, "toxicity_drivers_shap.png")
    mean_shap = None

    try:
        # 1. Use XGBoost's native SHAP calculation (the most reliable way)
        booster = model.get_booster()
        dtest = xgb.DMatrix(X_test.values, feature_names=features)
        
        # Native SHAP returns: (samples, classes * (features + 1))
        # The +1 is for the bias/base_score term
        raw_contribs = booster.predict(dtest, pred_contribs=True)
        
        # Reshape to (samples, classes, features + 1)
        num_classes = 3
        num_features = len(features)
        shap_values = raw_contribs.reshape(-1, num_classes, num_features + 1)
        
        # Remove the bias term (last column) and take absolute mean across classes
        # This gives us a single importance score per feature per sample
        # shap_values shape: (159, 3, 10) -> (159, 3, 9) after removing bias
        shap_values_no_bias = shap_values[:, :, :-1]
        
        # Mean absolute SHAP across all 3 classes for visualization
        mean_abs_shap = np.mean(np.abs(shap_values_no_bias), axis=1)
        mean_shap = mean_abs_shap.mean(axis=0)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(mean_abs_shap, X_test, plot_type="bar", show=False)
        plt.savefig(shap_plot_path, bbox_inches='tight')
        plt.close()
        print(f"🔹 SHAP Summary Plot generated (Native XGBoost Engine): {shap_plot_path}")

    except Exception as e:
        print(f"⚠️ Native SHAP calculation failed: {e}")
        # Final Fallback: XGBoost Feature Importance
        print("💡 Falling back to internal feature importance...")
        mean_shap = model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        pd.Series(mean_shap, index=features).sort_values(ascending=True).plot(kind='barh', color='skyblue')
        plt.title("Toxicity Drivers (XGBoost Feature Importance)")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(shap_plot_path)
        plt.close()
        print(f"🔹 Driver Plot saved (Manual Fallback): {shap_plot_path}")

    # Toxicity Drivers List (Final Output)
    if mean_shap is not None:
        print("\n🔥 Toxicity Drivers:")
        drivers_list = pd.Series(mean_shap, index=features).sort_values(ascending=False)
        for i, (feat, score) in enumerate(drivers_list.items()):
            print(f" {i+1}. {feat:.<30} {score:.4f}")

    # 6. Binary Export (ONNX)
    print("\n📦 Converting model to ONNX binary format...")
    onnx_model = onnxmltools.convert_xgboost(
        model, 
        initial_types=[('input', FloatTensorType([None, len(features)]))]
    )
    
    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 7. Export Metadata JSON for Node.js Translation
    meta["class_map"] = {0: "Severe", 1: "Moderate", 2: "Biosafe"}
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    # --- NEW: REFINED OUTPUT REPORT ---
    print("\n" + "="*50)
    print("🌟 REFINED PREDICTION REPORT (Sample Inference)")
    print("="*50)

    # 1. Take a sample from the test set
    sample_idx = 0
    sample_X = X_test.iloc[[sample_idx]]
    sample_y_true = y_test.iloc[sample_idx]
    
    # 2. Get Prediction and Probability
    pred_class = int(model.predict(sample_X.values)[0])
    probs = model.predict_proba(sample_X.values)[0]
    
    # Estimate a "Viability Percentage" from probabilities for the gauge
    # Severe (0) is <50, Moderate (1) is 50-80, Biosafe (2) is >80
    # We'll use a weighted average to "guess" a percentage for visualization
    estimated_viability = (probs[0] * 25) + (probs[1] * 65) + (probs[2] * 90)
    
    safety_status = "✅ SAFE" if pred_class == 2 else "⚠️ MODERATE" if pred_class == 1 else "❌ TOXIC"
    toxicity_class = meta["class_map"][pred_class]

    print(f"🔹 Safety Status:    {safety_status}")
    print(f"🔹 Toxicity Class:  {toxicity_class}")
    print(f"🔹 Est. Viability:  {estimated_viability:.1f}%")
    print("-" * 30)
    
    # 3. Generate Viability Gauge
    gauge_path = os.path.join(MODEL_DIR, "viability_gauge.png")
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    
    # Gauge parameters
    colors = ['#ff4b4b', '#ffa500', '#00cc44'] # Red, Orange, Green
    values = [0, 50, 80, 100]
    
    # Plot segments
    for i in range(len(colors)):
        start = np.deg2rad(180 - (values[i] * 1.8))
        end = np.deg2rad(180 - (values[i+1] * 1.8))
        ax.barh(1, start - end, left=end, color=colors[i], height=0.5)

    # Plot Needle
    needle_pos = np.deg2rad(180 - (estimated_viability * 1.8))
    ax.annotate('', xy=(needle_pos, 1.1), xytext=(0, 0),
                arrowprops=dict(arrowstyle='wedge', color='black', lw=2))
    
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)
    ax.set_thetagrids([0, 180], labels=['100%', '0%'])
    ax.set_rmax(1.2)
    ax.set_yticklabels([])
    ax.grid(False)
    plt.title(f"Predicted Viability: {estimated_viability:.1f}% ({toxicity_class})", y=1.1)
    
    plt.savefig(gauge_path, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"🔹 Viability Gauge saved: {gauge_path}")
    print("="*50)

    print("\n✅ PREDICTION SYSTEM READY")
    print(f"🔹 Binary Model: {ONNX_PATH}")
    print(f"🔹 Metadata Map: {META_PATH}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    try:
        train_and_export()
    except Exception as e:
        print(f"❌ Pipeline Failed: {e}")
        print("Tip: Ensure you ran 'pip install pandas xgboost onnxmltools shap matplotlib openpyxl skl2onnx onnxconverter-common'")
