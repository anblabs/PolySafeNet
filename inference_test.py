import onnxruntime as ort
import numpy as np
import json
import os
import pandas as pd

# Paths
MODEL_PATH = "output/polytox_model.onnx"
META_PATH = "output/polytox_metadata.json"
DATA_PATH = "data/Final datasets_Web develop.xlsx"

def run_inference(input_data, label="Unknown"):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        print("❌ Error: Model or Metadata files missing.")
        return

    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    
    processed_input = []
    for feat in meta['features']:
        val = input_data.get(feat)
        if feat in meta['encodings']:
            str_val = str(val) if val is not None else "Not reported"
            encoded_val = meta['encodings'][feat].get(str_val, meta['encodings'][feat].get("Not reported", 0))
            processed_input.append(float(encoded_val))
        else:
            try:
                num_val = float(val) if val is not None and not pd.isna(val) else meta['medians'].get(feat, 0.0)
                processed_input.append(num_val)
            except:
                processed_input.append(meta['medians'].get(feat, 0.0))

    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    input_tensor = np.array([processed_input], dtype=np.float32)
    outputs = session.run(None, {input_name: input_tensor})
    
    pred_class = int(outputs[0][0])
    probs = outputs[1][0] 
    conf_score = probs.get(pred_class, max(probs.values())) if isinstance(probs, dict) else probs[pred_class]
    
    class_name = meta['class_map'].get(str(pred_class), "Unknown")
    safety_emoji = "✅ SAFE" if pred_class == 2 else "⚠️ MODERATE" if pred_class == 1 else "❌ TOXIC"
    
    print(f"\n" + "="*45)
    print(f"🔬 INFERENCE RESULT [{label}]")
    print(f"="*45)
    print(f"🔹 Safety Status:      {safety_emoji}")
    print(f"🔹 Toxicity Class:     {class_name}")
    print(f"🔹 Prediction Conf.:   {conf_score*100:.1f}%")
    print(f"🔸 Global Model Acc.:  {meta.get('model_accuracy', 0.0)*100:.2f}%")
    print(f"="*45)

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        print(f"📖 Loading Ground Truth data from {DATA_PATH}...")
        df = pd.read_excel(DATA_PATH)
        print(f"✅ Loaded {len(df)} rows.")
        print(f"📋 Available Columns: {df.columns.tolist()}")
        
        # Robust selection of target column
        target_candidates = ['Viability (%)', 'AO_Proxy_Toxicity_Class', 'Viability_percent', 'Viability%', 'Toxicity_Class']
        target_col = None
        for cand in target_candidates:
            if cand in df.columns:
                target_col = cand
                break
        
        if not target_col:
            target_col = df.columns[-1]
            print(f"⚠️ Warning: Target column not found in {target_candidates}. Using last column: '{target_col}'")
        else:
            print(f"🎯 Using target column: '{target_col}'")

        print(f"📊 Unique values in '{target_col}': {df[target_col].unique()[:5]}")
        
        # Try to convert to numeric for filtering
        viability_numeric = pd.to_numeric(df[target_col], errors='coerce')
        
        # Severe: Numeric < 50 OR contains 'severe' string
        is_severe = (viability_numeric < 50) | (df[target_col].astype(str).str.lower().str.contains('severe'))
        # Biosafe: Numeric > 80 OR contains 'biosafe' string
        is_biosafe = (viability_numeric > 80) | (df[target_col].astype(str).str.lower().str.contains('biosafe'))
        
        print(f"🔍 Found {is_severe.sum()} 'Severe' and {is_biosafe.sum()} 'Biosafe' rows.")

        severe_rows = df[is_severe].head(1)
        biosafe_rows = df[is_biosafe].head(1)
        
        mapping = {
            'Synthesis method': 'Synthesis method',
            'Polymers': 'Polymers',
            'Polymer type': 'Polymer type',
            'Material_2': 'Material_2',
            'Core size (nm)': 'Core size (nm)',
            'Shape': 'Shape',
            'PDl': 'PDl',
            'Hydrodynamic size in water (nm)': 'Hydrodynamic size in water (nm)',
            'Surface charge in water (mV)': 'Surface charge in water (mV)'
        }

        if not severe_rows.empty:
            sample = severe_rows.iloc[0].to_dict()
            mapped_sample = {mapping[k]: v for k, v in sample.items() if k in mapping}
            print("\n🧪 Testing REAL 'Severe' sample from dataset...")
            run_inference(mapped_sample, label="GROUND TRUTH: SEVERE")
        
        if not biosafe_rows.empty:
            sample = biosafe_rows.iloc[0].to_dict()
            mapped_sample = {mapping[k]: v for k, v in sample.items() if k in mapping}
            print("\n🧪 Testing REAL 'Biosafe' sample from dataset...")
            run_inference(mapped_sample, label="GROUND TRUTH: BIOSAFE")
    else:
        print("❌ Error: Ground truth Excel file not found.")
