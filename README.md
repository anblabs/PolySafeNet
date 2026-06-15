# PolySafeNet: An Interpretable Nanoinformatics Platform for Transferable Biosafety Prediction of Functionalized Polymeric Nanoparticles

PolySafeNet is an explainable AI-driven nanoinformatics platform designed for the transferable biosafety prediction and interpretation of functionalized polymeric nanoparticles. By integrating machine learning models with curated nanotoxicology data, the platform enables both high-accuracy prediction and a mechanistic understanding of nanoparticle-bio interactions.

## Authors & Affiliations

**Sri Renukadevi Balusamy<sup>1*</sup>, Irfan Ullah<sup>2</sup>, Priyanka Singh<sup>3</sup>, Sumit Kumar<sup>4</sup>, Rajiv Nagaraj<sup>5</sup>, Seungah Lee<sup>6</sup>, Sumathi Sundaravadivelu<sup>7</sup>, Haribalan Perumalsamy<sup>6*</sup>**

1. Department of Food Science and Biotechnology, Sejong University, Seoul, Republic of Korea
2. Department of Computer Science and Engineering, Kyung Hee University, Yongin, South Korea
3. Department of Health Technology, Technical University of Denmark, Kongens Lyngby, Denmark
4. Affiliation
5. Affiliation
6. Department of Applied Chemistry and Institute of Natural Sciences, Kyung Hee University, Yongin-si, South Korea
7. Department of Biochemistry, Biotechnology and Bioinformatics, Avinashilingam Institute, Coimbatore, India
8. Center for Creative Convergence Education, Hanyang University, Seoul, Republic of Korea

**Corresponding Authors:**
- Sri Renukadevi Balusamy (renubalu@sejong.ac.kr)
- Haribalan Perumalsamy (harijai2004@hanyang.ac.kr)

---

## Abstract

Functionalized polymeric nanoparticles are increasingly explored for neurological and biomedical applications owing to their tunable physicochemical properties. However, safety assessment remains constrained by heterogeneous nano–bio interactions and reliance on labor-intensive experimental screening. 

**PolySafeNet** addresses this by providing an interpretable AI framework built using a literature-curated in vitro dataset of **1,047 nanoparticle conditions** collected from 107 independent studies. Among multiple algorithms, **XGBoost** achieved the most robust and generalizable performance. Using SHAP-based interpretation, the platform identifies hydrodynamic size, synthesis method, polymer chemistry, surface functionalization, and surface charge as key determinants of nanoparticle biosafety.

---

## Key Features

- **Explainable AI (XAI):** Implements SHAP (SHapley Additive exPlanations) to quantify the contribution of individual physicochemical descriptors to toxicity outcomes.
- **Safer-by-Design Optimization:** Provides design-guided recommendations to minimize toxicity risk prior to experimental validation.
- **High-Performance Modeling:** Optimized XGBoost classifier for stratifying nanoparticles into **Toxic, Moderately Toxic, and Biosafe** profiles.
- **Standardized Data:** Features a harmonized dataset of polymeric and hybrid nanomaterials (PLGA, PEGylated copolymers, dendrimers, nanogels, etc.).

---

## Project Structure

- `train_polytox.py`: Main pipeline for data preprocessing, model training (XGBoost, SVM, RF, NB), and ONNX export.
- `inference_test.py`: Script to validate the trained model using ground truth samples from the dataset.
- `check_data.py`: Utility to verify dataset integrity and class distributions.
- `investigate_data.py`: Advanced analysis of the curated literature data.
- `data/`: Contains the curated dataset (`Final datasets_Web develop.xlsx`).
- `output/`: Generated model files, metadata, and interpretability plots.

---

## Installation

Ensure you have Python 3.10+ installed. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone <repository-url>
cd anbl-model

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Training and Exporting the Model
Run the training script to evaluate multiple models and export the best-performing XGBoost model to ONNX format.
```bash
python train_polytox.py
```
**Outputs in `output/`:**
- `polytox_model.onnx`: The serialized prediction engine.
- `polytox_metadata.json`: Encodings, medians, and class mappings for deployment.
- `toxicity_drivers_shap.png`: Visual summary of the top features driving toxicity.
- `viability_gauge.png`: A sample visualization of predicted cell viability.

### 2. Running Inference
Verify the model performance on specific nanoparticle configurations:
```bash
python inference_test.py
```

---

## Input Descriptors

The model utilizes 9 primary physicochemical and synthesis-related parameters:

| Parameter | Description |
|-----------|-------------|
| **Synthesis Method** | Fabrication strategy (e.g., Emulsion, Nanoprecipitation) |
| **Polymers** | Core polymer composition |
| **Polymer Type** | Classification (e.g., Biodegradable, PEGylated) |
| **Functional Group** | Surface modification/coating |
| **Core Size (nm)** | Primary particle size |
| **Shape** | Morphological configuration (e.g., Spherical, Rod) |
| **PDI** | Polydispersity Index |
| **Hydrodynamic Size** | Size in aqueous media (nm) |
| **Surface Charge** | Zeta potential (mV) |

---

## Web Platform

PolySafeNet is available as a web-based decision-support tool:
🔗 [https://anbl.co.kr/webtools](https://anbl.co.kr/webtools)
*(Alternative: [https://anbl-website.netlify.app/webtools](https://anbl-website.netlify.app/webtools))*

The web interface integrates toxicity prediction, SHAP-based explainability analysis, and design optimization guidance.

---

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the foundation for International Cooperation (R&D) (No. RS-2026-25550229; No. RS-2026-25550231).
