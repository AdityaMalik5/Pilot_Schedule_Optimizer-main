# ✈️ Flight Schedule Optimizer

An intelligent, interactive web application built with **Streamlit** for analyzing and optimizing airline flight schedules.  
It leverages **Natural Language Processing (NLP)**, **Machine Learning**, and **Operations Research** techniques to provide data-driven decision support.

---

## 🚀 Features

- **Natural Language Querying**  
  - Accepts plain English queries like *“Best time to fly from Dallas”* or *“Average delay for Houston flights”*.  
  - Powered by **Sentence-BERT (SBERT)** for semantic understanding.

- **Statistical Analysis**  
  - Computes descriptive insights such as busiest carriers, average delays, and peak-hour distributions.

- **Delay Prediction**  
  - Dual-model ensemble (**Random Forest + XGBoost**) predicts probability of flight delays.  
  - Features engineered from time, airline, and congestion data.

- **Schedule Optimization**  
  - Recommends optimal flight slots to minimize congestion and expected delays.  
  - Uses airport-specific capacity and historical delay patterns.

- **Interactive Streamlit UI**  
  - Tab-based navigation for Statistics, Prediction, and Optimization.  
  - Interactive tables, charts, and recommendation cards.

---

## 🏗️ System Architecture

- **Configuration & Utilities** – stores airport details (IAH & DFW), dataset mappings, preprocessing helpers.  
- **NLP Engine** – SBERT embeddings + intent detection (statistics, prediction, optimization).  
- **Machine Learning Engine** – Random Forest + XGBoost ensemble for robust predictions.  
- **Schedule Optimization Engine** – congestion + delay modeling → optimization score ranking.  
- **Streamlit UI** – intuitive interface to interact with all features.

---

## 📂 Project Structure
```
├── data/                   # Dataset (hflights.csv)
├── models/                 # ML models (ignored in .gitignore)
├── outputs/                # Training outputs (ignored in .gitignore)
├── app.py                  # Main Streamlit app
├── utils/                  # Configs, preprocessing, helpers
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

---

## ⚡ Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/flight-schedule-optimizer.git
   cd flight-schedule-optimizer
   ```

2. Create & activate virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Example Queries
- *“What is the average delay for Houston flights?”* → **Statistics**  
- *“Best time to fly from Dallas”* → **Optimization**  
- *“Will my flight be delayed if I depart at 5 PM?”* → **Prediction**  

---

## 🔮 Future Work
- Extend to additional airports and datasets.  
- Deploy on cloud (AWS/GCP) with scalable data pipelines.  
- Add DVC or Hugging Face Hub for model storage.

---

