# 💓 CardioSense - AI-Powered Cardiovascular Health Monitor

A privacy-first, offline-capable health dashboard using **Gemma 3n** (via Ollama) and **machine learning** to detect cardiovascular risks — optimized for accessibility and real-world impact.

## 🔍 Use Case
This project is built for the [Gemma 3n Build for Impact Challenge]. It provides:
- 📈 Cardiovascular risk prediction using ML
- 🧠 Personalized health recommendations powered by **Gemma 3n (via Ollama)**
- ⚠️ Real-time alerts when critical conditions detected

## 🛠️ Tech Stack
- [x] `Streamlit` for interactive UI
- [x] `scikit-learn` for ML model
- [x] `Ollama` for LLM inference using **Gemma 3n**
- [x] `Matplotlib` for health trend charts

## 🧠 How We Use Gemma 3n
For each daily health entry, we generate a natural language report and run:
```bash
ollama run gemma3n:e4b "Daily Health Report..."
