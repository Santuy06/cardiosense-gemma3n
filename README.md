# 💓 CardioSense - AI-Powered Cardiovascular Health Monitor

A privacy-first, offline-capable health dashboard using **Gemma 3n** (via Ollama) and **machine learning** to detect cardiovascular risks — optimized for accessibility and real-world impact.

---

## 🔍 Use Case

This project is submitted for the **[Gemma 3n Build for Impact Challenge]**. It aims to:

- 📈 Predict cardiovascular risk using real-world biometric inputs (HRV, HR, SpO₂, etc.)
- 🧠 Deliver personalized health advice using **Gemma 3n** LLM (via [Ollama](https://ollama.com)). The live demo uses fallback logic due to Streamlit Cloud limitations. Full AI recommendation is available with Ollama locally.
- ⚠️ Trigger alerts for potentially life-threatening heart indicators (e.g., resting HR < 40 bpm)

---

## 🛠️ Tech Stack

- `Streamlit` – Interactive web dashboard
- `scikit-learn` – Random Forest Classifier for medical risk
- `Ollama` – LLM runtime for **Gemma 3n**
- `Matplotlib` – Visualization of time-series health data
- `pandas` – Data management

---

## 🧠 How We Use Gemma 3n

For each entry, we build a personalized health report and run:

```bash
ollama run gemma3n:e4b "Daily Health Report..."
