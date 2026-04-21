# 🔍 Multilingual Fake News Detector

A transformer-based fake news detection system fine-tuned on MuRIL 
(Multilingual Representations for Indian Languages) by Google.

## 🚀 Live Demo
[Try it on HuggingFace Spaces](#) 

## 📌 Project Highlights
- Fine-tuned `google/muril-base-cased` on the ISOT fake news dataset
- Achieved 99%+ accuracy on held-out test set
- Identified dataset bias through out-of-distribution testing
- Designed for multilingual extension (Hindi, Marathi) via MuRIL's Indic pretraining

## 🧠 Model Architecture
- Base: MuRIL (Multilingual Representations for Indian Languages)
- Task: Binary sequence classification (Real vs Fake)
- Input: Article title + content (max 256 tokens)
- Training: Fine-tuned for 3 epochs, lr=2e-5, batch size=16

## 📂 Project Structure
fake-news-detector/
├── data/              # datasets (not tracked in git)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_training.ipynb
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── app/
│   └── app.py         # Gradio demo
└── model/             # saved weights (not tracked in git)

## 🛠️ Tech Stack
- Python, PyTorch, HuggingFace Transformers
- Gradio (demo UI)
- scikit-learn (evaluation metrics)
- Pandas, NumPy (data processing)

## 📊 Results
| Split | Accuracy | F1 Score |
|-------|----------|----------|
| Test  | 100%     | 1.00     |

> ⚠️ Note: 100% accuracy on ISOT is a known dataset artifact — the model learns
> Reuters wire style as a signal for real news. Out-of-distribution testing reveals
> generalization limitations. LIAR dataset integration is in progress.

## 🗺️ Roadmap
- [ ] Add LIAR dataset for better generalization
- [ ] Hindi language support
- [ ] Marathi language support
- [ ] Deploy on HuggingFace Spaces

## 👤 Author
Tejas Waghere — B.Tech CSE (AI/ML), Pune
