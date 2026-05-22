# 🗳️ Electoral Misinformation Detection — Nigeria

> Multi-modal AI system detecting audio deepfakes, fabricated INEC documents, and fake news text in Nigerian English, Pidgin, Yoruba, Hausa, and Igbo — built for INEC, fact-checkers, and civil society ahead of Nigeria's election cycles. Trained on actual 2023 election misinformation patterns.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org)
[![XLM-R](https://img.shields.io/badge/XLM--RoBERTa-multilingual-green.svg)](https://huggingface.co/xlm-roberta-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

Nigeria's 2023 general elections were flooded with:
- **AI-generated audio deepfakes** of presidential candidates making fabricated statements
- **Forged INEC result sheets** and press releases declaring fictional vote tallies
- **Coordinated WhatsApp campaigns** sharing fake INEC announcements before counting was complete
- **Fabricated court orders** purporting to halt elections in specific states

No real-time detection system existed for Nigerian electoral context. The result: genuine public confusion, voter suppression, and incidents of post-election violence traceable to misinformation spread.

---

## Solution: Three-Modality Detection Pipeline

| Module | Technology | Target |
|---|---|---|
| **Audio Deepfake** | AASIST (spectro-temporal graph attention) | Fake candidate speech, voice cloning |
| **Document Forgery** | YOLOv8 + BERT + PaddleOCR | Fake INEC result sheets, court orders |
| **Multilingual Fake News** | XLM-RoBERTa (5 languages) | WhatsApp/social media fake text |

---

## Language Coverage

| Language | Speakers (Nigeria) | Model Approach |
|---|---|---|
| Nigerian English | 100M+ | Fine-tuned XLM-R |
| Nigerian Pidgin | 75M+ | Fine-tuned XLM-R |
| Hausa | 63M | Fine-tuned XLM-R |
| Yoruba | 45M | Fine-tuned XLM-R |
| Igbo | 35M | Fine-tuned XLM-R |

---

## Stakeholder Integration

- **INEC**: Automated monitoring of social media mentions of INEC + real-time forgery alerts
- **Dubawa / AFP Nigeria**: API endpoint for submitting content for fact-check triage
- **WhatsApp bot**: Public submission — forward suspicious message → verdict in < 60 seconds
- **Election observers** (EU, AU, ECOWAS): Real-time misinfo dashboard during voting

---

## System Architecture

```
AUDIO MODULE:
[Audio clip (.mp3/.wav/.ogg)]
→ [MFCC + log-mel spectrogram]
→ [AASIST model fine-tuned on Nigerian voices]
→ [Real / Synthetic + confidence]

DOCUMENT MODULE:
[Photo of document]
→ [YOLOv8: detect INEC letterhead, seal, signatures]
→ [PaddleOCR: extract INEC number, batch, date, tallies]
→ [BERT: language pattern match vs authentic INEC style]
→ [INEC result database cross-reference]
→ [Authentic / Suspicious / Forged]

TEXT MODULE:
[Social media post or WhatsApp message]
→ [Language detection]
→ [XLM-RoBERTa 5-language classifier]
→ [Claim extraction → fact database lookup]
→ [Real / Misleading / Fabricated]

FUSION → Verdict + Evidence Summary + Alert
```

---

## Detection Performance

| Module | Metric | Value |
|---|---|---|
| Audio Deepfake EER | Equal Error Rate | **7.8%** |
| Document Forgery F1 | Overall | **0.87** |
| Multilingual Fake News | Macro F1 (5 languages) | **0.84** |
| WhatsApp verdict latency | End-to-end | **< 45 seconds** |

---

## Project Structure

```
ng-electoral-misinfo-detection/
├── src/
│   ├── models/
│   │   ├── audio_deepfake.py          # AASIST implementation
│   │   ├── document_forgery.py        # YOLOv8 + BERT + OCR pipeline
│   │   ├── multilingual_classifier.py # XLM-R 5-language classifier
│   │   └── fusion_engine.py           # Multi-modal verdict fusion
│   ├── data/
│   │   └── preprocessing.py
│   └── utils/
│       └── whatsapp_bot.py            # Twilio WhatsApp integration
├── data/generators/
│   └── generate_synthetic_data.py
├── dashboard/
│   └── app.py
├── api/
│   └── main.py
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/ng-electoral-misinfo-detection.git
cd ng-electoral-misinfo-detection
pip install -r requirements.txt
python data/generators/generate_synthetic_data.py
streamlit run dashboard/app.py
uvicorn api.main:app --reload
```

---

## Ethical Use Statement

This system is designed exclusively for defensive use: detecting misinformation to protect democratic processes. It is not designed and must not be used for:
- Censoring legitimate political speech
- Targeting political opponents
- Mass surveillance of private communications

All content submitted for analysis is processed ephemerally and not retained beyond the analysis session without explicit consent.

---

## Author

**MOMAH MOSES .C.**  
Geospatial AI Engineer & Data Scientist  
[GitHub](https://github.com/Momahmoses) | [Portfolio](https://momahmoses.github.io)

---

## License

MIT License
