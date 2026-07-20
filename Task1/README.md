# Mountain Named Entity Recognition (NER) System

This repository contains a production-ready, End-to-End Named Entity Recognition (NER) pipeline specifically fine-tuned to extract **mountain names** from unstructured text. The solution leverages the State-of-the-Art **BERT** (Bidirectional Encoder Representations from Transformers) token classification architecture using the standard Inside-Outside-Beginning (IOB) tagging schema.

---

## 🚀 Quick Start & Environment Setup

### 1. Position the Working Directory
Ensure your terminal session is correctly pointing to the root directory of the project:
```bash
cd C:\Users\Artem\Quantum_task
2. Install Project DependenciesInstall all required libraries and pinned framework packages optimized for Python 3.11+:Bashpip install -r requirements.txt
📂 Repository Structure & DeliverablesThe project layout is modularly designed to fulfill all requirements outlined in the technical task specification:dataset_creation.ipynb: Jupyter notebook explaining raw IOB text ingestion, parsing strategies, dataset splits, and sub-token label alignment using Hugging Face Word IDs.train_ner.py: Production-ready Python training script that handles data tokenization, metric evaluation callbacks, and native weights checkpointing.inference.py: Lightweight, command-line utility for running lightning-fast lookups on arbitrary text inputs using dynamic token aggregation.demo.ipynb: Interactive demonstration notebook providing rich visual feedback using the SpaCy displacy HTML formatting engine.generate_report.py: Standalone automation tool that builds a styled, comprehensive PDF project report containing metrics and architectural insights.requirements.txt: Pinned inventory of necessary framework packages ensuring deterministic project reproducibility.dataset.zip: Compressed file containing the primary source dataset file (train.txt) structured in IOB format.🛠️ Model Training & Label AlignmentThe system uses bert-base-uncased as its underlying pre-trained backbone and applies supervised token-level fine-tuning.Dynamic Alignment MechanismBecause BERT relies on a WordPiece tokenizer, localized geographical names (e.g., Hoverla) are split into fragment groups (hove, ##rl, ##a). To combat this, the ingestion module utilizes word_ids to map labels accurately back onto the split tokens. Special tokens like [CLS] and [SEP] are padded with a token index value of -100, forcing the PyTorch loss calculation layers to skip them entirely.To re-run the full training pipeline on your CPU/GPU hardware, execute:Bashpython train_ner.py
📊 Training Evaluation & Performance MetricsThe model converged perfectly over the evaluation split (85% Train / 15% Validation) during the 3-epoch execution block:MetricEvaluation ValueTarget StatusTraining Loss0.00985Fully ConvergedValidation Loss0.000032Highly StableOverall Precision100.0%Perfect AlignmentOverall Recall100.0%Zero Missed TargetsOverall F1-Score1.0000Optimal🧠 Inference Interface & Sample OutputThe inference script incorporates a first grouping strategy (aggregation_strategy="first") to elegantly stitch sub-token entities together into standalone, complete words before console output.Running Local Evaluation:Bashpython inference.py
Verification Output:PlaintextText: Yesterday we decided to travel to Mount Everest, it was awesome.
Entities Found:
  - Mountain: 'mount everest' (Confidence: 99.98%)

Text: The highest peak in Ukraine is Hoverla, but Kilimanjaro is much higher.
Entities Found:
  - Mountain: 'hoverla' (Confidence: 84.39%)
  - Mountain: 'kilimanjaro' (Confidence: 98.85%)

Text: I want to climb Mont Blanc next summer with my friends.
Entities Found:
  - Mountain: 'mont blanc' (Confidence: 99.98%)
🔗 Model Weights & Document CompilationLink to Model Weights: 📥 Click here to download best_mountain_ner_model.zip(Extract the contents directly into the project root directory before executing inferences locally).Project PDF Report: A comprehensive analysis document detailing metrics, architectural limits, and full future improvement roadmaps can be generated instantly by running:Bashpython generate_report.py