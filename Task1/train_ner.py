import os
import zipfile
import numpy as np
import torch
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)

# ==========================================
# 1. FUNCTION FOR READING DATA FROM ZIP ARCHIVE
# ==========================================
def parse_iob_from_zip(zip_path, file_inside_zip):
    print(f"Reading data from the archive {zip_path} (file: {file_inside_zip})...")
    sentences = []
    tokens = []
    tags = []
    
    # Dictionary for converting text labels to numeric IDs
    label_map = {"O": 0, "B-MOUNTAIN": 1, "I-MOUNTAIN": 2}

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Error: Archive '{zip_path}' not found in current directory!")

    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(file_inside_zip, 'r') as f:
            for line in f:
                line = line.decode('utf-8').strip()
                
                if not line:
                    if tokens:
                        sentences.append({"tokens": tokens, "ner_tags": tags})
                        tokens = []
                        tags = []
                    continue
                
                splits = line.split()
                if len(splits) == 2:
                    token, tag = splits[0], splits[1]
                    tokens.append(token)
                    tags.append(label_map.get(tag, 0))
                    
            if tokens:
                sentences.append({"tokens": tokens, "ner_tags": tags})
                
    print(f"Успішно завантажено {len(sentences)} речень.")
    return Dataset.from_list(sentences)


# ==========================================
# 2. TOKENIZATION AND TAGS ALIGNMENT
# ==========================================
def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP]) are marked as -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                #The first subtoken of the word gets the original label
                label_ids.append(label[word_idx])
            else:
                # Subsequent subtokens receive the same label
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
            
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# ==========================================
# 3. MAIN PROCESS
# ==========================================
def main():
   # Path settings (CHANGE THE ARCHIVE NAME TO YOURS)
    ZIP_PATH = "dataset.zip" 
    FILE_INSIDE_ZIP = "train.txt"
    MODEL_NAME = "bert-base-uncased"
    OUTPUT_MODEL_DIR = "./best_mountain_ner_model"

    # Determine the device (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training will be performed on the device: {device.upper()}")

    #1. Downloading raw data
    raw_dataset = parse_iob_from_zip(ZIP_PATH, FILE_INSIDE_ZIP)

    # 2. Downloading the tokenizer
    print(f"Downloading the tokenizer'{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Dataset tokenization
    print("Токенізація та вирівнювання міток...")
    tokenized_dataset = raw_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), 
        batched=True
    )

    # 4. Splitting the sample into Train and Validation (85% / 15%)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    print(f"Розмір вибірки для навчання: {len(train_dataset)}")
    print(f"Розмір вибірки для валідації: {len(val_dataset)}")

    # 5. Preparing evaluation metrics (seqeval)
    metric = evaluate.load("seqeval")
    label_list = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # 6. Model Initialization
    print("Loading the BERT model for token classification...")
    
   # Clearly specify label matching dictionaries
    id2label = {0: "O", 1: "B-MOUNTAIN", 2: "I-MOUNTAIN"}
    label2id = {"O": 0, "B-MOUNTAIN": 1, "I-MOUNTAIN": 2}

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True #Will reset the old classification of the CoNLL-03 model
    )
    # 7. Creating a data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 8. Setting parameters
    training_args = TrainingArguments(
        output_dir="./mountain_ner_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True if torch.cuda.is_available() else False,
        report_to="none" # Disables sending logs to third-party services (Wandb, etc.)
    )

   # 9. Trainer Initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 10. Launching training
    print("\n=== BEGINNING OF TRAINING ===")
    trainer.train()
    print("=== TRAINING COMPLETED ===\n")

    #11. Saving the best model
    print(f"Saving the best model to a directory: {OUTPUT_MODEL_DIR}")
    trainer.save_model(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print("The process completed successfully.!")


if __name__ == "__main__":
    main()