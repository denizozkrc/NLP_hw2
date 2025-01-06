import numpy as np
from sklearn.utils import compute_class_weight
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, pipeline, AutoModelForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification, AutoModel
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import string
from sklearn.metrics import f1_score, accuracy_score
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
tokenizer_eng = AutoTokenizer.from_pretrained("bert-base-uncased")

tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model_llama = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.1-8B", num_labels=2)

model1 = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)  # used with orientation
model2 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # used with power

def compute_class_weights(labels):
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

def execute_llama(dataset_test):
    classifier = pipeline("zero-shot-classification", model=model_llama, tokenizer=tokenizer_llama, device=0)

    print(dataset_test)
    predictions = classifier(dataset_test["text"], [1, 0])

    print(predictions[0])

    predicted_labels = []
    print("len_pred", len(predictions))
    for prediction in predictions:
        predicted_label = prediction["labels"][prediction["scores"].index(max(prediction["scores"]))]
        predicted_labels.append(predicted_label)
    accuracy = 0
    for true_l, predicted_l in zip(dataset_test["label"], predicted_labels):
        if float(true_l) == predicted_l:
            accuracy += 1
    print("correct ones: ", accuracy, "total: ", len(predicted_labels))
    accuracy = accuracy / len(predicted_labels)
    return (accuracy)


def tokenize_function(example):
    return tokenizer(example["text"], padding=True, truncation=True)


def tokenize_function_eng(example):
    return tokenizer_eng(example["text"], padding=True, truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


def execute_task(org_lang: bool, is_orientation: bool, output_dir: string):
    dataset_raw = pd.read_csv("./datasets/orientation-tr-train.tsv", sep="\t", header=None, names=["id", "speaker", "sex", "text_org", "text_eng", "label"]) if is_orientation else pd.read_csv("./datasets/power-tr-train.tsv", sep="\t", header=None, names=["id", "speaker", "sex", "text_org", "text_eng", "label"])
    dataset = dataset_raw[dataset_raw["label"] != "label"]
    #dataset = dataset.iloc[:100] # for trial phase
    dataset = dataset[dataset["text_org"].notnull()]
    dataset["label"] = dataset["label"].astype(int)

    dataset_train_split, dataset_test_split = train_test_split(dataset, test_size=0.1, random_state=0) # TODO: random state

    dataset_train_dict = {
        "text": dataset_train_split["text_org"].tolist(),
        "label": dataset_train_split["label"].tolist()
    }

    dataset_train_dict_eng = {
        "text": dataset_train_split["text_eng"].tolist(),
        "label": dataset_train_split["label"].tolist()
    }

    dataset_test_dict = {
        "text": dataset_test_split["text_org"].tolist(),
        "label": dataset_test_split["label"].tolist()
    }

    dataset_test_dict_eng = {
        "text": dataset_test_split["text_eng"].tolist(),
        "label": dataset_test_split["label"].tolist()
    }

    dataset_train = Dataset.from_dict(dataset_train_dict)
    dataset_test = Dataset.from_dict(dataset_test_dict)

    dataset_train_eng = Dataset.from_dict(dataset_train_dict_eng)
    dataset_test_eng = Dataset.from_dict(dataset_test_dict_eng)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="steps",
        eval_steps=250,
        num_train_epochs=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        #metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=["tensorboard"],
        logging_dir="./logs",
        save_steps=250,
        save_total_limit=2,
    )

    dataset_train_mapped = dataset_train.map(tokenize_function, batched=True)
    dataset_test_mapped = dataset_test.map(tokenize_function, batched=True)

    dataset_train_mapped_eng = dataset_train_eng.map(tokenize_function_eng, batched=True)
    dataset_test_mapped_eng = dataset_test_eng.map(tokenize_function_eng, batched=True)

    class_weights = compute_class_weights(dataset_train["label"])
    training_args.class_weights = class_weights

    trainer = Trainer(
        model=model1 if org_lang else model2,
        args=training_args,
        train_dataset=dataset_train_mapped if org_lang else dataset_train_mapped_eng,
        eval_dataset=dataset_test_mapped if org_lang else dataset_test_mapped_eng,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    evaluation_results = trainer.evaluate()
    print("Evaluation Results original language:", evaluation_results) if org_lang else print("Evaluation Results English:", evaluation_results)

    accuracy = execute_llama(dataset_test_mapped)  # on original lang (tr)
    print("original language, ", "is orientation: ", is_orientation)
    print("Accuracy: ", accuracy)

    accuracy = execute_llama(dataset_test_mapped_eng)  # on english
    print("english, ", "is orientation: ", is_orientation)
    print("Accuracy: ", accuracy)


# TODO: Stratified k-fold 1 to 9 for the shared task

execute_task(True, True, "./or_results")
execute_task(False, False, "./pow_results")
