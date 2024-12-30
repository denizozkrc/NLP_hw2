# from sklearn.model_selection import StratifiedKFold
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
# model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
model1 = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)
model2 = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)


def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)
    # tokenizer(text, return_tensors="pt", padding=True, truncation=True)


def execute_task1():
    # on original text
    dataset_raw = (open(file="./datasets/orientation-tr-train.tsv", mode="r")).read()
    dataset_dict = {"text": [], "label": []}
    for line in dataset_raw.split("\n"):
        if len(line) == 0:
            continue
        parts = line.split("\t")
        if (parts[-1] == "label"):
            continue
        text_org = parts[-3]
        # text_eng = parts[-2]
        dataset_dict["text"].append(text_org)
        dataset_dict["label"].append(int(parts[-1]))
    dataset = Dataset.from_dict(dataset_dict)
    #ds1 = dataset1

    # on original text
    dataset_raw_test = (open(file="./datasets/orientation-tr-test.tsv", mode="r")).read()
    dataset_dict_test = {"text": [], "label": []}
    for line in dataset_raw_test.split("\n"):
        if len(line) == 0:
            continue
        parts = line.split("\t")
        if (parts[-1] == "label"):
            continue
        text_org = parts[-3]
        # text_eng = parts[-2]
        dataset_dict_test["text"].append(text_org)
        dataset_dict_test["label"].append(int(parts[-1]))
    dataset_test = Dataset.from_dict(dataset_dict_test)
    #ds1 = dataset1

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
    )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset_test = dataset_test.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model1,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset_test,
    )

    trainer.train()

    evaluation_results = trainer.evaluate()
    print("Evaluation Results task1:", evaluation_results)


def execute_task2():
    # on english translated text 
    dataset_raw = (open(file="./datasets/power-tr-train.tsv", mode="r")).read()
    dataset_dict = {"text": [], "label": []}
    for line in dataset_raw.split("\n"):
        if len(line) == 0:
            continue
        parts = line.split("\t")
        if (parts[-1] == "label"):
            continue
        # text_org = parts[-3]
        text_eng = parts[-2]
        dataset_dict["text"].append(text_eng)
        dataset_dict["label"].append(int(parts[-1]))
    dataset = Dataset.from_dict(dataset_dict)

    # on english translated text 
    dataset_raw_test = (open(file="./datasets/power-tr-test.tsv", mode="r")).read()
    dataset_dict_test = {"text": [], "label": []}
    for line in dataset_raw_test.split("\n"):
        if len(line) == 0:
            continue
        parts = line.split("\t")
        if (parts[-1] == "label"):
            continue
        # text_org = parts[-3]
        text_eng = parts[-2]
        dataset_dict_test["text"].append(text_eng)
        dataset_dict_test["label"].append(int(parts[-1]))
    dataset_test = Dataset.from_dict(dataset_dict_test)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
    )

    dataset = dataset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model2,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset_test, # use this later
    )

    trainer.train()

    evaluation_results = trainer.evaluate()
    print("Evaluation Results task2: ", evaluation_results)


# TODO: Stratified k-fold 1 to 9 for the shared task

# Tokenization check
#for i in range(len(dataset1)):
#    if(dataset1[i]["label"] != ds1[i]["label"]):
#        print("Error in tokenization")
