import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments, pipeline, AutoModelForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
import pandas as pd



tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")

# model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
model1 = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)
model2 = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)

model_gpt2 = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)

metric = evaluate.load("accuracy")
# TODO: max_length


def tokenize_function_gpt2(example):
    return tokenizer_gpt2(example["text"], padding="max_length", truncation=True)


def execute_gpt2(org_lang: bool, file_name: str):    
    dataset_raw = pd.read_csv(file_name, sep="\t", header=None, names=["id", "speaker", "sex", "text_org", "text_eng", "label"])
    dataset = dataset_raw[dataset_raw["label"] != "label"]
    dataset = dataset.iloc[:100] # for trial phase
    dataset = dataset[dataset["text_org"].notnull()]
    dataset["label"] = dataset["label"].astype(int)

    dataset_dict = {
        "text": dataset["text_org"].tolist() if org_lang else dataset["text_eng"].tolist(),
        "label": dataset["label"].tolist()
    }

    dataset_test= Dataset.from_dict(dataset_dict)
    dataset_test = dataset_test.map(tokenize_function_gpt2, batched=True)

    classifier = pipeline("text-classification", model=model_gpt2, tokenizer=tokenizer_gpt2)
    predictions = classifier(dataset_test["text"])

    predicted_labels = []
    for prediction in predictions:
        predicted_labels.append(int(prediction["label"]))

    accuracy = 0
    for true_l, predicted_l in zip(dataset_test["label"], predicted_labels):
        if true_l == predicted_l:
            accuracy += 1
    accuracy = accuracy / len(predicted_labels)
    print("original language: ", org_lang, "file_name: ", file_name)
    print("Accuracy: ", accuracy)


def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)
    # tokenizer(text, return_tensors="pt", padding=True, truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def execute_task(org_lang: bool, is_orientation: bool):
    dataset_raw = pd.read_csv("./datasets/orientation-tr-train.tsv", sep="\t", header=None, names=["id", "speaker", "sex", "text_org", "text_eng", "label"]) if is_orientation else pd.read_csv("./datasets/power-tr-train.tsv", sep="\t", header=None, names=["id", "speaker", "sex", "text_org", "text_eng", "label"])
    dataset = dataset_raw[dataset_raw["label"] != "label"]
    dataset = dataset.iloc[:100] # for trial phase
    dataset = dataset[dataset["text_org"].notnull()]
    dataset["label"] = dataset["label"].astype(int)

    dataset_train_split, dataset_test_split = train_test_split(dataset, test_size=0.1, random_state=0) # TODO: random state

    dataset_train_dict = {
        "text": dataset_train_split["text_org"].tolist() if org_lang else dataset_train_split["text_eng"].tolist(),
        "label": dataset_train_split["label"].tolist()
    }

    dataset_test_dict = {
        "text": dataset_test_split["text_org"].tolist() if org_lang else dataset_test_split["text_eng"].tolist(),
        "label": dataset_test_split["label"].tolist()
    }

    dataset_train = Dataset.from_dict(dataset_train_dict)
    dataset_test = Dataset.from_dict(dataset_test_dict)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
    )

    dataset_train = dataset_train.map(tokenize_function, batched=True)
    dataset_test = dataset_test.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model1,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    evaluation_results = trainer.evaluate()
    print("Evaluation Results original language:", evaluation_results) if org_lang else print("Evaluation Results English:", evaluation_results)

    if is_orientation:
        execute_gpt2(True, "./datasets/orientation-tr-test.tsv")  # on original lang (tr)
        execute_gpt2(False, "./datasets/orientation-tr-test.tsv")  # on english
    else:
        execute_gpt2(True, "./datasets/power-tr-test.tsv")  # on original lang (tr)
        execute_gpt2(False, "./datasets/power-tr-test.tsv")  # on english


# TODO: Stratified k-fold 1 to 9 for the shared task

# Tokenization check
#for i in range(len(dataset1)):
#    if(dataset1[i]["label"] != ds1[i]["label"]):
#        print("Error in tokenization")

execute_task(True, True)
execute_task(False, False)
