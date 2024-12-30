# from sklearn.model_selection import StratifiedKFold
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
# model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)  # Adjust `num_labels`


dataset1_raw = (open(file="./datasets/orientation-tr-train.tsv", mode="r")).read()
# dataset1_inputs = []
# dataset1_labels = []
data1 = {"text": [], "label": []}
for line in dataset1_raw.split("\n"):
    if len(line) == 0:
        continue
    parts = line.split("\t")
    if(parts[-1] == "label"):
        continue
    text_org = parts[-3]
    text_eng = parts[-2]
    # dataset1_inputs.append(text_org)
    data1["text"].append(text_org)
    data1["label"].append(int(parts[-1]))
    # dataset1_labels.append(parts[-1])
# del dataset1_inputs[0]
# del dataset1_labels[0]
del data1["text"][0]
del data1["label"][0]
dataset1 = Dataset.from_dict(data1)
ds1 = dataset1

dataset2_raw = (open(file="./datasets/power-tr-train.tsv", mode="r")).read()
# dataset2_inputs = []
# dataset2_labels = []
data2 = {"text": [], "label": []}
for line in dataset2_raw.split("\n"):
    if len(line) == 0:
        continue
    parts = line.split("\t")
    if(parts[-1] == "label"):
        continue
    text_org = parts[-3]
    text_eng = parts[-2]
    # dataset2_inputs.append(text_org)
    # dataset2_labels.append(parts[-1])
    data2["text"].append(text_org)
    data2["label"].append(int(parts[-1]))
# del dataset2_inputs[0]
# del dataset2_labels[0]
dataset2 = Dataset.from_dict(data2)

# TODO: Stratified k-fold 1 to 9 for the shared task

# inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get model outputs
# outputs = model(**inputs)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)

"""for i, text in enumerate(dataset1_inputs):
    input = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input["labels"] = int(dataset1_labels[i])
    dataset1.append(input)"""


# Tokenize datasets
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)


dataset1 = dataset1.map(tokenize_function, batched=True)
dataset2 = dataset2.map(tokenize_function, batched=True)

# Tokenization check
#for i in range(len(dataset1)):
#    if(dataset1[i]["label"] != ds1[i]["label"]):
#        print("Error in tokenization")


"""trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset1,
    # eval_dataset=eval_dataset, # use this later
)

trainer.train()"""
