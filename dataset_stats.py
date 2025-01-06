import pandas as pd

dataset_raw = pd.read_csv("./datasets/orientation-tr-train.tsv", sep="\t", header=None, names=["id", "speaker", "sex", "text_org", "text_eng", "label"]) 
dataset = dataset_raw[dataset_raw["label"] != "label"]
dataset = dataset[dataset["text_org"].notnull()]
dataset["label"] = dataset["label"].astype(int)

num_of_0_labels = len(dataset[dataset["label"] == 0])
num_of_1_labels = len(dataset[dataset["label"] == 1])
print("orientation: ")
print(f"    Number of 0 labels: {num_of_0_labels}")
print(f"    Number of 1 labels: {num_of_1_labels}")

dataset_raw = pd.read_csv("./datasets/power-tr-train.tsv", sep="\t", header=None, names=["id", "speaker", "sex", "text_org", "text_eng", "label"]) 
dataset = dataset_raw[dataset_raw["label"] != "label"]
#dataset = dataset.iloc[:100] # for trial phase
dataset = dataset[dataset["text_org"].notnull()]
dataset["label"] = dataset["label"].astype(int)

num_of_0_labels = len(dataset[dataset["label"] == 0])
num_of_1_labels = len(dataset[dataset["label"] == 1])
print("power: ")
print(f"    Number of 0 labels: {num_of_0_labels}")
print(f"    Number of 1 labels: {num_of_1_labels}")
