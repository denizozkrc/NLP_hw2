# Binary Text Classification with BERT for Turkish Parliamentary Speeches

This project implements a binary text classification  for Turkish parliamentary speeches using Hugging Face Transformers, PyTorch, and Scikit-learn. For this, pre-trained BERT model is fine-tuned and compared with zero-shot classification by LLaMA.

## Requirements
Install dependencies:
```bash
pip install torch transformers datasets scikit-learn pandas evaluate
```

## Dataset
"orientation-tr-dataset"  and  "power-tr-dataset" from the [ParlaMint](https://zenodo.org/doi/10.5281/zenodo.10450640) dataset are used.

## Steps to Run
- Make sure the tsv files are at the correct format in a directory called "datasets" located in the same directory as the main.py.
- Run the script:
 ```bash
python3 main.py
```

## Results
Results are stored in the specified output directories 

