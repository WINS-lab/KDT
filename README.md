## Requirements

Software:
```
Python3
Pytorch >= 1.0
argparse == 1.1
```


## Prepare

* Download the ``google_model.bin`` from [here](https://share.weiyun.com/5GuzfVX), and save it to the ``models/`` directory.


The directory tree of K-BERT:
```
KDT
├── brain
│   ├── config.py
│   ├── __init__.py
│   ├── kgs
│   │   ├── Symptom.spo
│   │   └── Medical.spo
│   └── knowgraph.py
├── datasets
│   ├── medicalQA
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   └── train.tsv
│   ├── Mandarin
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   └── train.tsv
│    ...
│
├── models
│   ├── google_config.json
│   ├── google_model.bin
│   └── google_vocab.txt
├── outputs
├── uer
├── README.md
├── requirements.txt
├── run_kdt_cls.py
└── run_kdt_ner.py
```


## KDT for text classification

### Classification example

Run example on medical with Symptom:
```sh
CUDA_VISIBLE_DEVICES='0' python3 -u run_kdt_cls.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/medicalQA/train.tsv \
    --dev_path ./datasets/medicalQA/dev.tsv \
    --test_path ./datasets/medicalQA/test.tsv \
    --epochs_num 50 --batch_size 32 --kg_name Symptom \
    --output_model_path ./outputs/kbert_medicalQA_cls_Medical.bin
#    > ./outputs/kbert_medical_ner_Medical.log 2>&1 &
```

Options of ``run_kdt_cls.py``:
```
useage: [--pretrained_model_path] - Path to the pre-trained model parameters.
        [--config_path] - Path to the model configuration file.
        [--vocab_path] - Path to the vocabulary file.
        --train_path - Path to the training dataset.
        --dev_path - Path to the validating dataset.
        --test_path - Path to the testing dataset.
        [--epochs_num] - The number of training epoches.
        [--batch_size] - Batch size of the training process.
        [--kg_name] - The name of knowledge graph, "HowNet", "CnDbpedia" or "Medical".
        [--output_model_path] - Path to the output model.
```

## Dataset Splitting
| Data Set Name | Total Records | Train Data (70%) | Validation Data (20%) | Test Data (10%) |
|---------------|---------------|-------------------|-----------------------|----------------|
| MDQA (Mandarin) | 2.6 million | 1,820,000 | 520,000 | 260,000 |
| MDQA (English) | 1.73 million | 1,211,000 | 346,000 | 173,000 |
| MedQuAD | 114,000 | 91,200 | 22,800 | 11,400 |






### Classification benchmarks

Results of KDT and the baseline transformers on MDQA and MedQuAD (%)

| Models        | MedQuAD      | MDQA          |
| :-----        | :----:       | :----:        |
| BERT-base     | 92.0         | 92.3          |
| BERT-large    | 95.3         | 95.7          |
| ALBERT-base   | 85.6         | 85.6          |
| ALBERT-large  | 86.6         | 87.9          |
| DistilBERT    | 89.8         | 89.8          |
| RoBERTa-base  | 93.7         | 96.9          |
| RoBERTa-large | 97.3         | 93.8          |
| GPT-2         | 94.1         | 94.3          |
| GPT-3         | 98.5         | 98.7          |
| K-BERT        | 96.8         | 97.5          |
| KDT           | 99.3         | 99.5          |





