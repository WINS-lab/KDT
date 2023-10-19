#  CUDA_VISIBLE_DEVICES='0,1' nohup python3.6 -u run_kbert_cls.py
CUDA_VISIBLE_DEVICES='0,1' python3.6 -u run_kbert_cls.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/medicalQA/train.tsv \
    --dev_path ./datasets/medicalQA/dev.tsv \
    --test_path ./datasets/medicalQA/test.tsv \
    --epochs_num 5 --batch_size 32 --kg_name Symptom \
    --output_model_path ./outputs/kbert_medicalQA_cls_Medical.bin
#    > ./outputs/kbert_medical_ner_Medical.log 2>&1 &
