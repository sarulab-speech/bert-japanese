export PYTHONPATH='/media/ssd/bert-japanese/models:PYTHONPATH'
export WORK_DIR='/media/ssd/bert-japanese'
python3 models/official/nlp/bert/run_pretraining.py \
--input_files="$WORK_DIR/bert/jawiki-20210201/wordpiece_unidic_lite/pretraining_data/pretraining_data_*.tfrecord" \
--model_dir="$WORK_DIR/bert/jawiki-20210201/wordpiece_unidic_lite/bert-tiny" \
--bert_config_file="$WORK_DIR/model_configs/bert-tiny/wordpiece/config.json" \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--train_batch_size=32 \
--learning_rate=1e-4 \
--num_train_epochs=100 \
--num_steps_per_epoch=10000 \
--optimizer_type=adamw \
--warmup_steps=10000 \
--num_gpus=2
