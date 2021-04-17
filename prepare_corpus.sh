python make_corpus_wiki.py \
    --input_file jawiki_dump/jawiki-20210201-cirrussearch-content.json.gz \
    --output_file jawiki_dump/jawiki-20210201/corpus.txt \
    --min_text_length 10 \
    --max_text_length 200 \
    --mecab_option "-r /etc/mecabrc -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"

python merge_split_corpora.py --input_files ./jawiki_dump/jawiki-20210201/corpus.txt --output_dir ./jawiki_dump/jawiki-20210201 --num_files 100

cat jawiki_dump/jawiki-20210201/corpus.txt|grep -v '^$'|shuf|head -n 1000000 > jawiki_dump/jawiki-20210201/corpus_sampled.txt

TOKENIZERS_PARALLELISM=false python train_tokenizer.py \
--input_files $WORK_DIR/jawiki_dump/jawiki-20210201/corpus_sampled.txt \
--output_dir $WORK_DIR/tokenizers/jawiki-20210201/wordpiece_unidic_lite \
--tokenizer_type wordpiece \
--mecab_dic_type unidic_lite \
--vocab_size 32768 \
--limit_alphabet 6129 \
--num_unused_tokens 10

mkdir -p $WORK_DIR/bert/jawiki-20210201/wordpiece_unidic_lite/pretraining_data



seq -f %02g 1 200|xargs -L 1 -I {} -P 20 python create_pretraining_data.py \
--input_file ./jawiki_dump/jawiki-20210201/corpus_{}.txt \
--output_file ./bert/jawiki-20210201/wordpiece_unidic_lite/pretraining_data/pretraining_data_{}.tfrecord.gz \
--vocab_file ./tokenizers/jawiki-20210201/wordpiece_unidic_lite/vocab.txt \
--tokenizer_type wordpiece \
--mecab_dic_type unidic_lite \
--do_whole_word_mask \
--gzip_compress \
--max_seq_length 512 \
--max_predictions_per_seq 80 \
--dupe_factor 10
udo 