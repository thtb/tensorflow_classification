#!/bin/bash

usage()
{
    echo "Usage: `basename $0` data_dir dataset_name category_level min_feature_count"
    exit 1
}
if [ $# -ne 4 ]
then
    usage
fi

set +v

data_dir=$1
data_set=$2
category_level=$3
min_feature_count=$4
output=${data_dir}/models/${data_set}.level${category_level}
export_dir=${data_dir}/models/${data_set}
input_train_file=${data_dir}/${data_set}.train
input_test_file=${data_dir}/${data_set}.test
train_tf_record=${data_dir}/${data_set}.train.level${category_level}.tfrecord-1-of-1
test_tf_record=${data_dir}/${data_set}.test.level${category_level}.tfrecord-1-of-1

if [ ! -f ${train_tf_record} ]; then
    echo Processing training dataset file
    python process_input.py \
        --feature_file_prefix=${data_dir}/${data_set} \
        --label_level=${category_level} \
        --min_feature_count=${min_feature_count} 1>log
fi

label_file=${data_dir}/${data_set}.level${category_level}.labels
gold_label_file=${data_dir}/${data_set}.test.gold_label
vocab_file=${data_dir}/${data_set}.level${category_level}.vocab
vocab_size=`cat ${vocab_file} | wc -l | sed -e "s/[ \t]//g"`

echo ${vocab_file}
echo ${vocab_size}

rm -r ${output}
python classifier.py \
    --train_records=${train_tf_record} \
    --eval_records=${test_tf_record} \
    --gold_label_file=${gold_label_file} \
    --label_file=${label_file} \
    --vocab_file=${vocab_file} \
    --vocab_size=${vocab_size} \
    --model_dir=${output} \
    --export_dir=${export_dir} \
    --embedding_dimension=16 \
    --use_attention=True \
    --attention_dimension=8 \
    --learning_rate=0.001 \
    --batch_size=4 \
    --num_epochs=10 \
    --num_threads=2 \
    --per_process_gpu_memory_fraction=0.45 \
    --fast \
    --debug >stat.txt
    #1>log/${embedding_dimension}_${attention_dimension}_${i}.std 2>log/${embedding_dimension}_${attention_dimension}_${i}.err
