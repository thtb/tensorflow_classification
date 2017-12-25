#!/bin/bash

usage()
{
    echo "Usage: `basename $0` data_dir dataset_name category_level"
    exit 1
}
if [ $# -ne 3 ]
then
    usage
fi

set +v

data_dir=$1
data_set=$2
category_level=$3
output=${data_dir}/models/${data_set}.level${category_level}
export_dir=${data_dir}/models/${data_set}
input_train_file=${data_dir}/${data_set}.train
input_test_file=${data_dir}/${data_set}.test
train_tf_record=${data_dir}/${data_set}.train.level${category_level}.tfrecord-1-of-1
test_tf_record=${data_dir}/${data_set}.test.level${category_level}.tfrecord-1-of-1

if [ ! -f ${train_tf_record} ]; then
    echo Processing training dataset file
    python process_input.py --feature_file_prefix=${data_dir}/${data_set} --label_level=${category_level} 1>log
fi

label_file=${data_dir}/${data_set}.level${category_level}.labels
gold_label_file=${data_dir}/${data_set}.test.gold_label
vocab_file=${data_dir}/${data_set}.level${category_level}.vocab
vocab_size=`cat ${vocab_file} | wc -l | sed -e "s/[ \t]//g"`

echo ${vocab_file}
echo ${vocab_size}

num_epochs=5
echo "TRAIN" >>stat/${batch_size}_${num_epochs}.stat

for (( batch_size=8; batch_size<=8; batch_size=batch_size+2 )); do
    echo "TRAIN" >>stat/${batch_size}_${num_epochs}.stat
    for (( embedding_dimension=32; embedding_dimension<=32; embedding_dimension=embedding_dimension+8 )); do
      for (( attention_dimension=4; attention_dimension<=4; attention_dimension=attention_dimension*2 )); do
        echo "${embedding_dimension} ${attention_dimension}" >> stat/${batch_size}_${num_epochs}.stat
        for (( i=0; i<=0; i=i+1 )); do
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
              --embedding_dimension=${embedding_dimension} \
              --attention_dimension=${attention_dimension} \
              --learning_rate=0.01 \
              --batch_size=${batch_size} \
              --train_steps=1000 \
              --eval_steps=100 \
              --num_epochs=${num_epochs} \
              --num_threads=2 \
              --nolog_device_placement \
              --per_process_gpu_memory_fraction=0.3 \
              --fast \
              --debug
              #1>log/${embedding_dimension}_${attention_dimension}_${i}.std 2>log/${embedding_dimension}_${attention_dimension}_${i}.err
        done
      done
    done
done
