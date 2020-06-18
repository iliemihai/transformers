#!/bin/bash
export TRAIN_FILE=corpus/encoded_train.txt
export TEST_FILE=corpus/encoded_valid.txt

export SAVED_MODELS=$(ls "./ElectraModel" -t1 |  head -n 1)
export DISCRIMINATOR="./ElectraModel"/${SAVED_MODELS}/"discriminator"
export GENERATOR="./ElectraModel"/${SAVED_MODELS}/"generator"
 
if [ -z "$(ls -A './ElectraModel')" ];then 
    python3 run_electra_pretraining.py --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --train_data_file $TRAIN_FILE --eval_data_file $TEST_FILE --output_dir "./ElectraModel"  --discriminator_config_name "./ElectraConfig/config_discriminator.json" --generator_config_name "./ElectraConfig/config_generator.json" --tokenizer_name "./ElectraConfig" --do_train --do_eval --learning_rate 1e-5 --num_train_epochs 5 --save_total_limit 20 --save_steps 5000 --warmup_steps=1000 --logging_steps=100 --gradient_accumulation_steps=1 --seed 666 --block_size=512 --save_steps 50
else 
    python3 run_electra_pretraining.py --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --train_data_file $TRAIN_FILE --eval_data_file $TEST_FILE --output_dir "./ElectraModel" --discriminator_name_or_path $DISCRIMINATOR --generator_name_or_path $GENERATOR --tokenizer_name "./ElectraConfig" --do_train --do_eval --learning_rate 1e-5 --num_train_epochs 5 --save_total_limit 20 --save_steps 50000 --warmup_steps=1000 --logging_steps=100 --gradient_accumulation_steps=1 --seed 666 --block_size=512 --save_steps 50 --overwrite_output_dir 
fi
