#!/bin/bash
export TRAIN_FILE=corpus/ro_train.txt
export TEST_FILE=corpus/ro_test.txt
python3 run_electra_pretraining.py --train_data_file $TRAIN_FILE --eval_data_file $TEST_FILE --output_dir ./ElectraModel  --discriminator_config_name ./ElectraConfig/config_discriminator.json --generator_config_name ./ElectraConfig/config_generator.json --tokenizer_name ./ElectraConfig --do_train --do_eval --learning_rate 1e-5 --num_train_epochs 5 --save_total_limit 20 --save_steps 5000 --warmup_steps=1000 --logging_steps=100 --gradient_accumulation_steps=4 --seed 666 --block_size=512 --no_cuda
