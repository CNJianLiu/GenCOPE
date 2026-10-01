#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=2

# # python evaluate_SPD_on_Wild6D.py --use_nocs_map --implicit
python test_wild6d.py --use_nocs_map --implicit --model 'log/backbone/GenCOPE' --select_class 'bottle'
python test_wild6d.py --use_nocs_map --implicit --model 'log/backbone/GenCOPE' --select_class 'bowl'
python test_wild6d.py --use_nocs_map --implicit --model 'log/backbone/GenCOPE' --select_class 'mug'
python test_wild6d.py --use_nocs_map --implicit --model 'log/backbone/GenCOPE' --select_class 'camera' 
python test_wild6d.py --use_nocs_map --implicit --model 'log/backbone/GenCOPE' --select_class 'laptop'
