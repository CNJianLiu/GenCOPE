#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=2

# python evaluate_SPD_on_Wild6D.py --use_nocs_map --implicit
# python visualize_wild6d.py --select_class 'bottle'
# python visualize_wild6d.py --select_class 'bowl'
# python visualize_wild6d.py --select_class 'mug'
python visualize_wild6d.py --select_class 'camera' --test_epoch '10'
# python visualize_wild6d.py --select_class 'laptop'