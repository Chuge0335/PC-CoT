#!/bin/bash

for dataset in banking hwu64 clinc liu54;do
    CUDA_VISIBLE_DEVICES=0 python vectorization.py --dataset $dataset
done