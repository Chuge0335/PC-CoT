#!/bin/bash

for dataset in banking hwu64 clinc liu54; do
    CUDA_VISIBLE_DEVICES=0 python -m get_fewshot --dataset $dataset
done