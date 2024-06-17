#!/bin/bash

# model type
export model='gpt-3.5-turbo'

# chat or completion
export API_MODE='chat'

# save path
export OUTS='output'

# Whether to use a hardship data
export HARD=1

#debug mode
# export DEBUG=1
# export DRYRUN=1

# subset range
export subset=0:1000

# example nums
export shot=3

# reduce method, can select sliding_window, topk, self-consistency
export select=topk

# full options method
function run_full(){
    for prompt in fewshot_classify zero_shot_classify zero_shot_cot_classify;do

        python run.py \
        --prompt $prompt \
        --model $model \
        --data $data \
        --subset $subset \
        --shot $shot \
        --algorithm 'run_options' \
        --selector 'raw'

    done

}

# options reduce
function run_selector(){
    python run.py \
    --model $model \
    --data $data \
    --subset $subset \
    --selector $select \
    --select-iter 5
}

# reduce-based method
function run_deduced_top2(){

    for prompt in maccot_classify zero_shot_top2_classify zero_shot_cot_top2_classify fewshot_top2_classify fewshot_cot_top2_classify; do
        python run.py \
        --prompt $prompt \
        --model $model \
        --data $data \
        --subset $subset \
        --shot $shot \
        --algorithm 'run_deduced' \
        --selector $select

    done
}

# predict
for datasest in banking hwu64 clinc liu54;do
    export data=$datasest
    # non-reduce
    run_full
    # reduce
    run_selector
    run_deduced_top2
done