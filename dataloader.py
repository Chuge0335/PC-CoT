from typing import Any
from utils import args
import utils
import os
import sys
import copy
import pandas as pd
import csv
import random
from collections import defaultdict
from metrics import calculate_metrics, calculate_topn_metrics
import logging

LOG = logging.getLogger(os.path.basename(__file__))

# for query dataset
def load(task, context):
    choice = -1
    if os.path.exists(f'{task.outs}/{task.prompt}.json'):
        LOG.warning(f'there are already exists files!! {task.outs}/{task.prompt}.json')
        while True:
            choice = input('\n1. skip\n2. overwrite\n3. run unfinished\n')
            try:
                choice = int(choice)
                break
            except:
                print('wrong format! input 1 or 2 or 3')
        
        if choice == 1:
            sys.exit(0)
        
        if choice == 3:
            if not os.path.exists(f'{task.outs}/{task.prompt}-unfinished.json'):
                sys.exit(0)
    
    elif os.path.exists(f'{task.outs}/{task.prompt}-unfinished.json'):
        while True:
            choice = input('\n1. skip\n2/3/... run unfinished\n')
            try:
                choice = int(choice)
                break
            except:
                print('wrong format! input 1 or other number')

        if choice == 1:
            sys.exit(0)

        choice = 3

    if choice == 3:

        test_data = utils.from_json(
            f'{task.outs}/{task.prompt}-unfinished.json'
        )
    elif task.name == 'classify' or '_reducer' in task.name or 'test' in task.name:

        if os.getenv('HARD'):
            is_hard = os.getenv('HARD')
            test_data = pd.read_csv(
                os.path.join('datasets', args.data, f"test_hardcase.tsv"), sep='\t', quoting=csv.QUOTE_NONE
            )
            if is_hard=='0':
                all_data = pd.read_csv(
                os.path.join('datasets', args.data, f"test.tsv"), sep='\t', quoting=csv.QUOTE_NONE
                )
                test_data = all_data[~all_data['text'].isin(test_data['text'])].dropna()
        else:
            test_data = pd.read_csv(
                os.path.join('datasets', args.data, f"test.tsv"), sep='\t', quoting=csv.QUOTE_NONE
            )
        # import pdb;pdb.set_trace()
        # TODO: make sure order is right
        # test_data['id'] = range(len(test_data))
        test_data = test_data.astype(str).to_dict("records")
        test_data = [i for i in test_data if '+' not in i['label']]
        options = list(set([i['label'] for i in test_data if '+' not in i['label']]))
    
        def sort_fun(text):
            return len(text), text
        
        options = sorted(options, key=sort_fun, reverse=True)
        if 'day_name' in options:
            options.remove('day_name')
        if 'test' in task.name:
            from utils import get_diverse_and_similar_labels
            similary_diversity_label = get_diverse_and_similar_labels(options, args.data, num_labels=args.option_num)
        for data in test_data:
            data["text"] = data["text"][:args.max_len]
            data['options'] = copy.deepcopy(options)
            if 'test' in task.name:
                data['diverse_labels'] = similary_diversity_label[data['label']][0]
                data['similar_labels'] = similary_diversity_label[data['label']][1]
    elif 'demo' in task.name:
        def sort_fun(s):
            return len(s),s
        
        if os.getenv('HARD'):
            is_hard = os.getenv('HARD')
            test_data = pd.read_csv(
                os.path.join('datasets', args.data, f"test_hardcase.tsv"), sep='\t', quoting=csv.QUOTE_NONE
            )
            if is_hard=='0':
                all_data = pd.read_csv(
                os.path.join('datasets', args.data, f"test.tsv"), sep='\t', quoting=csv.QUOTE_NONE
                )
                test_data = all_data[~all_data['text'].isin(test_data['text'])].dropna()
            test_data = test_data.to_dict('records')
        else:
            dataset = pd.read_csv(
                os.path.join('datasets', args.data, f"test.tsv"), sep='\t', quoting=csv.QUOTE_NONE
            ).to_dict("records")
        options = list(set([i['label'] for i in dataset if '+' not in i['label']]))
        label_set = set()
        test_data = []
        for data in dataset:
            label_set.add(data["label"])

        label_list = list(label_set)

        label_list = sorted(label_list,key=sort_fun,reverse=True)

        for i in range(0,(len(label_list)//2)*2,2):
            test_data.append({"label1":label_list[i],"label2":label_list[i+1],"options":options})
        if len(label_list)%2:
            test_data.append({"label1":label_list[i+1],"label2":label_list[i+2],"options":options})
            
    elif 'similar' in task.name or 'differ' in task.name or 'fewshot_maccot_explain' == task.name:
        test_data = utils.from_json(f'datasets/{args.data}/boundary_sample.json')
        test_data = [
            {
                'text1': d[0],
                'text2': d[1],
                'label1': l.split('&')[0],
                'label2': l.split('&')[1],
            } for l in test_data.keys() for d in test_data[l]
        ]
    
    elif 'explain' in task.name:
        test_data = utils.from_json(f'datasets/{args.data}/{args.sample}_sample.json')
        if args.sample=='boundary':
            test_data = conver(test_data)
        
        test_data = [
            {
                'text': d,
                'label': l
            } for l in test_data.keys() for d in test_data[l][:10]
        ]
    else:
        raise NotImplementedError
    test_data = [utils.SimpleNamespace(d) for d in test_data]
    if args.subset is not None:
        random.shuffle(test_data)
        tmp = args.subset.split(':')
        start, end = int(tmp[0]), int(tmp[1])
        test_data = test_data[start:end]

                
    if context is not None:
        # insert history data
        for n, vs in context :
            if len(test_data) != vs:
                # (boundary_sample) similar -> (random_sample) explain -> (test.csv) test_data
                pass
            for d,v in zip(test_data, vs):
                d[n] = v 
                
    return test_data

def save(task, results, context):
    if 'demo' in task.name:
        results_fromat = {}

        for result in results:
            results_fromat.update({result.label1:result.text1})
            results_fromat.update({result.label2:result.text2})
                
        utils.save_json(results_fromat, f'datasets/{args.data}/{args.sample}_generation.json')
        return 

    if task.name == 'classify' and 'unfinished' not in task.prompt or 'test' in task.name:
        metrics = calculate_metrics(results)
        utils.save_excel({**args, **metrics}, args.outs, name='compare-results')
    
    elif '_reducer' in task.name and 'unfinished' not in task.prompt:
        metrics = calculate_topn_metrics(results)
        utils.save_excel({**args, **metrics}, args.outs, name='reduce-results')

    utils.save_json(args.__dict__, f'{task.outs}/args_{task.prompt}.json')
    # for unfinished rerun
    if os.path.exists(f'{task.outs}/{task.prompt}.json'):
        LOG.info(f'rewrite to {task.outs}/{task.prompt}.json !!')
        if 'unfinished' in task.prompt:
            utils.save_json(results, f'{task.outs}/{task.prompt}.json', 'w')
        else:   
            utils.save_json(results, f'{task.outs}/{task.prompt}.json', 'a')
    else:
        if task.prompt == 'fewshot_cot_explain':
            utils.save_json(results, f'datasets/{args.data}/{task.prompt}.json')
        else:
            utils.save_json(results, f'{task.outs}/{task.prompt}.json')
        LOG.info(f'save to {task.outs}/{task.prompt}.json')

class topn_label_loader():

    def __init__(self, selector_dir='raw', ):

        self.selector_dir = selector_dir
        if selector_dir == 'raw':
            
            # don't do anything
            return

        # bert self_consistency topk self_consistency_remove topk_remove
        cache_path = f'{args.outs}/{args.data}/{selector_dir}/topn_label.json'

        if not os.path.exists(cache_path):
            raise Exception('topn_label file not exist!!')
        
        select_labels = utils.from_json(cache_path)
        self.topn_label = { str(l['id']) : l['topn_label'] for l in select_labels }

    def __call__(self, input):

        if self.selector_dir == 'raw':
            # don't do anything
            return

        input.topn_label = self.topn_label[str(input.id)]

# for fewshot demonstration data
class fewshot_raw:
    def __init__(self) -> None:
        pass

    def __call__(self, input) -> Any:
        return None
    
def conver(contents):
    data_buffer = defaultdict(list)
    for k,v in contents.items():
        names = k.split('&')
        for i in v:
            data_buffer[names[0]].append(i[0])
            data_buffer[names[1]].append(i[1])
    return data_buffer

class fewshot_classify:

    def __init__(self, ):
        self.demos = utils.from_json(
            f"datasets/{args.data}/{args.sample}_sample.json"
        )
        if args.sample=='boundary':
            self.demos = conver(self.demos)

    def __call__(self, input):

        demos = random.sample([
            {
                'text': d,
                'label': l
            } for l in input.options
            for d in random.sample(self.demos[l], args.shot)
        ], 5)
        return demos

class fewshot_top2_classify(fewshot_classify):

    def __call__(self, input):
        # avoid too long , only select 5 
        demos = random.sample([
            {
                'text': d,
                'label': l
            } for l in [input.label1, input.label2]
            for d in random.sample(self.demos[l], args.shot)
        ], 5)
        return demos

class fewshot_topn_classify(fewshot_classify):

    def __call__(self, input):

        demos = [
            {
                'text': d,
                'label': l
            } for l in input.topn_label
            for d in random.sample(self.demos[l], args.shot)
        ]
        return demos

class fewshot_cot_classify:

    def __init__(self) -> None:
        self.demos = utils.from_json(
            f'datasets/{args.data}/fewshot_cot_explain.json'
        )
        tmp = {}
        for demo in self.demos:
            demo['explain'] = demo['answer']
            if demo['label'] not in tmp:
                tmp[demo['label']] = []
            tmp[demo['label']].append(demo)
        self.demos = tmp

    def __call__(self, input):
        # avoid too long , only select 5 
        return random.sample([
            ll for l in input.options for ll in random.sample(self.demos[l], args.shot)
        ], 5)

class fewshot_cot_topn_classify(fewshot_cot_classify):

    def __call__(self, input):
        return [
            ll for l in input.topn_label for ll in random.sample(self.demos[l], args.shot)
        ]

class fewshot_cot_top2_classify(fewshot_cot_classify):

    def __call__(self, input):
        return [
            ll for l in [input.label1, input.label2] for ll in random.sample(self.demos[l], args.shot)
        ]

class maccot_classify:

    def __init__(self, ):
        # args.sample: boundary center random
        self.demos = utils.from_json(
            f"datasets/{args.data}/{args.sample}_sample.json"
        )

    def __call__(self, input):

        if args.sample == 'boundary':
            try:
                key = input.label1 + '&' + input.label2
                key2 = input.label2 + '&' + input.label1

                if self.demos.get(key, False):
                    sentence_a = [
                        example[0] for example in self.demos[key]
                    ]
                    sentence_b = [
                        example[1] for example in self.demos[key]
                    ]
                else:
                    sentence_a = [
                        example[1] for example in self.demos[key2]
                    ]
                    sentence_b = [
                        example[0] for example in self.demos[key2]
                    ]
            except:
                sentence_a = self.demos[input.label1]
                sentence_b = self.demos[input.label2]
            
        else:
            sentence_a = self.demos[input.label1]
            sentence_b = self.demos[input.label2]
            
        sentence_a = random.sample(sentence_a, args.shot)
        sentence_b = random.sample(sentence_b, args.shot)
        
        demos = []
        for sa, sb in zip(sentence_a, sentence_b):
            demos.append({
                'text1': sa,
                'text2': sb,
                'label1': input.label1,
                'label2': input.label2,
            })
        if len(demos) == 0:
            raise Exception('demos is null !')
        return demos

maccot_wo_sim_classify=maccot_classify
maccot_wo_diff_classify=maccot_classify
maccot_wo_simdiff_classify=maccot_classify

class fewshot_maccot_classify:

    def __init__(self, ):
        self.demos = utils.from_json(
            f"datasets/{args.data}/{args.sample}_sample.json"
        )

    def __call__(self, input):

        for cs in self.demos:
            if (
                cs['label1'] == input.label1 
            and cs['label2'] == input.label2
            ):
                return cs
            elif (
                cs['label2'] == input.label1 
            and cs['label1'] == input.label2
            ):
                return cs