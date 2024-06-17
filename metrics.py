import utils
import pandas as pd
import os
import csv
from sklearn.metrics import adjusted_rand_score, accuracy_score, f1_score
from utils import args
from collections import Counter
from extract_answer import extract_answer

def valid(input):
    
    if not hasattr(input, 'valid'):
        return False

    return input.valid == True


def extract_unfinished():
    
    test_data = pd.read_csv(os.path.join('datasets', 'mr', f"test.tsv"), sep='\t', quoting=csv.QUOTE_NONE)

    test_data['id'] = range(len(test_data))
    test_data = test_data.to_dict("records")
    test_data = [ i for i in test_data if '+' not in i['label']]

    finished_data = utils.from_json('outs/maccot/mr/finished.json')
    finished_id = [ d['id'] for d in finished_data ]
    unfinished_id = [ i for i in range(1000) if i not in finished_id ]
    unfinished_data = [test_data[i] for i in unfinished_id]
    utils.save_json(unfinished_data, 'outs/maccot/mr/unfinished.json')

def calculate_topn_metrics(data):

    def sc_acc(data):
        count = 0
        for d in data:
            sc_answer = Counter([item for sublist in d.topn_label_list
                                 for item in sublist]).most_common(1)[0][0]
            if sc_answer == d.label:
                count += 1
        return count / len(data)

    def top_acc(data):
        count = 0
        for d in data:
            if d.label in d.topn_label:
                count += 1
        return count / len(data)

    acc_sl = top_acc(data)
    metrics = {
        "ACC(TOP)": round(acc_sl * 100, 2),
    }
    if 'self_consistency' in args.selector:
        acc_sc = sc_acc(data)
        metrics.update({
            "ACC(SC)": round(acc_sc * 100, 2)
        })
    return metrics

def calculate_metrics(data):
    def top_acc(data):
        count = 0
        for d in data:
            if d.label in d.topn_label:
                count += 1
        return count / len(data)

    def sc_acc(data):
        count = 0
        for d in data:
            sc_answer = Counter([item for sublist in d.topn_label_list
                                 for item in sublist]).most_common(1)[0][0]
            if sc_answer == d.label:
                count += 1
        return count / len(data)

    def process(i):
        if isinstance(i, list):
            if len(i):
                return process(i[0])
            else:
                return 'unk'
        elif isinstance(i, str):
            if '\n\n' in i:
                i = i.split('\n\n')[0]
            return i

    predict = []
    label = []
    acc_sl, acc_sl, acc_sc = 0, 0, 0
    for d in data:
        d['answer'] = process(d['answer'])
    predict = [i['answer'] for i in data]
    label = [i['label'] for i in data]

    ari = adjusted_rand_score(label, predict)
    acc = accuracy_score(label, predict)

    macro_f1 = f1_score(label, predict, average='macro')
    micro_f1 = f1_score(label, predict, average='micro')
    if args.algorithm == 'run_deduced':
        acc_sl = top_acc(data)
    if 'self_consistency' in args.selector:
        acc_sc = sc_acc(data)
    metrics = {
        "ACC": round(acc * 100, 2),
        "MaF1": round(macro_f1 * 100, 2),
        "MiF1": round(micro_f1 * 100, 2),
        "ARI": round(ari * 100, 2),
        "ACC(TOP)": round(acc_sl * 100, 2),
        "ACC(SC)": round(acc_sc * 100, 2),
    }
    return metrics

def analysis_output(
    *, 
    arg_path: str ='outs-0120/banking/topk/args_fewshot_cot_top2_classify.json', 
    out_path: str ='outs-0120/banking/topk/fewshot_cot_top2_classify.json', 
):
    global args
    utils.args = utils.SimpleNamespace(utils.from_json(arg_path))
    args = utils.args

    data = utils.from_json(out_path)
    data = [ utils.SimpleNamespace(**d) for d in data ]
    old_len = len(data)

    data = [ d for d in data if d.label in d.topn_label ]
    new_len = len(data)

    print(f'skip error: {new_len}/{old_len} = {new_len/old_len}')

    # check the answer format
    for d in data:
        if d.answer != d.label :
            print(d.answer)
    
    import pdb; pdb.set_trace()
    # data = [d for d in data if d.answer != d.label and d.label in d.topn_label ]
    for d in data:
        if 'compare_messages' not in d:
            continue
        for conv in d['compare_messages']:
            predict = extract_answer(conv[-1]['content'], d.topn_label + d.remove_labels, 1)
            if len(predict) == 0:
                import pdb; pdb.set_trace()
                predict = extract_answer(conv[-1]['content'], d.topn_label + d.remove_labels, 1, filter=False)    
            else:        
                d.answer = predict[0]

        if d.answer != d.label and d.label in conv[-1]['content'][-30:]:
            import pdb; pdb.set_trace()

    print(calculate_metrics(data))

    data = [ d for d in data if d.label in d.topn_label and d.answer != d.label and d.label in d.remove_labels  ]

    utils.save_json(data, 'outs-0120/fewshot_cot_bad_case.json')


if __name__ == '__main__':
    import defopt
    try:
        defopt.run(analysis_output)
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)