import json
import os
import random
import torch
import time
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, accuracy_score, f1_score
import yaml
from tabulate import tabulate
import csv
import pickle
import functools
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class SimpleNamespace:

    def __init__(self, _data_ = None, **kwargs):
        if isinstance(_data_, dict):
            self.__dict__.update(**_data_) 
        self.__dict__.update(kwargs)

    def __delattr__(self, name):
        if hasattr(self, name):
            super().__delattr__(name)

    def __getitem__(self, item):
        return self.__dict__[item]
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, item):
        return item in self.__dict__
    
    def __delitem__(self, item):
        self.__dict__.__delitem__[item]

    def __len__(self):
        return len(self.__dict__)
    
    def __iter__(self,):
        return iter(self.__dict__)

    def copy(self):
        return type(self)(**self.__dict__)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def pop(self, key, default=None):
        return self.__dict__.pop(key, default)

    def popitem(self):
        return self.__dict__.popitem()
    
    def clear(self):
        self.__dict__.clear()
    
    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def items(self):
        return self.__dict__.items()
    
    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items() if v is not self)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        if isinstance(self, SimpleNamespace) and isinstance(other, SimpleNamespace):
           return self.__dict__ == other.__dict__
        return NotImplemented

# cannot reassign
args = SimpleNamespace()

random._inst.old_sample=random.sample

def new_sample(population, k, *, counts=None):
    if len(population) < k:
        k = len(population)

    return random._inst.old_sample(population, k, counts=counts)

setattr(random,'sample', new_sample)

def to_jsonl(data, path, mode='w'):
    if not isinstance(data, list):
        data = [data]
    with open(path, mode) as f:
        for line in data:
            f.write(json.dumps(line,ensure_ascii=False)+'\n')
    print("Save data to: {}".format(path))

def to_tsv(data, path, mode='w'):
    if not isinstance(data, list):
        data = [data]
    with open(path, mode) as f:
        f.write('text\tlabel\n')
        for line in data:
            f.write(line['text']+'\t'+line['label']+'\n')
    print("Save data to: {}".format(path))

def get_class_info(path):
    class_list = []
    class_dict = {}
    save_path = os.path.join(os.path.dirname(path),'name2id.json')
    data = json.load(open(path))

    for content in data:
        class_list.append(content["label"])
    class_set = set(class_list)
    
    for i, class_name in enumerate(class_set):
        class_dict[class_name] =  i

    with open(save_path,'w') as f:
        json.dump(class_dict,f,indent=2,ensure_ascii=False)
    print("Save class info to: {}".format(save_path))

    return 
    
def pwd(file):
    return os.path.dirname(os.path.abspath(file))

def top_acc(data):
    count = 0
    for d in data:
        if d.get('predict_list',False):
            if d['label'] in d['predict_list']:
                count += 1
        elif d.get('top2_answer',False):
            if d['label'] in d['top2_answer']:
                count += 1
        else:
            if d['label'] in d['predict']:
                count += 1
    return count/len(data)

def calculate_metrics(data,args):
    predict = []
    label = []
    
    predict=[ i["predict"] for i in data]
    label = [ i['label'] for i in data]

    ari = adjusted_rand_score(label, predict)
    
    acc = accuracy_score(label, predict)
    # Assuming y_true and y_pred are your true labels and predicted labels
    macro_f1 = f1_score(label, predict, average='macro')
    micro_f1 = f1_score(label, predict, average='micro')
    metrics = {"ACC": round(acc * 100, 2), "MaF1":macro_f1, "MiF1":micro_f1, "ARI": round(ari * 100, 2)}
    if args.method == 'ccot':
        acc=top_acc(data)
        metrics['ACC'] = str(metrics['ACC']) + '/' + str(round(acc*100,2))
    print(metrics)
    return metrics

def save_excel(data, out_path, name='results'):
    # save excel
    df = pd.DataFrame(data, index=[0])

    xlsx_path = os.path.join(out_path,f'{name}.xlsx')
    md_path = os.path.join(out_path,f'{name}.md')

    if os.path.exists(xlsx_path):
        previous = pd.read_excel(xlsx_path,index_col=0)
        df = pd.concat([previous,df])

    df.to_excel(xlsx_path, index=True)

    markdown_table = tabulate(df, headers='keys', tablefmt='pipe')
    print(markdown_table)
    print(markdown_table, file=open(md_path, 'w'))

def save_result(predict,args):
    metrics_path = os.path.join(args.save_dir,'result.xlsx')
    save_path = get_save_path(args.save_dir,args.save_name)
    if args.compute_metrics:
        metrics = calculate_metrics(predict,args)
        file_name, file_extension = os.path.splitext(os.path.basename(save_path))
        save_info = {
            "name":file_name,
            "dataset":args.dataset,
            "mode":args.mode, 
            "sample": args.sample if args.sample is not None else "all",
            "method":args.method, 
            "deep":args.deep,
            "model":args.model,
            "sample_method":args.sample_method,
            "ablation":args.ablation,
            "selector":args.selector,
            "zero-shot":args.zero,
            "prompt": args.prompt_path,
            }
        save_excel({**save_info, **metrics},metrics_path)
    
    save_json(predict, save_path)
    return

def data_preprocess(dataset,args):
    if not args.multi_cls:
        dataset = [ i for i in dataset if '+' not in i['label']]
    if args.sample is not None:
        dataset = dataset[:args.sample]
    return dataset

def save_json(output,save_path, mode='w'):

    def simple_parser(obj):
        if isinstance(obj, SimpleNamespace):
            return vars(obj)
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    if mode == 'a':
        data = from_json(save_path)
        output += (data)
    with open(save_path,'w') as f:
        json.dump(output,f,indent=2,ensure_ascii=False, default=simple_parser)
    print(f"Save json to: {save_path}")
    return


def get_save_path(save_dir,prefix=None,posfix='.jsonl'):
    current_time = time.localtime()
    formatted_time = time.strftime("%m%d-%H%M", current_time)
    save_name = ""
    if prefix:
        save_name += prefix 
    save_name += (formatted_time + posfix)
    save_path = os.path.join(save_dir,save_name)
    return save_path


def from_yaml(path,):
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.load(file, yaml.SafeLoader)
    return data

def from_json(path):
    return json.load(open(path))

def from_jsonl(path):
    return [json.loads(line) for line in open(path, 'r',encoding='utf8') ]


def extract_label_info():

    label_infos = {}
    for file in os.listdir('datasets'):

        if file == '20news' or file == 'dbpedia':
            continue

        if os.path.exists(f'datasets/{file}/train.tsv'):
            train_labels = set()
            test_labels = set()
            train_data = pd.read_csv(
                os.path.join('datasets', file, f"train.tsv"), sep='\t', quoting=csv.QUOTE_NONE
            ).to_dict("records")
            test_data = pd.read_csv(
                os.path.join('datasets', file, f"test.tsv"), sep='\t', quoting=csv.QUOTE_NONE
            ).to_dict("records")
            for d in train_data:
                train_labels.add(d['label'])
            for d in test_data:
                test_labels.add(d['label'])
            label_infos[file] = {
                'train_labels': list(train_labels),
                'test_labels': list(test_labels),
                'len_train_labels': len(train_labels),
                'len_test_labels': len(test_labels),
                'len_train': len(train_data),
                'len_test': len(test_data),
            }

    save_json(label_infos, f'label_info.json')

def reload():
    import utils
    import importlib
    importlib.reload(utils)

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# for `classifier.dense.out_proj` nest subojects / chained properties
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def get_diverse_and_similar_labels(labels, data, num_labels=16):
    with open(f'datasets/{data}/label_vectors.pkl', 'rb') as file:
        label_to_vector = pickle.load(file)
    label_to_vector = {k:v for k,v in label_to_vector.items() if k in labels}
    result_dict = {}
    all_vectors = np.array(list(label_to_vector.values()))
    kmeans = KMeans(n_clusters=num_labels)
    cluster_assignments = kmeans.fit_predict(all_vectors)
    for label in labels:
        target_vector = label_to_vector[label]

        tgt_cluster_index = list(label_to_vector.keys()).index(label)
        tgt_cluster_assignment = cluster_assignments[tgt_cluster_index]

        diverse_labels = []

        for cluster_index in range(num_labels):
            if cluster_index != tgt_cluster_assignment:
                cluster_center = np.mean(all_vectors[cluster_assignments == cluster_index], axis=0)
                distances = np.linalg.norm(all_vectors[cluster_assignments == cluster_index] - cluster_center, axis=1)
                nearest_index = np.argmin(distances)
                cluster_labels = np.array(list(label_to_vector.keys()))[cluster_assignments == cluster_index]
                diverse_labels.append(cluster_labels[nearest_index])

        aother_vectors = [v for k,v in label_to_vector.items() if k!=label]
        similarities = cosine_similarity([target_vector], aother_vectors)[0]
        similar_labels_idx = np.argsort(similarities)[::-1][:num_labels-1]
        similar_labels = [list(label_to_vector.keys())[i] for i in similar_labels_idx]
        result_dict[label] = [diverse_labels, similar_labels]
    return result_dict

if __name__ == "__main__":
    try:
        extract_label_info()
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)
