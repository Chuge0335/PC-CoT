import copy
import utils
import pickle
import random
import logging
import numpy as np
import openai_client
from utils import args
from collections import Counter
from sklearn.cluster import KMeans
from construct_prompt import construct_prompt
from extract_answer import extract_answer, extract_list

LOG = logging.getLogger('ray')


def most_common_elements(candidate, top=3):

    flat_data = [item for sublist in candidate for item in sublist]

    counter = Counter(flat_data)

    most_common_elements = counter.most_common(top)
    result_list = [element for element, count in most_common_elements]
    return result_list


class LabelSelector:
    def __init__(self, engine):

        self.engine = engine

    def postprocess(self, input):

        input.valid = hasattr(input, 'topn_label') and len(input.topn_label) > 0

        if input.label not in input.topn_label:
            input.wrong = True

        self.engine.clear()


    def __call__(self, input):
        if not len(input.options):
            print('error input without options!', input)
            raise openai_client.OpenaiAPIDataItemError
        
        if len(input.options) < 5:
            # label is small enough
            input.topn_label = input.options
            input.valid = True
            return input

        for _ in range(2):
            try:
                self.engine.clear()
                self.select(input)
                self.postprocess(input)
                self.engine.clear()
                break
            except openai_client.OpenaiAPIRestartError:
                LOG.error('has error. restart.')
                pass

        return input


class raw(LabelSelector):
    def select(self, input):
        pass


class no(LabelSelector):
    def select(self, input):
        pass

    def __call__(self, input):
        input.valid = True
        pass


class bert(LabelSelector):
    def __init__(self, engine):
        super().__init__(engine)
        if args.selector == 'bert':
            # full sft bert select top 2 labels
            self.top2_labels = utils.from_jsonl(f'datasets/{args.data}/bert_prediction.jsonl')
            self.top2_labels = {t['id']: t['predicted_labels'] for t in self.top2_labels}

    def select(self, input):

        input.select_messages = []
        input.topn_label = self.top2_labels[input.id]
        input.topn_label_list = [input.topn_label]

        return input


class self_consistency(LabelSelector):
    def select(self, input):

        topn_label_list = []
        select_messages = []
        unseen_labels = []
        options = copy.deepcopy(input.options)

        i = 0
        while i < args.select_iter:

            self.engine.clear()

            input.options = copy.deepcopy(options)
            random.shuffle(input.options)

            if len(input.options) == 1:
                topn_label_list.append(input.options)
                select_messages.append(self.engine.messages)
                i += 1
                continue

            # select related label
            topn_label = []
            input.text = input.text[-2048:]
            query = construct_prompt('sc_prompt', input)[0]
            resp = self.engine.generate(query)
            topn_label, unseen_label = extract_list(resp, input.options)

            if len(unseen_label):
                unseen_labels.append(unseen_label)

            if len(topn_label) == 0:
                continue

            topn_label_list.append(topn_label)
            select_messages.append(self.engine.messages)
            i += 1

        input.topn_label_list = topn_label_list
        input.unseen_label = unseen_labels
        input.select_messages = select_messages
        input.topn_label = most_common_elements(topn_label_list, 3)

        return input


class self_consistency_remove(LabelSelector):
    def select(self, input):

        topn_label_list = []
        select_messages = []
        unseen_labels = []
        options = copy.deepcopy(input.options)

        i = 0
        while i < args.select_iter:

            self.engine.clear()

            input.options = copy.deepcopy(options)
            random.shuffle(input.options)

            # 1. remove unrelated label
            input.text = input.text[-2048:]
            query = construct_prompt('label_filter_options_prompt', input)[0]
            resp = self.engine.generate(query)
            removeset, _ = extract_list(resp, input.options)

            if len(removeset) == len(input.options):
                continue

            for r in removeset:
                try:
                    input.options.remove(r)
                except:
                    pass

            select_messages.append(self.engine.messages)

            if len(input.options) == 1:
                topn_label_list.append(input.options)
                select_messages.append(self.engine.messages)
                i += 1
                continue

            self.engine.clear()

            # 2. select related label
            topn_label = []
            input.text = input.text[-2048:]
            query = construct_prompt('sc_prompt', input)[0]
            resp = self.engine.generate(query)
            topn_label, unseen_label = extract_list(resp, input.options)

            if len(unseen_label):
                unseen_labels.append(unseen_label)

            if len(topn_label) == 0:
                continue

            topn_label_list.append(topn_label)
            select_messages.append(self.engine.messages)
            i += 1

        input.topn_label_list = topn_label_list
        input.unseen_label = unseen_labels
        input.select_messages = select_messages
        input.topn_label = most_common_elements(topn_label_list, 3)

        return input


class topk(LabelSelector):
    def select(self, input):
        topn_label_list = []
        select_messages = []

        old_options = copy.deepcopy(input.options)
        options = copy.deepcopy(input.options)

        i = 0
        r = 0
        while i < args.select_iter:
            if r > 4:
                topn_label_list = [input.options[0]]
                break
                # raise openai_client.OpenaiAPIDataItemError
            if len(options) == 0:
                break

            self.engine.clear()

            input.options = copy.deepcopy(options)
            random.shuffle(input.options)

            topn_label = []

            # here only let model extract top1 label
            input.text = input.text[-2048:]
            query = construct_prompt('zero_shot_classify', input)[0]
            resp = self.engine.generate(query)
            topn_label = extract_answer(resp, input.options, 1)

            if len(topn_label) == 0:
                self.engine.clear()
                input.text = input.text[-2048:]
                query2 = construct_prompt('zero_shot_classify2', input)[0]
                resp2 = self.engine.generate(query2)
                topn_label = extract_answer(resp2, input.options, 2)
                if len(topn_label) == 0:
                    if len(topn_label_list) > 0:
                        break
                    r += 1
                    continue
            if len(topn_label) != 0:
                topn_label_list.append(topn_label[:1])
                select_messages.append(self.engine.messages)
                options.remove(topn_label[0])
                r = 0
                i += 1

        input.topn_label_list = topn_label_list
        input.select_messages = select_messages
        input.options = old_options
        input.topn_label = most_common_elements(topn_label_list, args.select_iter)
        return input


class sliding_window(LabelSelector):
    def __init__(self, engine):

        self.engine = engine

        with open(f'datasets/{args.data}/label_vectors.pkl', 'rb') as file:
            self.label_to_vector = pickle.load(file)

    def select(self, input):
        input.curr_label_selected = []
        old_options = copy.deepcopy(input.options)
        curr_labels = copy.deepcopy(input.options)
        select_messages = []

        def get_winds(cluster_dict, select_similary=False):
            selected_options = []
            if not select_similary:
                for labels in cluster_dict.values():
                    selected_options.append(labels[0])
            else:
                cluster_dict = dict(
                    sorted(cluster_dict.items(), key=lambda k: len(k[1]), reverse=True)
                )
                cluster_elements = [i for j in cluster_dict.values() for i in j]
                selected_options = cluster_elements[:len(cluster_dict.keys())]

            return selected_options

        def list_differ(list1, list2):
            set1 = set(list1)
            set2 = set(list2)
            difference = set1.difference(set2)
            result_list = list(difference)
            return result_list

        K = 16
        input.top_k = 3
        while len(curr_labels) > args.select_num:

            selected_vectors = [self.label_to_vector[label] for label in curr_labels]
            selected_vectors_array = np.array(selected_vectors)

            kmeans = KMeans(n_clusters=K,n_init='auto')
            cluster_labels = kmeans.fit_predict(selected_vectors_array)

            cluster_dict = {}
            for label, cluster_label in zip(curr_labels, cluster_labels):
                if str(cluster_label) not in cluster_dict:
                    cluster_dict[str(cluster_label)] = [label]
                else:
                    cluster_dict[str(cluster_label)].append(label)

            curr_selected = []

            input.options = get_winds(cluster_dict, args.select_similary)

            if len(input.options) <= input.top_k:
                curr_selected.extend(input.options)
                continue

            j = 0
            while j < args.select_iter:

                self.engine.clear()

                random.shuffle(input.options)

                topn_label = []

                # here only let model extract top1 label
                input.text = input.text[-2048:]
                query = construct_prompt('zero_shot_top_classify', input)[0][-2048:]
                resp = self.engine.generate(query)
                topn_label = extract_answer(
                    resp.replace("cognized", "cognised"), input.options, input.top_k, False
                )
                if len(topn_label) != 0:
                    curr_selected.extend(topn_label)
                    select_messages.append(self.engine.messages)
                    curr_labels = list_differ(curr_labels, list_differ(input.options, topn_label))
                    break
                j += 1
            if j >= args.select_iter:
                curr_selected.extend([None])
                select_messages.append(self.engine.messages)
                curr_labels = list_differ(curr_labels, input.options)
            if len(curr_labels) == 0:
                curr_labels = old_options

            input.curr_label_selected.append(cluster_dict)
            if len(curr_labels) <= K:
                input.options = curr_labels
                j = 0
                curr_selected = []
                while j < args.select_iter:

                    self.engine.clear()
                    random.shuffle(input.options)

                    topn_label = []

                    # here only let model extract top1 label
                    input.top_k = args.select_num
                    input.text = input.text[-2048:]
                    query = construct_prompt('zero_shot_top_classify', input)[0]
                    resp = self.engine.generate(query)
                    topn_label = extract_answer(
                        resp.replace("cognized", "cognised"), input.options, input.top_k, False
                    )
                    if len(topn_label) != 0:
                        curr_selected.extend(topn_label)
                        select_messages.append(self.engine.messages)
                        break
                    j += 1

                input.curr_label_selected.append(curr_selected)
                break

        input.select_messages = select_messages
        input.topn_label = list(set(curr_selected))

        return input


class topk_remove(LabelSelector):
    def select(self, input):

        topn_label_list = []
        select_messages = []

        old_options = copy.deepcopy(input.options)
        options = copy.deepcopy(input.options)

        i = 0
        while i < args.select_iter:

            if len(options) == 0:
                break

            self.engine.clear()

            # NOTE!!
            input.options = copy.deepcopy(options)
            random.shuffle(input.options)

            topn_label = []

            # 1. remove unrelated label
            input.text = input.text[-2048:]
            query = construct_prompt('label_filter_options_prompt', input)[0]
            resp = self.engine.generate(query)
            removeset, _ = extract_list(resp, input.options)

            if len(removeset) == len(input.options):
                continue

            for r in removeset:
                try:
                    input.options.remove(r)
                except:
                    pass

            if len(input.options) == 1:
                topn_label_list.append(input.options)
                select_messages.append(self.engine.messages)
                i += 1
                continue

            # here only let model extract top1 label
            input.text = input.text[-2048:]
            query = construct_prompt('zero_shot_classify', input)[0]
            resp = self.engine.generate(query)
            topn_label = extract_answer(resp, input.options, 1)

            if len(topn_label) == 0:
                continue

            topn_label_list.append(topn_label)
            select_messages.append(self.engine.messages)
            options.remove(topn_label[0])
            i += 1

        input.topn_label_list = topn_label_list
        input.select_messages = select_messages
        input.options = old_options
        input.topn_label = most_common_elements(topn_label_list, args.select_iter)
        return input


class llm(LabelSelector):
    def select(self, input):

        topn_label = []

        while True:

            self.engine.clear()

            random.shuffle(input.options)

            input.text = input.text[-2048:]
            query = construct_prompt('zero_shot_classify', input)[0]
            resp = self.engine.generate(query)
            topn_label = extract_answer(resp, input.options, 3)

            if len(topn_label) != 0:
                break

        input.topn_label_list = [topn_label]
        input.select_messages = [self.engine.messages]
        input.topn_label = topn_label

        return input


class llm_remove(LabelSelector):
    def select(self, input):

        topn_label = []

        while True:

            self.engine.clear()

            random.shuffle(input.options)

            # 1. remove unrelated label
            input.text = input.text[-2048:]
            query = construct_prompt('label_filter_options_prompt', input)[0]
            resp = self.engine.generate(query)
            removeset, _ = extract_list(resp, input.options)

            if len(removeset) == len(input.options):
                continue

            for r in removeset:
                try:
                    input.options.remove(r)
                except:
                    pass

            input.text = input.text[-2048:]
            query = construct_prompt('zero_shot_classify', input)[0]
            resp = self.engine.generate(query)
            topn_label = extract_answer(resp, input.options, 3)

            if len(topn_label) != 0:
                break

        input.topn_label_list = [topn_label]
        input.select_messages = [self.engine.messages]
        input.topn_label = topn_label

        return input
