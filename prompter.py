import os
import json
import copy
import random
import logging
import numpy as np
import openai_client
from utils import args
from extract_answer import extract_answer
from construct_prompt import construct_prompt
from sklearn.metrics.pairwise import cosine_similarity
import itertools

DEFAULT_MESSAGE = "Who are you?"

FAIL_PATTERN = ['None of', 'Neither of', 'Please provide more details']

LOG = logging.getLogger('ray')

def calculate_similarity(vector1, vector2):
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return similarity

def find_least_similar_pair(label_vector, curr_label):
    least_similarity = float('inf')
    least_similar_pair = None

    for pair in itertools.combinations(curr_label, 2):
        label1, label2 = pair
        vector1 = label_vector[label1]
        vector2 = label_vector[label2]
        similarity = calculate_similarity(vector1, vector2)

        if similarity < least_similarity:
            least_similarity = similarity
            least_similar_pair = (label1, label2)

    return least_similar_pair   

class Prompter:
    def __init__(
        self,
        engine,
    ):
        self.engine = engine

    def preprocess(self, input):
        return input

    def postprocess(self, input):

        input.valid = self.valid(input)
        self.engine.clear()

        return input

    def valid(self, input):
        return True

    def run(self, input):

        input.resp = self.engine.generate(DEFAULT_MESSAGE)
        return input

    def __call__(self, input):

        for _ in range(2):
            try:
                self.engine.clear()
                input = self.preprocess(input)
                if input.skip == False:
                    input = self.run(input)
                    input = self.postprocess(input)
                    self.engine.clear()
                break
            except openai_client.OpenaiAPIRestartError:
                LOG.error('has error. restart.')
                pass

        return input


class TextClassifier(Prompter):
    def __init__(self, prompt_type, engine, fewshot_loader, topn_label_loader, run):
        self.prompt_type = prompt_type
        self.engine = engine
        self.fewshot_loader = fewshot_loader
        self.topn_label_loader = topn_label_loader
        self.run = getattr(self, run)

    def preprocess(self, input):

        # prepare query samples / demos for specific input.
        # self.labelselector(input)
        self.topn_label_loader(input)
        input.skip = False
        if args.selector != 'raw' and input.label not in input.topn_label:
            input.valid = True
            input.skip = True
            input.answer = input.topn_label[0]

        return input

    def postprocess(self, input):

        input.valid = self.valid(input)
        self.engine.clear()

        return input

    def valid(self, input):
        if not hasattr(input, 'answer'):
            return False

        ans = len(input.answer) > 0
        if 'classify' in self.prompt_type:
            if hasattr(input, "topn_label"):
                ans = ans and input.answer in input.topn_label
        return ans

    def run_plain(self, input):

        demos = self.fewshot_loader(input)
        queries = construct_prompt(self.prompt_type, input, demos)
        for query in queries:
            resp = self.engine.generate(query)
        # not classify , don't need extract_answer
        input.answer = resp
        input.messages = self.engine.messages

        return input
    
    
    def run_deduced(self, input):
        from dataloader import fewshot_cot_classify
        final_answer = []
        remove_labels = []
        compare_messages = []

        # with open(f'datasets/{args.data}/label_vectors.pkl', 'rb') as file:
        #     self.label_to_vector = pickle.load(file)

        topn_label = copy.deepcopy(input.topn_label)

        for i in range(5):

            if len(topn_label) < 2:
                break

            self.engine.clear()
            # (input.label1,input.label2) = find_least_similar_pair(self.label_to_vector,topn_label)
            input.label1 = topn_label[0]
            input.label2 = topn_label[1]
            demos = self.fewshot_loader(input)
            queries = construct_prompt(self.prompt_type, input, demos)
            for query in queries:
                resp = self.engine.generate(query)
            final_answer = extract_answer(resp, topn_label[:2])

            if len(final_answer) == 0:

                for pattern in FAIL_PATTERN:
                    if pattern.lower() in resp.lower():
                        raise openai_client.OpenaiAPIRestartError

                final_answer = extract_answer(resp, topn_label[:2], filter=False)
                if len(final_answer) == 0 or final_answer[0] not in topn_label[:2]:
                    final_answer = topn_label[:1]

            final_answer = final_answer[0]

            if final_answer == input.label1:
                remove_label = input.label2
            else:
                remove_label = input.label1

            topn_label.remove(remove_label)
            remove_labels.append(remove_label)
            compare_messages.append(self.engine.messages)

        input.answer = topn_label[0]
        input.remove_labels = remove_labels
        input.compare_messages = compare_messages

        del input.label1
        del input.label2

        return input

    def run_options(self, input):

        total_options = copy.deepcopy(input.options)
        if args.option_num != -1:
            input.options.remove(input.label)
            input.options = random.choices(input.options, k=args.option_num - 1)
            input.options.append(input.label)
            if not os.getenv('shuff_id'):
                random.shuffle(input.options)

        if os.getenv('shuff_id'):
            if os.getenv('shuff_id') == '-1':
                random.shuffle(input.options)
            elif os.getenv('shuff_id') == '-2':
                input.text = 'N/A'
            else:
                input.options.remove(input.label)
                input.options.insert(int(os.getenv('shuff_id')), input.label)

        if args.similar_rate != -1:
            split_index = int(args.option_num * args.similar_rate)
            input.options = input.similar_labels[:split_index] + input.diverse_labels[split_index:]
            input.options.append(input.label)

        demos = self.fewshot_loader(input)

        final_answer = []

        for i in range(2):

            if len(final_answer):
                break

            self.engine.clear()

            random.shuffle(input.options)
            queries = construct_prompt(self.prompt_type, input, demos)
            for query in queries:
                resp = self.engine.generate(query)

            final_answer = extract_answer(
                resp, input.topn_label if hasattr(input, "topn_label") else input.options
            )

            if len(final_answer) == 0:
                # TODO: may both label is incorrect
                queries = construct_prompt('repeat_prompt', input, demos)
                for query in queries:
                    resp = self.engine.generate(query)
                final_answer = extract_answer(
                    resp, input.topn_label if hasattr(input, "topn_label") else input.options
                )
                for pattern in FAIL_PATTERN:
                    if pattern.lower() in resp.lower():
                        final_answer = [input.options[0]]
                        # raise openai_client.OpenaiAPIRestartError

        if i == 2 or len(final_answer) == 0:
            input.valid = True
            input.skip = True
            input.answer = input.options[0]
            del input.label1
            del input.label2

            return input

        input.answer = final_answer[0]
        input.compare_messages = [self.engine.messages]
        input.options = total_options
        return input

    def run_verification(self, input):

        pass

    def demo_gen(self, input):
        import pickle
        with open(f'datasets/{args.data}/label_vectors.pkl', 'rb') as file:
            self.label_to_vector = pickle.load(file)

        if args.sample == 'random':
            self.engine.clear()
            prompt_template_name = "random_demonstraion_generation"

            output_template = construct_prompt("demonstraion_template", input)[0]
            options = f"'{input.label1}' and '{input.label2}'"

            queries_for_demonstration = construct_prompt(
                prompt_template_name, {
                    'options': options,
                    'template': output_template
                }
            )[0]

            failure_time = 0
            while (failure_time < 2):
                resp = self.engine.generate(queries_for_demonstration)
                resp = resp.replace('"},\n{"', '", "').replace('"},\n\n{"', '", "')
                try:
                    demos = json.loads(resp)
                    break
                except:
                    failure_time += 1
            if failure_time >= 2:
                raise openai_client.OpenaiAPIDataItemError
        else:
            if args.sample == 'boundary':
                prompt_template_name = 'similary_demonstraion_generation'
            elif args.sample == 'center':
                prompt_template_name = 'differ_demonstraion_generation'
            demos = {}
            input.options = list(self.label_to_vector.keys())
            for curr_label in [input.label1, input.label2]:
                given_vector = next(
                    vector for label, vector in self.label_to_vector.items() if label == curr_label
                )
                similarities = cosine_similarity([given_vector], list(self.label_to_vector.values()))
                most_similar_index = similarities.argsort()[0][-2]
                similar_label = input.options[most_similar_index]
                options = f"'{curr_label}' and '{similar_label}'"

                output_template = construct_prompt(
                    "single_demonstraion_template", {'label1': curr_label}
                )[0]
                queries_for_demonstration = construct_prompt(
                    prompt_template_name, {
                        'label1': curr_label,
                        'label2': similar_label,
                        'options': options,
                        'template': output_template
                    }
                )[0]
                failure_time = 0
                while (failure_time < 4):
                    resp = self.engine.generate(queries_for_demonstration)
                    resp = resp.replace('"},\n{"','", "').replace('"},\n\n{"','", "').replace("cognized","cognised").strip(".")
                    try:
                        new_demo = json.loads(resp)
                        for k, v in new_demo.items():
                            new_demo[curr_label] = v
                        demos.update(json.loads(resp))
                        break
                    except:
                        queries_for_demonstration = queries_for_demonstration.replace("\n", ". ")
                        failure_time += 1
                if failure_time >= 2:
                    raise openai_client.OpenaiAPIDataItemError
                self.engine.clear()

        input.text1 = demos[input.label1]
        input.text2 = demos[input.label2]
        input.answer = ['finish']
        return input
