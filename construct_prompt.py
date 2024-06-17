import re
import glob
import utils

def read_prompt_template(dir):

    data = {
        k: v 
        for file in glob.glob(f'{dir}/*.yml')
        for k, v in utils.from_yaml(file).items()
    }
    return data

PROMPT_TEMPLATE = read_prompt_template('prompt')

def get_parameters(s):
    match = re.findall("{(.*?)}", s)
    return match

def split_multi_round(s):
    return s.split('||\n')

def construct_prompt(key, input=None, demos=None):

    template = PROMPT_TEMPLATE[key.replace("test","")]

    if isinstance(template, str):
        # zero shot
        if input!=None:
            prompt = template.format_map(input)
        else:
            prompt = template
    elif isinstance(template, dict):
        # few shot
        prompt = ''
        if 'sys' in template:
            prompt += template['sys'].format_map(input)
        if 'demo' in template: 
            if len(get_parameters(template['demo'])) == 0:
                prompt += template['demo']
            elif demos is not None:
                for demo in demos:
                    prompt += template['demo'].format_map(demo)
            else:
                raise NotImplementedError

        prompt += template['query'].format_map(input)

        if 'format' in template: 
            prompt += template['format']

    else:
        raise NotImplementedError

    prompts = split_multi_round(prompt)

    return prompts

if __name__ == "__main__":

    s = "SENTENCE: {text}\nLABEL: {label}\n"
    parameters = get_parameters(s)
    print(f"The parameters needed are: {parameters}")

    inputs = {
        'text1': 'a',
        'text': 'b',
        'label': 'c',
        'label2': 'd'
    }
    
    s = s.format_map(inputs)
    print(s)