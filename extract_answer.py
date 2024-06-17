import re
import os
import random

PATTERNS = [
    "it is more likely (.*?){}",
    "{}\" would be ",
    "{} would be ",
    "{} is more likely",
    "be categorized as {}",
    "be classified under the {}",
    "be categorized under the {}",
    "fall under the category of {}",
    "{}\" is more likely",
    "be categorized as \"{}",
    "be classified under the \"{}",
    "be categorized under the \"{}",
    "fall under the category of \"{}",
    "{} instead of",
    "{} rather than",
    "{} category rather than",
    "{}\" instead of",
    "{}\" rather than",
    "{}\" category rather than",
    "is \"{}\"",
    "is {}",
    "is \"{}\.\"",
    "is \"{}",
    "be \"{}\"",
    "be \"{}\"\.",
    "be \"{}\.\"",
    "be {}\.",
    "be \"{}",
    "{} is",
    "\"{}\" is",
    "{} is(.*?)likely",
    "\"{}\" is(.*?)likely",
    "{}(.*?)rather than",
    "\"{}\"(.*?)rather than",
    "{}(.*?)instead of",
    "\"{}\"(.*?)instead of",
    "be(.*?)as \"{}\"\.",
    "be(.*?)as \"{}\.\"",
    "be(.*?)as {}\.",
    "be(.*?)as \"{}",
    "is(.*?)\"{}\"\.",
    "is(.*?)\"{}\.\"",
    "is(.*?){}\.",
    "is(.*?)\"{}",
    "be(.*?)\"{}\"\.",
    "be(.*?)\"{}\.\"",
    "be(.*?){}\.",
    "be(.*?)\"{}",
    "\"{}\"",
    "1. {}:",
    "2. {}:",
    ". {}:",
    "{}\.",
    " {} ",
    " {}",
    "{}",
]

def sort_fun(string):
    return len(string),string


def extract_format(text, formatter):

    matches = re.findall(fr'{formatter}:(.*?)(?:\n|\*|`|$)', text, re.DOTALL | re.IGNORECASE)
    if len(matches):
        matches = matches[-1].strip()
        if '[' in matches:
            matches = re.findall(r"\[(.*?)\]", matches, re.DOTALL )[0]
        matches = matches.replace(' ','').replace('"','').split(',')
        return matches
    
    return None

def extract_list(text, options=None):

    if os.getenv('DRYRUN'):
        if options:
            return [random.choice(options)], []
        return [random.choice(['DRYRUN1', 'DRYRUN2', 'DRYRUN3'])], []

    remove_pattern=re.escape(".'?\"\n")
    values = re.split(r',\s*', re.sub(f'[{remove_pattern}]', '', text.strip()))
    values = list(set(values))
    if options is not None:
        seen = [ v for v in values if v in options ]
        unseen = [ v for v in values if v not in options ]
        return seen, unseen
    
    return [], values

def extract_zeroshot_answer(text, options, top):

    match_list = []
    for pattern_template in PATTERNS:
        for name in options:
            if top == 0:
                return match_list
            pattern = pattern_template.format(name)
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if name not in match_list:
                    match_list.append(name)
                    top -= 1
                continue
            pattern = pattern_template.format(name.replace('_', ' '))
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if name not in match_list:
                    match_list.append(name)
                    top -= 1
                continue

    return match_list


def extract_answer(text, options, top=1, filter=True):
    if filter:
        if os.getenv('DRYRUN'):
            return options[:top]

        if 'LABEL: ' in text.strip():
            # fewshot
            ans = text.split("LABEL: a\"")[-1].split("LABEL: ")[-1].split(':')[0].split('\n')[0].split('.')[0].split('"')[0]
            ans = ans.replace('\\', '')
            for o in options:
                if ans == o:
                    return [o]
            for o in options:
                if o in ans:
                    return [o]
            if len(ans) and ans[0]!='a':
                return [ans]
        
    options = sorted(options, key=sort_fun, reverse=True)
    
    return extract_zeroshot_answer(text, options, top)


if __name__ == "__main__":
    
    print(
        extract_zeroshot_answer(
            "Based on the similarity of expressing concern or confusion about the status of the transfer and seeking clarification, and the difference of the transfer not going through despite multiple attempts, the term that is more likely to represent the label for the sentence \"I've tried numerous times to submit a transfer of funds. Why isn't it going through?\" would be \"pending_transfer\". This is because the sentence indicates that the transfer is not yet completed or approved despite repeated attempts, suggesting it is still in a pending state.",
            ["pending_transfer", "declined_transfer"],
            1
        )
    )