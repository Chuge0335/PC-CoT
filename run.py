import os
import sys
import utils
import logging
import openai_client
import dataloader
import prompter
import label_select

LOG = logging.getLogger(os.path.basename(__file__))

# call bash/Python shell scripts -> wait the disk result files
# call openai_client.run

def pipeline(args):
    # run label reducer
    if args.algorithm is None and args.selector is not None:
        task = utils.SimpleNamespace(
            name = args.selector + '_reducer',
            prompt = f'topn_label',
            dataloader = dataloader.load,
            datasaver = dataloader.save,
            prompter = getattr(label_select, args.selector)(
                engine = openai_client.OpenAI(args.model),
            ), 
            runner = openai_client.run ,
            outs = f'{args.outs}/{args.data}/{args.selector}',
        )
    # run label compairation
    else:
        task = utils.SimpleNamespace(
            name = args.prompt.split('_')[-1],
            prompt = args.prompt,
            dataloader = dataloader.load,
            datasaver = dataloader.save,
            prompter = prompter.TextClassifier(
                prompt_type=args.prompt,
                engine = openai_client.OpenAI(args.model),
                fewshot_loader = getattr(dataloader, args.prompt, dataloader.fewshot_raw)(),
                topn_label_loader=dataloader.topn_label_loader(args.selector),
                run=args.algorithm, 
            ), 
            runner = openai_client.run ,
            outs = f'{args.outs}/{args.data}/{args.selector}',
        )
    
    context = []
    task_data = task.dataloader(task, context)
    # task.prompter.__call__
    task_result, unfinished = task.runner(task_data, task.prompter)
    if len(task_result):
        task.datasaver(task, task_result, context)
    if len(unfinished):
        LOG.error('there is unfinished data, need mannally handled.')
        task.prompt += '-unfinished'
        task.datasaver(task, unfinished, None)

## starter
def main(
    *,
    # for debug
    dry_run: bool = False, 
    debug: bool = False, # only use one apikey
    # for io 
    outs: str = 'outs',
    data: str = 'agnews',
    subset: str = None,
    max_len: int = 512,
    # for prompter
    prompt: str = 'maccot',
    shot: int = 0,
    sample: str = 'boundary',
    # for engine
    model: str = 'gpt-3.5-turbo',
    unfinished: bool = None,    
    # 1) for label comparison
    algorithm: str = None,
    # run_deduced
    selector: str = 'raw',
    select_iter: int = None,
    select_num: int = 5,
    option_num: int = -1,
    select_similary: bool =False,
    similar_rate: float = -1,
):
    import inspect
    frame = inspect.currentframe()
    keys, _, _, values = inspect.getargvalues(frame)
    values = {k: values[k] for k in keys}
    # utils.args = utils.SimpleNamespace(**values)
    utils.args.update(**values)
    args = utils.args

    # hack
    def check_arg_and_env(arg, env_name, postfix):
        if os.getenv(env_name, False):
            setattr(args, arg, True)
        elif getattr(args, arg):
            os.environ[env_name] = 'True'
        if getattr(args, arg) and len(postfix):
            args.outs = args.outs + '-' + postfix

    if os.getenv('OUTS'):
        args.outs = os.getenv('OUTS')
    check_arg_and_env('dry_run', 'DRYRUN', 'dryrun')
    check_arg_and_env('debug', 'DEBUG', 'debug')

    if args.debug and args.dry_run:
        raise Exception("You cannot set both debug and dry_run!")

    os.makedirs(f'{args.outs}/{args.data}', exist_ok=True)
    LOG.info(args)

    # support for pipeline
    pipeline(args)


if __name__ == '__main__':
    import defopt
    try:
        defopt.run(main)
    except:
        import sys, pdb, bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit or type == SystemExit:
            exit()
        print(type, value)
        pdb.post_mortem(tb)
