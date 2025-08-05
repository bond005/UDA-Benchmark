from argparse import ArgumentParser
import gc
import json
import logging
import os
import random
import sys
import warnings

import numpy as np
import torch
from tqdm import tqdm

root_directory = os.path.join('..', '..', os.path.dirname(__file__))
sys.path.append(root_directory)
os.chdir(root_directory)


from uda.utils import retrieve as rt
from uda.utils import preprocess as pre
from uda.utils import llm
from uda.utils import inference
import uda.utils.access_config

rag_benchmark_logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42


def call_finance_eval_custom(dataset, data):
    if dataset == 'tat':
        from uda.eval.utils.tat_eval import TaTQAEmAndF1 as EvalClass

        em_and_f1 = EvalClass()
        for res in data:
            em_and_f1(res)
        global_em, global_f1, _, _ = em_and_f1.get_overall_metric()
        rag_benchmark_logger.info('Numerical F1 score: {0:.2f}'.format(global_f1 * 100))
    elif dataset == 'fin':
        from uda.eval.utils.fin_eval import FinQAEm as EvalClass

        em = EvalClass()
        for res in data:
            em(res)
        global_em = em.get_overall_metric()
        rag_benchmark_logger.info('Exact-match accuracy: {0:.2f}'.format(global_em * 100))


def call_f1_eval_custom(dataset, answers, preds):
    from uda.eval.utils.paper_eval import paper_evaluate
    from uda.eval.utils.feta_eval import feta_evaluate
    from uda.eval.utils.nq_eval import nq_evaluate

    res = None
    if dataset == 'paper':
        res = paper_evaluate(answers, preds)
    elif dataset == 'feta':
        res = feta_evaluate(answers, preds)
    elif dataset == 'nq':
        res = nq_evaluate(answers, preds)

    if res is not None:
        rag_benchmark_logger.info(f'{res}')


def eval_main_custom(dataset_name, data_list):
    if dataset_name in ['paper_tab', 'paper_text']:
        dataset = 'paper'
    else:
        dataset = dataset_name

    if dataset in ['tat', 'fin']:
        call_finance_eval_custom(dataset, data_list)
    elif dataset in ['paper', 'feta', 'nq']:
        answers = {}
        preds = {}
        for item in data_list:
            gts = item['answers']
            if len(gts) == 0:
                continue
            q_uid = item['q_uid']
            if item["response"] is None:
                continue
            pred = item['response'].split('The answer is: ')[-1]
            answers[q_uid] = gts
            preds[q_uid] = {'answer': pred}
        call_f1_eval_custom(dataset, answers, preds)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.random.manual_seed(RANDOM_SEED)
    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        rag_benchmark_logger.error(err_msg)
        raise RuntimeError(err_msg)
    torch.cuda.random.manual_seed_all(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='tested_llm', type=str, required=True,
                        help='The tested large language model.')
    parser.add_argument('-d', '--dataset', dest='used_dataset', type=str, required=False, default=None,
                        choices=['fin', 'feta', 'tat', 'paper_text', 'nq', 'paper_tab'],
                        help='The used dataset (if it is not specified, then all datasets are used).')
    parser.add_argument('--emb', dest='embedder', type=str, required=False, default='colbert',
                        choices=['bm25', 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'openai', 'colbert'],
                        help='The tested embedder.')
    parser.add_argument('--hf_token', dest='hf_token', type=str, required=False, default='',
                        help='The Huggingface token to access the HuggingFace models')
    args = parser.parse_args()

    if len(args.hf_token.strip()) == 0:
        uda.utils.access_config.HF_TOKEN = False
    else:
        uda.utils.access_config.HF_TOKEN = args.hf_token.strip()

    if args.used_dataset is None:
        dataset_name_list = ['fin', 'feta', 'tat', 'paper_text', 'nq', 'paper_tab']
    else:
        dataset_name_list = [args.used_dataset]

    res_dir = os.path.join(root_directory, 'experiment', 'e2e', 'res')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    rag_benchmark_logger.info(f'The directory with results is {res_dir}.')

    llm_name = args.tested_llm
    llm_base_name = os.path.basename(llm_name)

    llm_service = inference.LLM(llm_name)
    llm_service.init_llm()
    rag_benchmark_logger.info(f'LLM service {llm_name} is initialized.')

    for ds_name in dataset_name_list:
        rag_benchmark_logger.info(f'=== Start {ds_name} on {llm_base_name} with {args.embedder} embedder ===')
        res_file_name = os.path.join(res_dir, f'{ds_name}_{llm_base_name}_{args.embedder}.jsonl')
        if os.path.isfile(res_file_name):
            os.remove(res_file_name)

        # Load the benchmark data
        bench_json_file = pre.meta_data[ds_name]['bench_json_file']
        with open(bench_json_file, "r") as f:
            bench_data = json.load(f)

        # Run experiments on the benchmark documents
        doc_list = list(bench_data.keys())
        for doc in tqdm(doc_list):
            pdf_path = pre.get_pdf_path(ds_name, doc)
            if pdf_path is None:
                continue
            # Prepare the index for the document
            collection_name = f'{ds_name}_vector_db'
            collection = rt.prepare_collection(pdf_path, collection_name, args.embedder)
            for qa_item in bench_data[doc]:
                question = qa_item['question']
                # Retrieve the contexts
                contexts = rt.get_contexts(collection, question, args.embedder)
                context_text = '\n'.join(contexts)
                # Create the prompt
                llm_message = llm.make_prompt(question, context_text, ds_name, llm_base_name)
                # Generate the answer
                response = llm_service.infer(llm_message)
                # log the results
                res_dict = {'model': llm_name, 'question': question, 'response': response, 'doc': doc,
                            'q_uid': qa_item['q_uid'], 'answers': qa_item['answers']}
                with open(res_file_name, 'a') as f:
                    f.write(json.dumps(res_dict) + '\n')
                del res_dict
            rt.reset_collection(collection_name, args.embedder)

        with open(res_file_name, 'r') as f:
            data = [json.loads(line) for line in f]
        eval_main_custom(ds_name, data)
        del data

        rag_benchmark_logger.info(f'=== Finish {ds_name} on {llm_base_name} with {args.embedder} embedder ===\n')

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    rag_benchmark_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    rag_benchmark_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('uda_benchmark.log')
    file_handler.setFormatter(formatter)
    rag_benchmark_logger.addHandler(file_handler)
    main()
