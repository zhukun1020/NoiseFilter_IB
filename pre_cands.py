import gc
import glob
import hashlib
import itertools
import json
import os
import re
import time
from os.path import join as pjoin
from tqdm import tqdm
import argparse

from nltk.tokenize import sent_tokenize
from utils import load_dataset, write_dataset, _get_word_ngrams


def pre_prompt(question, ctxs, summ):

    system = '''
You are now an intelligent assessment assistant. Your task is to read the context and then find coherent excerpts that can effectively answer the given question.
After generating the answer, you need to determine whether the generated excerpt contributes to addressing the question. 
'''
    query_prompt = '''
Question: {}
Context: 
{}
'''
    answer_prompt = '''
Question: {}
Excerpt: {}'''
    # question = "Given the ['question', 'context'], predict the answer to the question.\n\nquestion: who got the first nobel prize in physics\ncontext: Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was awarded to physicist Wilhelm R\u00f6ntgen in recognition of the extraordinary services he"

    query = query_prompt.format(question,ctxs)
    if summ is None:
        answer = None
    else:
        answer = answer_prompt.format(question, summ)
    return system, query, answer


def pre_prompt_few(question, ctxs):
    prompt = '''<s>[INST] <<SYS>>
    You are now an intelligent assessment assistant. Your task is to read the context and then find coherent excerpts that can effectively answer the given question.
    After generating the answer, you need to determine whether the generated excerpt contributes to addressing the question. 

    For example:
    Question: Who founded google?
    Context：Google, American search engine company, founded in 1998 by Sergey Brin and Larry Page, that is a subsidiary of the holding company Alphabet Inc. More than 70 percent of worldwide online search requests are handled by Google, placing it at the heart of most Internet users experience. Learn more about Google.
    Excerpt: Google, American search engine company, founded in 1998 by Sergey Brin and Larry Page.
    Contribution：[Y]

    Question: Are both The New Pornographers and Kings of Leon American rock bands?
    Context：The New Pornographers are a Canadian indie rock band, formed in 1997 in Vancouver. Kings of Leon (formed 2000) are an American rock band, consisting of three brothers (Caleb, Nathan and Jared) and their cousin (Matthew), all of whom have the last name Followill, best known for their single 'Sex on Fire'.
    Excerpt: The New Pornographers are a Canadian indie rock band. Kings of Leon (formed 2000) are an American rock band.
    Contribution：[Y]

    Question: What position on the Billboard Top 100 did Alison Moyet's late summer hit achieve?
    Context：\"All Cried Out\" is a song by English singer-songwriter Alison Moyet. It was written by Moyet and producers Jolley & Swain for her debut studio album \"Alf\" (1984). Released as the album's second single in the autumn of 1984, the track peaked within the top ten on both the Irish and the UK Singles Chart, also reaching the top twenty in Switzerland.
    Excerpt: "All Cried Out" is a song by English singer-songwriter Alison Moyet. the track peaked within the top ten on both the Irish and the UK Singles Chart, also reaching the top twenty in Switzerland.
    Contribution：[N]
    Explanation: The excerpt does not provide information about Alison Moyet's late summer hit or its position on the Billboard Top 100. Therefore, it does not contribute to answering the question.


    Question: Who was the first person killed in a car accident?
    Context：The first driver fatality from a collision (not counting Ward’s unfortunate ejection) happened in 1898, when Englishman Henry Lindfield and his son were driving from Brighton to London. Near the end of their trip, Lindfield lost control of the car while going down a hill.
    Excerpt: The first driver fatality from a collision happened in 1898, when Englishman Henry Lindfield and his son were driving from Brighton to London.
    Contribution: [N]
    Explanation: The excerpt provide information about The first driver fatality, which can't ensure he was the first person killed in a car accident. Therefore, it does not contribute to answering the question.

    (END OF EXAMPLE)

    <</SYS>>

    Question: {}
    Context: 
    {}

    [/INST]
    Question: {}
    Excerpt: '''
    # question = "Given the ['question', 'context'], predict the answer to the question.\n\nquestion: who got the first nobel prize in physics\ncontext: Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was awarded to physicist Wilhelm R\u00f6ntgen in recognition of the extraordinary services he"
    prom = prompt.format(question, ctxs, question)
    return prom


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    # print(doc_sent_list)
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            # print(c)
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        s = s.lower()
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    # abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract_sent_list)).split()
    # sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    sents = [_rouge_clean(s).split() for s in doc_sent_list]

    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    # print(evaluated_1grams)
    # print(reference_1grams)
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return max_rouge, selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge
    # print(selected,sorted(selected))
    return max_rouge, sorted(selected)


def exact_selection(source, tgt):
    source = [s.lower() for s in source]
    tgt = [t.lower() for t in tgt]
    selected = []
    for index, s in enumerate(source):
        for ans in tgt:
            if ans in s and (ans!='yes' and ans!="no") and index not in selected:
                selected.append(index)
    return selected


def exact_para_selection(source, tgt):
    source = [s.lower() for s in source]
    tgt = [t.lower() for t in tgt]
    selected = []
    for index, s in enumerate(source):
        for ans in tgt:
            if ans in s and index not in selected:
                selected.append(index)
    return selected



def get_answer(tmp_ans):

    if type(tmp_ans) == str:
        ans_list = [tmp_ans]
    elif type(tmp_ans) == list:
        ans_list = tmp_ans
    elif type(tmp_ans) == dict:
        # print(answer["Aliases"])
        ans_list = tmp_ans["Aliases"]
    else:
        return []

    return ans_list


def add_dataname(name,folder,filename):
    columns = {
        "system": "system",
        "prompt": "input",
        "response": "summary"
    }

    # "/home/huawei/kzhu/LLaMA-Factory-main/data/dataset_info.json"
    info_path = os.path.join(folder,"dataset_info.json")
    if os.path.exists(info_path):
        data_info = load_dataset(info_path)
    else:
        data_info = dict()

    # assert name not in data_info
    data_info[name] = {
        "folder": folder,
        "file_name": filename,
        "columns": columns
    }
    write_dataset(info_path, data_info)


def prepare_summary(json_file, save_file, oracle_mode, summary_size):

    # if (os.path.exists(save_file)):
    #     return
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    add_dataname(os.path.basename(save_file).split(".")[0], os.path.dirname(save_file), os.path.basename(save_file))

    jobs = load_dataset(json_file)
    datasets = []
    pos_example = 0
    neg_example = 0
    count = 0
    for d in tqdm(jobs):
        # print(d)
        question, answers = d['question'], get_answer(d["answer"])

        if oracle_mode == 'greedy':
            tgt = [question] + answers
        else:
            tgt = answers

        # print(tgt)
        ctxs = d['retrieval'][:5]
        ctxs_comb = [ctx['contents'] for ctx in ctxs]
        # print(ctxs_comb)
        source = sent_tokenize("\n".join(ctxs_comb))
        # ctxs_comb = [f"Document {i+1}: {ctx}" for i, ctx in enumerate(ctxs_comb)]
        ctxs_comb = [f"{ctx}" for i, ctx in enumerate(ctxs_comb)]
        # print(source)

        if (oracle_mode == 'greedy_ans'):
            if "supporting_facts" in d:
                sf = d["supporting_facts"]
                tgt = sf + answers
            else:
                tgt = answers
            score, oracle_ids = greedy_selection(source, tgt, summary_size)
            greedy_summary = [source[ids] for ids in oracle_ids]
            # dp_summary = [source[ids] for ids in oracle_ids_com]
            # input = pre_prompt(question, "\n".join(ctxs_comb))

            data_dict = {"summary": greedy_summary,
                         "question": question,
                         "answer": answers,
                         "score": score,
                         "retrieval": ctxs_comb, # "dp_summary": dp_summary,"ctxs": source, "score": score, 'question': question, "answers": answers,
                         }

        elif (oracle_mode == 'greedy'):
            if "supporting_facts" in d:
                sf = d["supporting_facts"]
                tgt = [question] + sf + answers
            else:
                tgt = [question] + answers
            score, oracle_ids = greedy_selection(source, tgt, summary_size)
            greedy_summary = [source[ids] for ids in oracle_ids]

            data_dict = {"summary": greedy_summary,
                         "question": question,
                         "answer": answers,
                         "score": score,
                         "retrieval": ctxs_comb, # "dp_summary": dp_summary,"ctxs": source, "score": score, 'question': question, "answers": answers,
                         }

        elif (oracle_mode == 'exact_para'):
            oracle_ids = exact_para_selection(ctxs_comb, tgt)
            exact_summary = [ctxs_comb[ids] for ids in oracle_ids]
            # dp_summary = [source[ids] for ids in oracle_ids_com]
            # input = pre_prompt(question, "\n".join(ctxs_comb))

            data_dict = {"summary": exact_summary,
                         "question": question,
                         "answer": answers,
                         "retrieval": ctxs_comb, # "dp_summary": dp_summary,"ctxs": source, "score": score, 'question': question, "answers": answers,
                         }

        elif (oracle_mode == 'exact'):
            oracle_ids = exact_selection(source, tgt)
            exact_summary = [source[ids] for ids in oracle_ids]
            # input = pre_prompt(question, "\n".join(ctxs_comb))

            data_dict = {"summary": exact_summary,
                         "question": question,
                         "answer": answers,
                         "retrieval": ctxs_comb, # "dp_summary": dp_summary,"ctxs": source, "score": score, 'question': question, "answers": answers,
                         }


        datasets.append(data_dict)

    write_dataset(save_file, datasets)
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_mode", default='greedy', type=str,
                        help='how to generate oracle summaries, greedy or combination, combination will generate more accurate oracles but take much longer time.')
    parser.add_argument("--source_path", default='./data/source')
    parser.add_argument("--compressed_path", default='./data/compressed/')
    parser.add_argument("--summary_size", type=int, default=5)
    # parser.add_argument("-dataset_name", type=str, required=True)

    args = parser.parse_args()

    prepare_summary(args.json_path, args.save_path, args.oracle_mode, args.summary_size)
