import os 
import json
from utils import load_dataset,write_dataset
from Info_Bottle import calculate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from check_base import extract_pred, has_answer, calc_unigram_f1, calc_exact_match, get_answer
import scipy


def sort_newindex(sort_list):
    # list_zip = [(index + 1, num) for index, num in enumerate(sort_list)]
    result1 = sorted(enumerate(sort_list), key=lambda x: x[1]) # (index_old, num)
    # print(result1)
    result2 = sorted(enumerate(result1), key=lambda x: x[1][0]) # (index_new,(index_old, num))
    # print(result2)
    return [r[0] for r in result2]


def pre_prompt_mask(question, ctxs, summ, contri):

    system = '''
You are now an intelligent assessment assistant. Your task is to read the context and then find coherent excerpts that can effectively answer the given question.
After generating the answer, you need to determine whether the generated excerpt contributes to addressing the question. 
'''
    query_prompt = '''
Question: {}
Context: 
{}
[/INST]
'''
    answer_prompt = '''Question: {}
Excerpt: {}
 Contribution: [{}]'''
    # question = "Given the ['question', 'context'], predict the answer to the question.\n\nquestion: who got the first nobel prize in physics\ncontext: Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was awarded to physicist Wilhelm R\u00f6ntgen in recognition of the extraordinary services he"

    query_prompt2 = '''
Question: {}
Context: 
{}
[/INST]
 Question: {}
Excerpt: {}
'''
    answer_prompt2 = '''Contribution: [{}]'''
    if "y" in contri.lower():
        query = query_prompt.format(question,ctxs)
        answer = answer_prompt.format(question, summ, contri)
    else:
        query = query_prompt2.format(question,ctxs, question, summ)
        answer = answer_prompt2.format(contri)

    return system, query, answer


def add_dataname(name,folder,filename):
    columns = {
        "system": "system",
        "prompt": "input",
        "response": "output"
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


def with_ib_filter(beta,dataset_name):
# /home/kzhu/project/noise_filtering/src/data/compressed_top5/combine/nq_train_loss-sam2000.jsonl
    
    loss_path = f"/home/kzhu/project/noise_filtering/src/data/compressed_top5/combine/{dataset_name}_train_loss_all.jsonl"
    save_path = f"/home/kzhu/project/noise_filtering/src/data/compressed_top5/ib_select/{dataset_name}_train_loss2_10_mask_addno50.jsonl"
    loss_data = load_dataset(loss_path)
    # origin_path = f"/home/kzhu/project/noise_filtering/src/data/source_top5/{dataset_name}_train_top100.jsonl"
    origin_path = f"/home/kzhu/project/noise_filtering/src/data/compressed_top5/greedy_ans/{dataset_name}_train.jsonl"
    origin_data = load_dataset(origin_path)
    # loss_data = load_dataset(f"/home/kzhu/project/noise_filtering/src/data/compressed_top5/combine/{dataset_name}_dev_loss.jsonl")
    # loss_data = loss_data[:10]

    spss_data2 = []
    count = 0
    len_sum = 0
    question_list = set()
    for index, data in enumerate(loss_data):
        question_list.add(data["question"])
        # min_ib = 10000
        # tmp_compress_key = ""

        ib_list = []
        for key,value in data["compressed"].items():
            # if key != "greedy_ans":
            #     continue
            # print(value)
            if key == "exact_para":
                continue

            loss2 = float(data["compressed"][key]["loss2"])
            summary_len = data["compressed"][key]["summary_len"]
            ib = 1*loss1-beta*loss2

            # ib = -10*loss2
            # summ_len = len(" ".join(value["summary"]).split())
            # print(value)
            ib_list.append((ib,key))
            # if ib < min_ib:
            #     min_ib = ib
            #     tmp_compress_key = key
        result1 = sorted(ib_list, key=lambda x: x[0]) # (index_old, num)
        # print(result1)
        tmp_compress_key = result1[0][1]

        if data["compressed"][tmp_compress_key]["summary_len"] != 0:
            system, input_, output = pre_prompt_mask(data['question'], "\n".join(data["retrieval"]), " ".join(data["compressed"][tmp_compress_key]["summary"]), "Yes")
            count += 1  
            len_sum += data["compressed"][tmp_compress_key]["summary_len"]
        else:
            # continue
            system, input_, output = pre_prompt_mask(data['question'], "\n".join(data["retrieval"]), " ".join(data["compressed"]["greedy_ans"]["summary"]), "No") 

        new_data = dict() #data
        # new_data["ib_best"] = tmp_compress_key
        new_data["system"] = system
        new_data["input"] = input_
        new_data["output"] = output

        spss_data2.append(new_data)   
    print(count,len(spss_data2),count/len(spss_data2))
    count_no = count - (len(spss_data2) - count)
    count = 0
    
    spss_data_no = []
    for o_data in origin_data:
        if o_data["question"] in question_list:
            continue
        count += 1
        if len(o_data["summary"]) > 0 and count_no > 0:
            system, input_, output = pre_prompt_mask(o_data['question'], "\n".join(o_data["retrieval"]), " ".join(o_data["summary"]), "No") 
            new_data = dict()
            new_data["system"] = system
            new_data["input"] = input_
            new_data["output"] = output
            spss_data_no.append(new_data) 
            count_no -= 1

    print(len_sum/count)
    print(len(spss_data_no))
    print(count,len(origin_data), len(question_list), len(spss_data2))
    # assert len(question_list)==len(spss_data2)
    # assert (count+len(question_list)) == len(origin_data)
    spss_data2.extend(spss_data_no)
    print(len(spss_data2))
    # write_dataset(save_path,spss_data2)
    # add_dataname(os.path.basename(save_path).split(".")[0], os.path.dirname(save_path), os.path.basename(save_path))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/kzhu/project/noise_filtering/src/data/compressed_top5/combine/")
    parser.add_argument("--data_name", default='nq_dev.jsonl')
    parser.add_argument("--save_name", default='nq_dev_loss.jsonl')
    parser.add_argument("--model_path", default="/home/kzhu/project/noise_filtering/models/llama2_13b_chat_hf")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_example", type=int, default=None)
    # parser.add_argument("-dataset_name", type=str, required=True)

    args = parser.parse_args()


    alpha = 10
    with_ib_filter(alpha,"nq") 


