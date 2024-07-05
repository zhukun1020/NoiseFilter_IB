import os 
import json
from utils import load_dataset,write_dataset
from Info_Bottle import calculate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def pre_prompt_source(ques,source,summary):
    system1 = '''
You are now an intelligent assessment assistant. 
Your task is to read the summary and the question then guess the original text of the summary.
'''

    query_prompt = '''Question: {}\n'''
    summary_prompt = '''Question: {}
Summary: {}'''
    output_prompt = '''Context: {}'''

    total_prompt = "[INST] <<SYS>>\n{}\n<</SYS>>\n{}\n[/INST]{}"
    input_p = total_prompt.format(system1, summary_prompt.format(ques,summary), query_prompt.format(ques))
    out_p = output_prompt.format(source)

    return input_p,out_p


def pre_prompt_answer(ques,summary,answer):

    system2 = '''
You are now an intelligent assessment assistant. 
Your task is to predict the answer to the question based on the given context. 
If you don't know the answer to a question, please don't share false information.
Answer the question as accurately as possible and put the answer in the form [answer].
'''
    summary_prompt = '''Question: {}
Summary: {}'''
    answer_prompt = '''Question: {}\n'''
    output_prompt = '''Answer: {}'''

    total_prompt = "[INST] <<SYS>>\n{}\n<</SYS>>\n{}\n[/INST]{}"
    input_p = total_prompt.format(system2, summary_prompt.format(ques,summary), answer_prompt.format(ques))
    out_p = output_prompt.format(answer)

    return input_p,out_p


def resume_iboutput():
    oridata = load_dataset("/home/kzhu/project/noise_filtering/src/data/compressed_top5/combine/hotpot_dev.jsonl")[:3000]
    loss_data = load_dataset("/home/kzhu/project/noise_filtering/src/data/compressed_top5/combine/hotpot_dev_loss.jsonl")

    new_dataset = []
    for index, data in enumerate(loss_data):
        compressed_data = dict(data)
        answer = data["answer"]
        summary_info_new = {}
        for key,value in oridata[index]["compressed"].items():

            hasanswer_bool = has_answer(extract_pred("\n".join(value)), answer)

            summary_dic_new = {
                "has_answer": hasanswer_bool,
                "summary": value,
                "summary_len": len("\n".join(value).split()),
                "loss1": data["compressed"][key]["loss1"],
                "loss2": data["compressed"][key]["loss2"]
            }
            summary_info_new[key] = summary_dic_new
        compressed_data["compressed"] = summary_info_new
        new_dataset.append(compressed_data)
    
    write_dataset("/home/kzhu/project/noise_filtering/src/data/compressed_top5/combine/hotpot_dev_loss-3.jsonl",new_dataset)


def sort_newindex(sort_list):
    # list_zip = [(index + 1, num) for index, num in enumerate(sort_list)]
    result1 = sorted(enumerate(sort_list), key=lambda x: x[1]) # (index_old, num)
    # print(result1)
    result2 = sorted(enumerate(result1), key=lambda x: x[1][0]) # (index_new,(index_old, num))
    # print(result2)
    return [r[0] for r in result2]


def test():

    filepath = os.path.join(args.data_path,args.data_name)
    if args.max_example is not None:
        dataset = load_dataset(filepath)[:args.max_example]
    else:
        dataset = load_dataset(filepath)
    model_path = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", low_cpu_mem_usage=True)

    # input_content = []
    # output_content = []
    for data in tqdm(dataset):
        question = data["question"]
        answer = data["answer"]
        retrieval = data["retrieval"]
        summary_dic = data["compressed"]
        
        # tmp_l1i, tmp_l1o, tmp_l2i, tmp_l2o = []
        input_content = []
        output_content = []
        for key,value in summary_dic.items():
            print(key,value)
            a,b = pre_prompt_source(question,"\n".join(retrieval),"\n".join(value))
            input_content.append(a)
            output_content.append(b)
            print(a,b)
            a, b = pre_prompt_answer(question,"\n".join(value),answer)
            input_content.append(a)
            output_content.append(b)
            print(a,b)
        loss = calculate(model, tokenizer, input_content, output_content).tolist()
        print(loss)


def pre_dataset(compressed_path, source_path, savedir, compress_type, dataname):

    all_dataset = {}
    for cmp_type in compress_type:
        filepath = os.path.join(compressed_path,cmp_type,f"{dataname}")
        dataset = load_dataset(filepath)
        all_dataset[cmp_type] = dataset

    filepath = os.path.join(source_path,f"{dataname}")
    dataset = load_dataset(filepath)
    all_dataset["retrieval"] = dataset
    for key in all_dataset.keys():
        print(key, len(all_dataset[key]))

    comb_dataset = []
    for index,data1 in enumerate(all_dataset[compress_type[0]]):            

        tmp_item = {
            "question": data1["question"],
            "answer": data1["answer"],
            "retrieval": [p["contents"] for p in dataset[index]["retrieval"][:5]]
        }
        compressed_list = {}
        flag = True
        for i in range(len(compress_type)):
            summ = all_dataset[compress_type[i]][index]["summary"]
            compressed_list[compress_type[i]] = summ
            if len(summ) == 0:
                flag = False
        tmp_item["compressed"] = compressed_list
        if flag:
            comb_dataset.append(tmp_item)
    print(len(dataset), len(comb_dataset))
    
    write_dataset(os.path.join(savedir,f"{dataname}"),comb_dataset)



def main():

    compress_type = []
    for ctype in os.listdir(args.compressed_path):
        compress_type.append(ctype)
    pre_dataset(args.compressed_path, args.source_path, args.save_path, compress_type, args.data_name)

    filepath = os.path.join(args.save_path,args.data_name)
    if args.max_example is not None:
        if args.max_example2 is not None:
            dataset = load_dataset(filepath)[args.max_example:args.max_example2]
        else:
            dataset = load_dataset(filepath)[:args.max_example]
    else:
        dataset = load_dataset(filepath)

    model_path = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", low_cpu_mem_usage=True)

    input_content = []
    output_content = []
    for data in tqdm(dataset):
        question = data["question"]
        answer = data["answer"]
        retrieval = data["retrieval"]
        summary_dic = data["compressed"]
        a, b = pre_prompt_answer(question," ",answer)
        input_content.append(a)
        output_content.append(b)
        
        # tmp_l1i, tmp_l1o, tmp_l2i, tmp_l2o = []
        # input_content = []
        # output_content = []
        for key,value in summary_dic.items():
            a,b = pre_prompt_source(question,"\n".join(retrieval),"\n".join(value))
            input_content.append(a)
            output_content.append(b)
            a, b = pre_prompt_answer(question,"\n".join(value),answer)
            input_content.append(a)
            output_content.append(b)
    
    print(len(dataset),len(input_content),len(output_content))
    batch_size = args.batch_size
    if len(input_content)%batch_size == 0:
        num = int(len(input_content)/batch_size)
    else:
        num = int(len(input_content)/batch_size) + 1
    print(f"Batch_size {batch_size}, Num {num}")
    loss_all = []
    try:
        for i in tqdm(range(num)):
            loss = calculate(model, tokenizer, input_content[batch_size*i:batch_size*(i+1)], output_content[batch_size*i:batch_size*(i+1)]).tolist()
            loss_all.extend(loss)
    except:
        print("ERROR")
        
    print(len(loss_all), loss_all)
    index = 0
    final_dataset = []
    for data in tqdm(dataset):
        summary_dic = data["compressed"]
        compressed_count = 2*(len(summary_dic)) + 1
        tmp_data = dict(data)
        summary_info_new = {}

        summary_dic_new = {
                "summary": [],
                "summary_len": len("\n".join(value).split()),
                "loss1": "0",
                "loss2": "%.5f"%loss_all[index*compressed_count]
            }
        summary_info_new["no_retrieval"] = summary_dic_new
        count = 1

        for key,value in summary_dic.items():
            summary_dic_new = {
                "summary": value,
                "summary_len": len("\n".join(value).split()),
                "loss1": "%.5f"%loss_all[index*compressed_count+count],
                "loss2": "%.5f"%loss_all[index*compressed_count+count+1]
            }
            count += 2
            summary_info_new[key] = summary_dic_new
        tmp_data["compressed"] = summary_info_new
        final_dataset.append(tmp_data)
        index+=1

    savepath = os.path.join(args.save_path,args.save_name)
    write_dataset(savepath, final_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", default='./data/source')
    parser.add_argument("--compressed_path", default="./data/compressed_top5/")
    parser.add_argument("--save_path", default="./data/combine/")
    parser.add_argument("--data_name", default='nq_dev.jsonl')
    parser.add_argument("--save_name", default='nq_dev_loss.jsonl')
    # parser.add_argument("--compress_type", default='llama_summ')
    parser.add_argument("--model_path", default="/home/kzhu/project/noise_filtering/models/llama2_13b_chat_hf")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_example", type=int, default=None)
    parser.add_argument("--max_example2", type=int, default=None)
    # parser.add_argument("--top", type=int, default=5)


    args = parser.parse_args()

    main()



