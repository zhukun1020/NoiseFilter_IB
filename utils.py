import json
import os
import random


def clean_output(text: str) -> str:
    """Extract out the answer text portion.

    prediction: e.g., "The above context is (-23.0) helpful. answer: Violet Alva"
    returns: e.g., "Violent Alva"
    """
    text = text.replace("Answer", "answer")
    if "answer:" in text:
        text = text.split("answer:")[-1].strip()

    text = text.replace("Response", "response")
    if "response:" in text:
        text = text.split("response:")[-1].strip()

    text = text.replace("Judgement", "judgement")
    if "judgement:" in text:
        text = text.split("judgement:")[-1].strip()

    text = text.replace("Score", "score")
    if "score:" in text:
        text = text.split("score:")[-1].strip()
    if "answer:" in text:
        text = text.split("answer:")[0].strip()

    return text


def normalize_text(text: str) -> str:
    """Normalize text with lowercasing, removing articles, and punctuation."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        exclude.add("'")
        exclude.add('"')
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(remove_articles(lower(text))))


def has_answer(text: str, answers: list[str]) -> float:
    """Check if text contains any of the answers."""
    norm_text = normalize_text(text)
    norm_answers = [normalize_text(ans) for ans in answers]
    # print(text,norm_text)
    return float(any([(ans in norm_text) for ans in norm_answers]))
    # return float(any([(ans.lower() in text.lower()) for ans in answers]))


def calc_exact_match(text: str, answers: list[str]) -> bool:
    """Check if prediction is exactly the same as any of the answers."""
    norm_text = normalize_text(text)
    norm_answers = [normalize_text(ans) for ans in answers]
    # print(norm_text,norm_answers)
    return max([(norm_text == norm_ans) for norm_ans in norm_answers])


def calc_unigram_f1(text: str, answers: list[str], field: str = "f1") -> float:
    """Calculate unigram f1 score between the text and reference answers."""
    norm_pred = normalize_text(text)
    norm_answers = [normalize_text(ans) for ans in answers]
    common_tokens = [
        Counter(norm_pred) & Counter(norm_ans) for norm_ans in norm_answers
    ]
    num_same = [sum(common.values()) for common in common_tokens]

    score_list = []
    for i, num in enumerate(num_same):
        if num == 0:
            score_list.append(0.0)
        else:
            p = 1.0 * num / len(norm_pred)
            r = 1.0 * num / len(norm_answers[i])
            f1 = 2 * p * r / (p + r)
            if field == "precision":
                score_list.append(p)
            elif field == "recall":
                score_list.append(r)
            elif field == "f1":
                score_list.append(f1)
            else:
                raise ValueError(f"Unknown field: {field}")
    return max(score_list)


def extract_pred(answer_ori):
    # return answer_ori
    # answer_ori = lower(answer_ori)
    # print(answer_ori)

    if "a summary of the contexts related to the question: " in answer_ori:
        return answer_ori.split("a summary of the contexts related to the question: ")[-1].strip()
    if "Excerpt:" in answer_ori:
        # print(answer_ori.split("Contribution:")[0].split("Excerpt:")[-1].strip())
        return answer_ori.split("Contribution:")[0].split("Excerpt:")[-1].strip()

    if "[/INST]" in answer_ori:
        answer_ori = answer_ori.split("[/INST]")[-1].strip()
    if "Answer:" in answer_ori:
        answer_ori = answer_ori.split("Answer:")[-1].strip()

    pattern = r"\[(.+)\]"
    # print(answer_ori)
    res = re.findall(pattern, answer_ori)
    # if len(res) > 1:
    #     if res[0] != res[1]:
    #         print(answer_ori)
    #         print("____")
    # print(answer_ori,res)
    # print("-----")
    if len(res) > 0:
        return res[0]

    # return answer_ori
    return answer_ori.strip().split("\n")[0].strip()


def pre_anwer_prompt_nortv(question):
    prompt = '''<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. 
    If you don't know the answer to a question, please don't share false information.
    Please make sure your answer is as concise as possible while still being correct!
    <</SYS>>

    Given the ['question'], predict the answer to the question.
    question: {} 

    [/INST] 
    answer: '''
    # question = "Given the ['question', 'context'], predict the answer to the question.\n\nquestion: who got the first nobel prize in physics\ncontext: Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was awarded to physicist Wilhelm R\u00f6ntgen in recognition of the extraordinary services he"
    prom = prompt.format(question)
    return prom


def pre_anwer_prompt(question, context):
    prompt = '''<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. 
    If you don't know the answer to a question, please don't share false information.
    Please make sure your answer is as concise as possible while still being correct!
    <</SYS>>

    Given the ['question', 'context'], predict the answer to the question.
    question: {} 
    context: 
    {}

    [/INST] 
    answer: '''
    # question = "Given the ['question', 'context'], predict the answer to the question.\n\nquestion: who got the first nobel prize in physics\ncontext: Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was awarded to physicist Wilhelm R\u00f6ntgen in recognition of the extraordinary services he"
    prom = prompt.format(question, context)
    return prom


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


def get_answer_str(tmp_ans):

    if type(tmp_ans) == str:
        ans_list = [tmp_ans]
    elif type(tmp_ans) == list:
        ans_list = tmp_ans
    elif type(tmp_ans) == dict:
        # print(answer["Aliases"])
        ans_list = tmp_ans["Aliases"]
    else:
        return []

    return ", ".join(ans_list)


def load_remote(remote_path, hostname="10.58.226.17"):

    # 服务器信息，主机名（IP地址）、端口号、用户名及密码
    # hostname = "10.58.226.17"
    port = 22
    username = "root"
    password = "Huawei12#$"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, password, compress=True)
    sftp_client = client.open_sftp()
    remote_file = sftp_client.open(remote_path, "r")  # 文件路径

    try:
        # print(json.load(remote_file)[0])
        if remote_path.endswith(".json"):
            return json.load(remote_file)
        elif remote_path.endswith(".jsonl"):
            return [json.loads(line.strip()) for line in remote_file]
        else:
            extension = remote_path.split(".")[-1]
            raise ValueError(f"File extension [{extension}] not valid.")
    finally:
        remote_file.close()


def load_dataset(path: str) -> list[dict]:
    """Load dataset from JSON or JSONL file."""
    if path.endswith(".json"):
        return json.load(open(path, "r"))
    elif path.endswith(".jsonl"):
        return [json.loads(line.strip()) for line in open(path, "r")]
    else:
        extension = path.split(".")[-1]
        raise ValueError(f"File extension [{extension}] not valid.")


def write_dataset(path: str, dataset: list[dict]):
    """Write dataset to JSON or JSONL file."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    if path.endswith(".json"):
        json.dump(dataset, open(path, "w"))
    elif path.endswith(".jsonl"):
        with open(path, "w") as fw:
            for res_dict in dataset:
                fw.write(json.dumps(res_dict) + "\n")
    else:
        extension = path.split(".")[-1]
        raise ValueError(f"File extension [{extension}] not valid.")


def select_random_dataset(dataset, sample_count):
    total_count = len(dataset)
    random_index = sorted(random.sample(range(total_count), sample_count))
    dataset_new = [dataset[index] for index in random_index]
    return random_index, dataset_new


def dataset_sample():
    path = '/home/huawei/kzhu/code-src/data/source'
    save = "/home/huawei/kzhu/code-src/data/source_sample"

    index_list = []
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        save_path = os.path.join(save, file)
        dataset = load_dataset(filepath)
        index, dataset_sam = select_random_dataset(dataset, 1000)
        write_dataset(save_path, dataset_sam)
        index.append(index)
    write_dataset(os.path.join(path,"sample_index.json"),index_list)


def dataset_split():
    path = './data/compressed_top5/exact_sft/2wiki_train.jsonl'
    save1 = "./data/compressed_top5/exact_sft/2wiki_train_8w.jsonl"
    save2 = "./data/compressed_top5/exact_sft/2wiki_train_left.jsonl"

    index_list = {}
    dataset = load_dataset(path)
    index, dataset_sam = select_random_dataset(dataset, 80000)
    write_dataset(save1, dataset_sam)
    dataset_left = []
    for i in range(len(dataset)):
        if i in index:
            continue
        else:
            dataset_left.append(dataset[i])
    write_dataset(save2, dataset_left)
    index_list["2wiki"] = index
    write_dataset(os.path.join("./data/compressed_top5/exact_sft","sample_index.json"),index_list)


def get_sample_index():
    path1 = '/home/huawei/kzhu/code-src/data/source_top20'
    path2 = "/home/huawei/kzhu/code-src/data/source_sample"

    index_dic = {}
    for file in os.listdir(path2):
        filepath = os.path.join(path1, file)
        sample_path = os.path.join(path2, file)
        dataset1 = load_dataset(filepath)
        dataset2 = load_dataset(sample_path)
        index_list = []
        last_index = 0
        for index, data in enumerate(dataset2):
            # print(last_index)
            # print(range(last_index, len(dataset1)))
            for i in range(last_index, len(dataset1)):
                # print(i)
                data_ori = dataset1[i]
                if data["question"] == data_ori["question"]:
                    index_list.append(i)
                    last_index = i
                    break
        print(len(dataset2), len(index_list))
        assert len(dataset2) == len(index_list)
        index_dic[file.replace("_dev_top100.jsonl",".json")] = index_list
    write_dataset(os.path.join(path1, "sample_index.json"), index_dic)


def get_sample_file():
    path1 = '/home/huawei/kzhu/code-src/data/source_top20/sample_index.json'
    path2 = "/home/huawei/kzhu/code-src/data/output/self_compress_output/"

    key = "tqa.json"
    file = "tqa_self_compress.json"
    save_file = "tqa_sam1000.json"

    index_dic = load_dataset(path1)
    dataset2 = load_dataset(os.path.join(path2,file))

    index_list = index_dic[key]
    sam_dataset = [dataset2[index] for index in index_list]
    write_dataset(os.path.join(path2, save_file), sam_dataset)


def remove_sample_file():
    path1 = '/home/huawei/kzhu/code-src/data/source_top20/sample_index.json'
    # path2 = "/home/huawei/kzhu/code-src/data/gen_answer/tqa_exact_top3/exact_para/"
    path2 = "/home/huawei/kzhu/code-src/data/source_top20"

    file = "tqa_dev_top100.jsonl"
    save_file = "tqa_dev_top3_newdev.jsonl"

    index_dic = load_dataset(path1)
    dataset2 = load_dataset(os.path.join(path2,file))

    index_list = index_dic["tqa.json"]
    sam_dataset = []
    for i in range(len(dataset2)):
        if i not in index_list:
            # sam_dataset.append(dataset2[i])
            tmp = {
                "question": dataset2[i]["question"],
                "answer": get_answer(dataset2[i]["answer"]),
                "retrieval": dataset2[i]["retrieval"][:3]
            }
            sam_dataset.append(tmp)
    print(len(sam_dataset))
    write_dataset(os.path.join(path2, save_file), sam_dataset)


def find_evidence():
    # path1="/home/huawei/duxiyuan/retrieval/source/HotPot/hotpot_dev_distractor_v1.json"
    # path1="/home/huawei/duxiyuan/retrieval/source/HotPot/hotpot_train_v1.1.json"
    path1="/home/huawei/duxiyuan/retrieval/source/2WikiMultihopQA/dev.json"
    path2="/home/huawei/kzhu/code-src/data/source_top20/2wiki_dev_top100.jsonl"
    save = "/home/huawei/kzhu/code-src/data/compressed_top5/has_evidence_top5/2wiki_dev.json"

    dataset_ori = load_dataset(path1)
    dataset_retr = load_dataset(path2)
    dataset_new = []
    assert len(dataset_retr) == len(dataset_ori)
    for i, ori in enumerate(dataset_ori):
        retr = dataset_retr[i]
        tmp = dict(retr)
        assert ori["question"] == retr["question"]
        tmp["retrieval"] = retr["retrieval"][:5]
        tmp["supporting_facts"] = [sf[0] for sf in ori["supporting_facts"]]
        dataset_new.append(tmp)
    write_dataset(save,dataset_new)



def prepare_tqa(path,outpath):
    dataset = load_dataset(path)["Data"]
    write_dataset(outpath,dataset)

# prepare_tqa("/home/huawei/duxiyuan/retrieval/source/trialQA/unfiltered-web-dev.json","./tviviaqa_dev.json")


def example_nq():
    path="/home/huawei/duxiyuan/retrieval/source/nq/nq_dev_short_answer.jsonl"
    dataset = load_dataset(path)
    for data in dataset[:10]:
        print(data["question"])
        fin = []
        for tmp in data["document_text"]:
            fin.append(tmp["token"])
        print(" ".join(fin))
        print(data["answer"])

    with open("./example/nq-dev.json",'w',encoding='utf-8') as f:
        json.dump(dataset[0:10],f,indent=2)
        # print(dataset[0])


def example():
    path="/home/kzhu/project/noise_filtering/src/llama-factory/data/gen_summary/ib_select/hotpot/top5_dev_ck4500.json"
    dataset = load_dataset(path)

    with open("./example/hotpot-final.json",'w',encoding='utf-8') as f:
        json.dump(dataset[100:400],f,indent=2)
        # print(dataset[0])


def data_splice():
    path="D:/Projects/code-src/data/top5_data/source_top5"
    file="nq_train_top100.jsonl"
    filepath=os.path.join(path,file)
    dataset = load_dataset(filepath)
    interval = 2800
    count = int(len(dataset)/interval) + 1
    for i in range(count):
        tmp_data = dataset[i*interval:(i+1)*interval]
        savepath=os.path.join(path,"split",file[:-6]+f"_{i}.jsonl")
        if not os.path.exists(savepath):
            write_dataset(savepath,tmp_data)

def data_combine():
    path="/home/kzhu/project/noise_filtering/src/data/gen_summary/ib_select/hotpot_train_loss2_10_mask_addno/"
    file="hotpot_top5_dev_ck13500_llama_ans_poslabel"
    dataset_all = []
    for count in range(100):
        filepath=os.path.join(path,file+f"_{count}.json")
        # print(filepath)
        if os.path.exists(filepath):
            # print(filepath)
            dataset = load_dataset(filepath)
            # print(len(dataset))
            dataset_all.extend(dataset)
    # path="/home/kzhu/project/noise_filtering/src/data/baseline"
    print(file,len(dataset_all))
    write_dataset(os.path.join(path,file+".jsonl"),dataset_all)


def example_answer():
    path="/home/huawei/kzhu/code-src/data/source_sample"
    save="/home/huawei/kzhu/code-src/data/source_top5"
    # dataset = load_dataset(path)

    for file in os.listdir(path):
        filepath = os.path.join(path,file)
        savepath = os.path.join(save, file)
        dataset = load_dataset(filepath)
        datasetnew = []
        for data in dataset:
            tmp = {
                "question": data["question"],
                "answer": get_answer(data["answer"]),
                "retrieval": [ret for ret in data["retrieval"][:5]]
            }
            datasetnew.append(tmp)

        with open(savepath,'w',encoding='utf-8') as f:
            json.dump(datasetnew,f,indent=2)


def check_count():
    path = "/home/huawei/duxiyuan/retrieval/kzhu/source"
    for file in os.listdir(path):
        filepath = os.path.join(path,file)
        print(filepath)
        dataset = load_dataset(filepath)
        print(len(dataset))


def prepare_top20():
    path = "/home/huawei/kzhu/retrieval/source/"
    save_path = "/home/huawei/kzhu/code-src/data/source_top20/"
    for file in os.listdir(path):
        filepath = os.path.join(path,file)
        savepath = os.path.join(save_path,file)
        dataset = load_dataset(filepath)
        dataset_new = []
        for data in dataset:
            ques = data["question"]
            ans = get_answer(data["answer"])
            contexts = data["retrieval"][:20]
            dataset_new.append({
                "question": ques,
                "answer": ans,
                "retrieval": contexts
            })

        write_dataset(savepath,dataset_new)


def check_dataset_cover():
    path1 = "/home/huawei/kzhu/code-src/data/source_top20/nq_dev_top100.jsonl"
    path2 = "/home/huawei/kzhu/code-src/data/source_top20/filco-nq-test.json"

    save = "/home/huawei/kzhu/code-src/data/source_top20/nq_dev_top100_newans.jsonl"

    our = load_dataset(path1)
    new = []
    filco = load_dataset(path2)

    print(len(our),len(filco))

    our_ques = [tmp["question"] for tmp in our]
    filco_ques = [tmp["question"] for tmp in filco]

    count = []
    # for q in our_ques:
    #     if q in filco_ques:
    #         count.append(1)

    for i,d in enumerate(our):
        tmp = dict(d)
        for j,n in enumerate(filco):
            if d["question"] == n["question"]:
                tmp["answer"] = n["answers"]
                tmp["retrieval"] = n["ctxs"]
                break
        new.append(tmp)
                # print(d["answer"])
                # print(n["answers"])
        # if d["question"] in filco_ques:
        #     print(d["answer"])
    write_dataset(save,new)
    print(len(new))
    print(sum(count),sum(count)/len(our))


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


if __name__=="__main__":
    # check_count()
    # example()
    # example_answer()
    # prepare_top20()
    # get_sample_index()
    # remove_sample_file()
    # check_dataset_cover()
    # find_evidence()
    # dataset_split()
    # get_sample_file()
    data_combine()
    # print(sorted(select_random_index(50, 10)))
