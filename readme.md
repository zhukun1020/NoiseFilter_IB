# NoiseFilter_IB

The code for our ACL 2023 paper [An Information Bottleneck Perspective for Effective Noise Filtering on Retrieval-Augmented Generation](https://arxiv.org/abs/2406.01549)  is provided at this repo. If you have any questions, please reach me at [kzhu@ir.hit.edu.cn](kzhu@ir.hit.edu.cn).

## Data Construction

Our approach is not fixated on finding the optimal solution in the initial stages, but rather focuses on gradually approaching the optimal filter model through iterative training.
Here we choose the two easiest methods, which can greatly reduce computational costs on the construction of training data

The goal of `exact search` is to find the paragraphs or sentences containing the ground answers.  `Greedy search` is one of the most popular heuristic method by far used in extractive summarization. This algorithm extracts oracle labels with the highest ROUGE scores compared to human-annotated abstracts. 

We considered two silver summaries, one that concatenates the query and answer, and the other that focuses solely on the answer itself. 
The former can cover more information, while the latter focuses more on the answer itself.
Specially, the answer in intermediate state, supporting facts, are incorporated for multi-hop questions.

```shell
for oracle_mode in exact exact_para greedy_ans greedy
do
python pre_cands.py \
--oracle_mode ${oracle_mode} \
--source_path ./data/source/${dataset_name}.jsonl \
--compressed_path ./data/compressed/${oracle_mode}/${dataset_name}.jsonl
done
```

```shell
dataset_name=nq_dev
batch_size=2
max_example=5

python cal_ib.py \
--source_path ./data/source/ \
--compressed_path ./data/compressed/ \
--save_path ./data/combine/ \
--data_name ${dataset_name}.jsonl \
--save_name ${dataset_name}_loss.jsonl \
--model_path ./models/llama2_13b_chat_hf \
--batch_size ${batch_size} \
--max_example ${max_example}

```



