import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

def calculate(model, tokenizer, input_content, output_content):
    '''
    calculate information_bottle_neck:
    model: llama

    I(X,X'):
        paramters:
            input_content `list of sentences': (Query) + Compressed Content + (prompt) + ... 
            output_content `list of sentences': Retrieved Contents !!!only!!!

        return:
            avg_prob: average probability

    I(X',Y):
        parameters:
            input_content `list of sentences': (Query) + Compressed Content + (prompt) + ... 
            output_content `list of sentences': Answers !!!only!!!

        return:
            avg_prob: average probability

            
    examples:
        llama_path = "/share/home/fengxiaocheng/pretrained_models/llama-models/llama-2-7b"
        tokenizer = AutoTokenizer.from_pretrained(llama_path)
        input_content=["Question: Who founded google? Excerpt: Google, American search engine company, founded in 1998 by Sergey Brin and Larry Page.", \
                    "Question: Who was the first person killed in a car accident? Excerpt: The first driver fatality from a collision happened in 1898, when Englishman Henry Lindfield and his son were driving from Brighton to London."]
        output_content=["Context: Google, American search engine company, founded in 1998 by Sergey Brin and Larry Page, that is a subsidiary of the holding \
                    company Alphabet Inc. More than 70 percent of worldwide online search requests are handled by Google, placing it at the heart of most \
                    Internet users experience. Learn more about Google.", "Context: The first driver fatality from a collision (not counting Ward\'s unfortunate \
                    ejection) happened in 1898, when Englishman Henry Lindfield and his son were driving from Brighton to London. Near the end of their trip, \
                    Lindfield lost control of the car while going down a hill."]

        
        output_mask = tokenizer(output_content, return_tensors='pt', padding=True).attention_mask[:,:-1]
        batch_size, output_length = output_mask.shape
        all_content = [input_sentence + ' ' + output_sentence for input_sentence, output_sentence in zip(input_content, output_content)]
        all_tokens = tokenizer(all_content, return_tensors='pt', padding=True)
        
        input_ids = all_tokens.input_ids
        attention_mask = all_tokens.attention_mask

        target_mask = torch.cat([torch.zeros([batch_size, attention_mask.shape[-1] - output_length], dtype=torch.long), output_mask], dim=-1)
    '''
    ### tokenizer is required to be based on left padidng:
    device = model.device
    output_mask = tokenizer(output_content, return_tensors='pt', padding=True).attention_mask[:,:-1]
    batch_size, output_length = output_mask.shape
    all_content = [input_sentence + ' ' + output_sentence for input_sentence, output_sentence in zip(input_content, output_content)]
    all_tokens = tokenizer(all_content, return_tensors='pt', padding=True)

    input_ids = all_tokens.input_ids.to(device)
    attention_mask = all_tokens.attention_mask.to(device)
    # print(attention_mask.shape[-1], output_length)
    target_mask = torch.cat([torch.zeros([batch_size, attention_mask.shape[-1] - output_length], dtype=torch.long), output_mask], dim=-1).to(device)


    with torch.no_grad():
        logits = model(input_ids=input_ids, 
                attention_mask=attention_mask
            ).logits
        
        labels = input_ids
        labels[~target_mask.bool()] = -100
        target_mask = target_mask[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        batch_size, output_length = shift_labels.shape

        loss = loss_fct(shift_logits.view(batch_size * output_length, model.config.vocab_size), shift_labels.view(batch_size * output_length)).view(batch_size, output_length)

        prob = torch.exp(-loss)

        assert prob.shape == target_mask.shape

        avg_prob = torch.sum(prob * target_mask.float(), dim=-1) / target_mask.sum(dim=-1)

    return avg_prob

