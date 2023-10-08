from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft.src.peft import (
    PeftConfig,
    PeftModel,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
import numpy as np
import torch


def get_prompt_embeds(model_name, batch_size, peft_model_id):
    """
    プロンプトの埋め込み部分だけ取得
    """
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=10,
        tokenizer_name_or_path=model_name,
    )
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)
    prompt_embeds = model.get_prompt(batch_size)
    return prompt_embeds

def cos_sim_measure(vector, matrix):
    """
    コサイン類似度測定
    """
    vector = vector.to('cuda')
    matrix = matrix.to('cuda')
    dot = vector @ matrix.T
    vector_norm = (vector * vector).sum(axis=1, keepdims=True) ** .5
    matrix_norm = (matrix * matrix).sum(axis=1, keepdims=True) ** .5
    cos_sim = dot / vector_norm / matrix_norm.T
    return cos_sim

def cos_sim_measure_decode(vector, matrix):
    """
    コサイン類似度測定
    """
    vector = vector.to('cuda')
    vector = vector.unsqueeze(dim=0)
    matrix = matrix.to('cuda')
    dot = vector @ matrix.T
    # vector_norm = np.linalg.norm(vector, ord=2)
    # matrix_norm = np.linalg.norm(matrix, ord=2)
    vector_norm = (vector * vector).sum(axis=1, keepdims=True) ** .5
    matrix_norm = (matrix * matrix).sum(axis=1, keepdims=True) ** .5
    cos_sim = dot / vector_norm / matrix_norm.T
    return cos_sim


def soft_to_hard(model_name, gen_model_name, lr, num_virtual_tokens, peft_model_id, sim_mode, sim_index):
    tokenizer_sh = AutoTokenizer.from_pretrained(model_name)
    config_sh = AutoConfig.from_pretrained(model_name)
    model_sh = AutoModelForCausalLM.from_pretrained(model_name, config=config_sh)

    soft_prompt = get_prompt_embeds(model_name, 1, peft_model_id)  # batch * token * hidden_size

    # 既存の単語に対するベクトルを抽出
    for named_param, value in model_sh.base_model.named_parameters():
        if value.shape[0] == model_sh.base_model.config.vocab_size:
            wte = value
            break

    token_ids = []
    similarity_output= 0
    if sim_index == "average":
        for i in range(num_virtual_tokens):
            vector = soft_prompt[:,i,:]
            # ボキャブラリーから最も類似するベクトルを持つ単語を選択（内積を類似度とする）
            similarity = cos_sim_measure(vector, wte)
            token_id = int(similarity.argmax())
            token_ids.append(token_id)
            similarity_output += similarity[0,token_id]
            print(token_id)
            print(tokenizer_sh.decode([token_id]))
        similarity_output /= num_virtual_tokens
    
    elif sim_index == "max":
        sim_list = list()
        for i in range(num_virtual_tokens):
            vector = soft_prompt[:, i, :]
            similarity = cos_sim_measure(vector, wte)
            token_id = int(similarity.argmax())
            token_ids.append(token_id)
            sim_list.append(similarity[0, token_id])
        similarity_output = max(sim_list)
            
    
    prompt = tokenizer_sh.decode(token_ids)
    print(prompt)
    similarity_path = f'/home/kyotaro/peft_clean/similarities/{model_name}/{gen_model_name}/{sim_mode}_{lr}_{sim_index}.txt'

    with open(similarity_path, "a") as sim_path:
        sim_path.write(f'{similarity_output}\n')
    return prompt



############## eucrid ###############

def soft_to_hard_eucrid(model_name, gen_model_name, lr, num_virtual_tokens, peft_model_id, sim_mode):
    tokenizer_sh = AutoTokenizer.from_pretrained(model_name)
    config_sh = AutoConfig.from_pretrained(model_name)
    model_sh = AutoModelForCausalLM.from_pretrained(model_name, config=config_sh)

    soft_prompt = get_prompt_embeds(model_name, 1, peft_model_id)  # batch * token * hidden_size

    # 既存の単語に対するベクトルを抽出
    for named_param, value in model_sh.base_model.named_parameters():
        if value.shape[0] == model_sh.base_model.config.vocab_size:
            wte = value
            break

    token_ids = []
    similarity_ave = 0
    similarity_ave_list = list()
    for i in range(num_virtual_tokens):
        vector = soft_prompt[:,i,:]
        # ボキャブラリーから最も類似するベクトルを持つ単語を選択（内積を類似度とする）
        similarity = eucrid_measure(vector, wte)
        token_id = int(np.argmin(similarity))
        token_ids.append(token_id)
        similarity_ave += similarity[token_id]
        print(token_id)
        print(tokenizer_sh.decode([token_id]))
    similarity_ave /= num_virtual_tokens
    
    prompt = tokenizer_sh.decode(token_ids)
    print(prompt)
    similarity_path = f'/home/kyotaro/peft_clean/similarities/{model_name}/{gen_model_name}/{sim_mode}_{lr}.txt'
    with open(similarity_path, "a") as sim_path:
        sim_path.write(f'{similarity_ave}\n')
    return prompt

def eucrid_measure(vector, wte):
    """
    ユークリッド距離の確認
    """
    sim_array = np.zeros(wte.shape[0])
    vector = vector.to('cuda')
    wte = wte.to('cuda')
    for i in range(wte.shape[0]):
        w = wte[i, :]
        L2_dist = torch.dist(vector,w,p=2)
        sim_array[i] = L2_dist
    return sim_array

# if __name__ == "__main__":
#     peft_model_id = "/home/kyotaro/peft_clean/model/sst2/gpt2/token_10/lr_3e-05/PROMPT_TUNING_CAUSAL_LM_100_1_100_not_gpt3mix"
#     soft_to_hard(model_name="gpt2", gen_model_name="gpt2", lr=3e-5, num_virtual_tokens=10, peft_model_id=peft_model_id, sim_mode="cosine", sim_index="average")