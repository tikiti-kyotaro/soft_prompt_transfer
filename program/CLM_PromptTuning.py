import argparse
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn
from data.preprocess import load_create_prompt_function, tokenize_function, tokenize_function_ft
from datasets import load_dataset
from models.load_model import load_model, load_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score
from soft_prompt_trainer import PEFTTrainer, SoftPromptTrainer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from peft.src.peft import (
    PeftConfig,
    PeftModel,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import math
from data_programs.load_dataset import GetDataset
from data_programs.preprocess_funcs import PreprocessData
from soft_to_hard import soft_to_hard, get_prompt_embeds, cos_sim_measure, cos_sim_measure_decode
import pickle
from collections import defaultdict


class CLM_PromptTuning:
    def __init__(
            self,
            seed,
            model_name_or_path, 
            num_virtual_tokens,
            text_column,
            label_column, 
            train_file_name,
            valid_file_name,
            test_file_name,
            max_length, 
            lr, 
            num_epochs, 
            batch_size, 
            device, 
            dataset_name,
            save_model,
            train_model,
            check_output,
            early_stopping,
            harded_mode,
            sim_index,
            gen_model_name_or_path,
            diff_num_mode,
            only_change_mode,
            decode_per_epoch,
            temp_mode,
            test_mode
        ):
        self.seed = seed
        self.model_name_or_path = model_name_or_path
        self.num_virtual_tokens = num_virtual_tokens
        self.text_column = text_column
        self.label_column = label_column
        self.train_file_name = train_file_name
        self.valid_file_name = valid_file_name
        self.test_file_name = test_file_name
        self.max_length = max_length
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.dataset_name = dataset_name
        self.save_model = save_model
        self.train_model = train_model
        self.check_output = check_output
        self.early_stopping = early_stopping
        self.harded_mode = harded_mode
        self.sim_index = sim_index
        self.gen_model_name_or_path = gen_model_name_or_path
        self.diff_num_mode = diff_num_mode
        self.only_cahnge_mode = only_change_mode
        self.decode_per_epoch = decode_per_epoch
        self.temp_mode = args.temp_mode
        self.test_mode = test_mode

        if args.dataset_name == "gpt3mix/sst2":
            self.text_column = "text"
            self.label_column = "label"
            self.train_file_name = "train"
            self.valid_file_name = "validation"
            self.test_file_name = "test"
        
        elif args.dataset_name == "sst2":
            self.text_column = "sentence"
            self.label_column = "label"
            self.train_file_name = "train"
            self.valid_file_name = "train"
            self.test_file_name = "validation"
        
        elif args.dataset_name == "anli":  # 生成の時に用いるから後々修正する必要あり！！！！！！
            self.premise_column = "premise"
            self.hypothesis_column = "hypothesis"
            self.label_column = "label"
            train_file_name = "train"
            valid_file_name = "validation"
            test_file_name = "test"


################################## 前処理 ##################################
    def fix_seed(self):
        """
        seed固定
        """
        # random
        random.seed(self.seed)
        # Numpy
        np.random.seed(self.seed)
        # Pytorch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True


    def verbalizer(self, label):
        """
        バーバライザー
        """
        if self.dataset_name == "gpt2mix/sst2":
            if label == 0:
                return "positive"
            elif label == 1:
                return "negative"
        elif self.dataset_name == "sst2":
            if label == 0:
                return "negative"
            elif label == 1:
                return "positive"
        elif self.dataset_name == "anli":
            if label == 0:
                return "entailment"
            elif label == 1:
                return "neutral"
            elif label == 2:
                return "contradiction"


################################## 訓練 ##################################
    def train(self, train_dataloader, eval_dataloader, peft_config):
        """
        modelの訓練
        """
        print("***** Running Train *****")
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        model = get_peft_model(model, peft_config)
        wte = self.get_wte(model)
        model.print_trainable_parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * self.num_epochs),
        )
        # training and evaluation
        model = model.to(self.device)

        for i, epoch in enumerate(range(self.num_epochs)):
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            model.eval()
            eval_loss = 0
            eval_preds = []
            for step, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                eval_preds.extend(
                    self.tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                )
            eval_epoch_loss = eval_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            print(total_loss)
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

            if self.temp_mode:
                train_model_id = f"../../model/{self.dataset_name}/{self.model_name_or_path}/token_{self.num_virtual_tokens}/lr_{self.lr}/{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}_{epoch+1}"
            else:
                train_model_id = f"../../model/no_temp/{self.dataset_name}/{self.model_name_or_path}/token_{self.num_virtual_tokens}/lr_{self.lr}/{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}_{epoch+1}"
            model.save_pretrained(train_model_id)

    def decode_train(self, train_dataloader, eval_dataloader, peft_config):
        """
        modelの訓練
        """
        print("***** Running Train *****")
        if self.diff_num_mode:
            print("mode : diff_num")
        if self.only_cahnge_mode:
            print("mode : only_change")
        if self.decode_per_epoch:
            print("mode : per_epoch")
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        model = get_peft_model(model, peft_config)
        wte = self.get_wte(model)
        model.print_trainable_parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * self.num_epochs),
        )
        # training and evaluation
        model = model.to(self.device)

        prompt_embeds = model.prompt_encoder.embedding.weight
        _, pre_token_ids = self.get_prompt_embeds(model, wte)
        pre_prompt = prompt_embeds.to('cpu').detach().numpy().copy()

        for i, epoch in enumerate(range(self.num_epochs)):
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()  # 埋め込みで初期化以降、prompt_encoder.embedding.weight が更新されない...?
                lr_scheduler.step()
                optimizer.zero_grad()
                
                prompt_embeds, token_ids = self.get_prompt_embeds(model, wte)
                prompt_embeds_comp = prompt_embeds.to('cpu').detach().numpy().copy()

                # if np.array_equal(prompt_embeds_comp, pre_prompt) == False:
                #     # requires_grad を True
                #     prompt_embeds.requires_grad = True
                #     print(token_ids)
                #     # prompt_encoder.weight に代入
                #     # model.prompt_encoder.embedding.weight = torch.nn.Parameter(prompt_embeds)  これだとopimizer.step()で値が更新されなくなってしまう
                #     model.prompt_encoder.embedding.weight.data = prompt_embeds
                #     pre_prompt = prompt_embeds.to('cpu').detach().numpy().copy()
                #     pre_token_ids = token_ids

                diff_list, dif_num = self.get_dif_ids_list(token_ids, pre_token_ids)  # diff_list : 変わったidの部分だけ1になる　　dif_num : 変わったIDの数

                if self.diff_num_mode:
                    if dif_num >= 4:
                        prompt_embeds.requires_grad = True
                        print(token_ids)
                        model.prompt_encoder.embedding.weight.data = prompt_embeds
                        pre_prompt = prompt_embeds.to('cpu').detach().numpy().copy()
                        pre_token_ids = token_ids

                if self.only_cahnge_mode:
                    if dif_num:
                        prompt_embeds.requires_grad = True
                        print(token_ids)
                        for i in range(len(diff_list)):
                            if diff_list[i] == 1:
                                model.prompt_encoder.embedding.weight.data[i] = prompt_embeds[i]
                        pre_token_ids = token_ids
            
            if self.decode_per_epoch:
                prompt_embeds, token_ids = self.get_prompt_embeds(model, wte)
                diff_list, dif_num = self.get_dif_ids_list(token_ids, pre_token_ids)
                prompt_embeds.requires_grad = True
                print(token_ids)
                model.prompt_encoder.embedding.weight.data = prompt_embeds
                pre_token_ids = token_ids

            model.eval()
            eval_loss = 0
            eval_preds = []
            for step, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                eval_preds.extend(
                    self.tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                )
            eval_epoch_loss = eval_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

            if self.diff_num_mode:
                train_model_id = f"../../model/decode/diff_num/{self.dataset_name}/{self.model_name_or_path}/token_{self.num_virtual_tokens}/lr_{self.lr}/{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}_{epoch+1}_not_gpt3mix_decode"
            elif self.only_cahnge_mode:
                train_model_id = f"../../model/decode/only_change/{self.dataset_name}/{self.model_name_or_path}/token_{self.num_virtual_tokens}/lr_{self.lr}/{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}_{epoch+1}_not_gpt3mix_decode"
            elif self.decode_per_epoch:
                train_model_id = f"../../model/decode/change_epoch/{self.dataset_name}/{self.model_name_or_path}/token_{self.num_virtual_tokens}/lr_{self.lr}/{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}_{epoch+1}_not_gpt3mix_decode"
            model.save_pretrained(train_model_id)


    def get_prompt_embeds(self, model, wte):
        # soft prompt を持ってくる
        soft_prompt = model.prompt_encoder.embedding.weight  # batch * token * hidden_size
        # soft prompt = backwardした埋め込みに一番近い単語埋め込み
        token_ids = list()
        prompt_embeds = torch.zeros(soft_prompt.shape[0], soft_prompt.shape[1], dtype=model.prompt_encoder.embedding.weight.dtype, device=self.device)
        for i in range(self.num_virtual_tokens):
            vector = soft_prompt[i,:]
            # ボキャブラリーから最も類似するベクトルを持つ単語を選択（内積を類似度とする）
            similarity = cos_sim_measure_decode(vector, wte)
            token_embeds = wte[int(similarity.argmax())]
            prompt_embeds[i,:] = token_embeds
            token_ids.append(int(similarity.argmax()))
        return prompt_embeds, token_ids
    
    def get_dif_ids_list(self, token_list, pre_token_list):
        diff_list = [0] * 10
        diff_num = 0
        for i, (x, y) in enumerate(zip(token_list, pre_token_list)):
            if x != y:
                diff_list[i] = 1
            if diff_list[i] == 1:
                diff_num += 1
        return diff_list, diff_num

################################## 生成 ##################################
    def soft_prompt_gen(self, dataset, model, peft_config):
        """
        OPTのgenerate関数を使う（soft prompt 適応）
        """
        output_path = f"/home/kyotaro/peft_test/outputs/{self.model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}_{self.dataset_name}_not_gpt3mix.txt"
        correct = 0
        with open(output_path, "w") as out_:
            if self.test_mode == "test":
                file_name = self.test_file_name
            elif self.test_mode == "valid":
                file_name = self.valid_file_name
            for i in tqdm(range(dataset["test"].num_rows)):
                inputs = self.tokenizer(f'{dataset["test"][i][self.text_column]} It is ', return_tensors="pt") # shape [1, 26]
                answer = self.verbalizer(dataset["test"][i][self.label_column])
                with torch.no_grad():
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = model.generate(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=2, eos_token_id=3, temperature=0
                    )
                    result_list = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)  # 出力した全文
                    result = result_list[0].split()[-1]
                    print(result_list)
                    if result == answer:
                        correct += 1
                    if self.check_output:
                        out_.write(f'{result}\n')
            print(correct)
        print(correct / dataset["test"].num_rows)
        return correct / dataset["test"].num_rows
    
    def sum_soft_prompt_gen(self, dataset, peft_config):
        accuracy_dict = defaultdict(lambda: float)
        for i in range(90, 100):
            print(f'epoch{i+1}')
            peft_model_id = f"/home/kyotaro/peft_clean/model/{self.dataset_name}/{self.model_name_or_path}/token_{self.num_virtual_tokens}/lr_{self.lr}/{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}_{i+1}"
            config = PeftConfig.from_pretrained(peft_model_id)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(model, peft_model_id)

            model.to(self.device)
            model.eval()
            print(peft_model_id)

            accuracy = self.soft_prompt_gen(dataset=dataset, model=model, peft_config=peft_config)
            accuracy_dict[i+1] = accuracy
        accuracy_dict_path = f"/home/kyotaro/peft_clean/result_dict/{self.dataset_name}/{self.model_name_or_path}/token_{self.num_virtual_tokens}/lr_{self.lr}/{self.model_name_or_path}.pkl"
        # with open(accuracy_dict_path, "wb") as f:
        #     pickle.dump(accuracy_dict, f)
        print(accuracy_dict)
        for key, value in accuracy_dict.items():
            print(value)
        print(max(accuracy_dict, key=accuracy_dict.get))

    def soft_prompt_gen_anli(self, dataset, model, peft_config):
        """
        OPTのgenerate関数を使う（soft prompt 適応）
        """
        output_path = f"/home/kyotaro/peft_clean/outputs/test.txt"
        correct = 0
        with open(output_path, "w") as out_:
            for i in tqdm(range(dataset[self.test_file_name].num_rows)):
                inputs = self.tokenizer(f'premise: {dataset[self.test_file_name][i][self.premise_column]}\nhypothesis: {dataset[self.test_file_name][i][self.hypothesis_column]}\nprediction: ', return_tensors="pt") # shape [1, 26]
                answer = self.verbalizer(dataset[self.test_file_name][i][self.label_column])
                with torch.no_grad():
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = model.generate(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=2, eos_token_id=3, temperature=0
                    )
                    result_list = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)  # 出力した全文
                    print(result_list)
                    result = result_list[0].split()[-1]
                    if result == answer:
                        correct += 1
                    if self.check_output:
                        out_.write(f'{result}\n')
            print(correct)
        print(correct / dataset[self.test_file_name].num_rows)    

    def base_model_gen_hard(self, dataset, model, peft_config, harded_prompt):
        """
        OPTのgenerate関数を使う（soft prompt 使わない）
        """
        output_path = f"/home/kyotaro/peft_test/outputs/{self.model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}_basemodel2.txt"
        correct = 0
        with open(output_path, "w") as out_:
            for i in tqdm(range(dataset[self.test_file_name].num_rows)):
                inputs = self.tokenizer(f'{dataset[self.test_file_name][i][self.text_column]} It is ', return_tensors="pt") # shape 26
                answer = self.verbalizer(dataset[self.test_file_name][i][self.label_column])
                with torch.no_grad():
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = model.base_model.generate(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=2, eos_token_id=3, temperature=0
                    )
                    result_list = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
                    result = result_list[0].split()[-1]
                    if result == answer:
                        correct += 1
                    if self.check_output:
                        out_.write(f'{result}\n')
        print(correct / dataset[self.test_file_name].num_rows)


    def gen_only_labels(self, dataset, model, classes, peft_config):
        """
        ラベルを指定してそれしか出さない生成方法
        """
        output_path = f"/home/kyotaro/peft_test/outputs/{self.model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}_label_only.txt"
        correct = 0
        with torch.no_grad():
            with open(output_path, "w") as out_:
                id_dict = self.check_labels_id(classes)
                for i in tqdm(range(dataset[self.test_file_name].num_rows)):
                    inputs = f'{dataset[self.test_file_name][i][self.text_column]} It is '
                    answer = self.verbalizer(dataset[self.test_file_name][i][self.label_column])
                    vocab_logit = self.check_logit(model, inputs)  # vocab
                    if vocab_logit[id_dict["positive"]] > vocab_logit[id_dict["negative"]]:
                        result = "positive"
                    else:
                        result = "negative"
                    out_.write(f'{result}\n')
                    if result == answer:
                        correct += 1
        print(correct / dataset[self.test_file_name].num_rows)


    def gen_only_labels_harded(self, dataset, classes, peft_config, harded_prompt):
        """
        自然言語にしたバージョン
        """
        output_path = f"/home/kyotaro/peft_test/outputs/{self.model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}_label_only_harded_not_gpt3mix.txt"

        model_gen = AutoModelForCausalLM.from_pretrained(self.gen_model_name_or_path)
        # model_gen = AutoModelForCausalLM.from_pretrained(self.gen_model_name_or_path)
        correct = 0
        with open(output_path, "w") as out_:
            id_dict = self.check_labels_id(classes)
            for i in tqdm(range(dataset[self.test_file_name].num_rows)):
                if harded_prompt:
                    inputs = f'{harded_prompt} {dataset[self.test_file_name][i][self.text_column]} It is '
                else:
                    inputs = f'{dataset[self.test_file_name][i][self.text_column]} It is '
                if i == 0:
                    print(inputs)
                # print(inputs)
                answer = self.verbalizer(dataset[self.test_file_name][i][self.label_column])
                vocab_logit = self.check_logit(model_gen, inputs)
                if vocab_logit[id_dict["positive"]] > vocab_logit[id_dict["negative"]]:
                    result = "positive"
                else:
                    result = "negative"
                out_.write(f'{result}\n')
                if result == answer:
                    correct += 1
        print(correct / dataset[self.test_file_name].num_rows)


    def gen_only_labels_harded_new(self, dataset, classes, peft_config, harded_prompt):
        """
        自然言語にしたバージョン (BOS)
        """
        model_gen = AutoModelForCausalLM.from_pretrained(self.gen_model_name_or_path)
        output_path = f"/home/kyotaro/peft_test/outputs/{self.model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}_label_only_harded_new.txt"
        correct = 0
        harded_prompt_token = self.tokenizer(harded_prompt)
        harded_prompt_token["input_ids"] = harded_prompt_token["input_ids"][1:]
        harded_prompt_token["attention_mask"] = harded_prompt_token["attention_mask"][1:]
        # harded_prompt_token_new = {'input_ids': torch.tensor([harded_prompt_token["input_ids"][1:]]), 'attention_mask': torch.tensor([harded_prompt_token["attention_mask"][1:]])}

        with open(output_path, "w") as out_:
            id_dict = self.check_labels_id(classes)
            for i in tqdm(range(dataset[self.test_file_name].num_rows)):
                inputs_sentence = f'{dataset[self.test_file_name][i][self.text_column]} It is '
                answer = self.verbalizer(dataset[self.test_file_name][i][self.label_column])
                inputs = self.tokenizer(inputs_sentence)
                if len(harded_prompt.split()) > 0:
                    input_ids = harded_prompt_token["input_ids"] + inputs["input_ids"]
                    attention_mask = harded_prompt_token["attention_mask"] + inputs["attention_mask"]
                    inputs = {"input_ids": torch.tensor([input_ids]), "attention_mask": torch.tensor([attention_mask])}
                else:
                    inputs = {"input_ids": torch.tensor([inputs["input_ids"]]), "attention_mask": torch.tensor([inputs["attention_mask"]])}
                vocab_logit = self.check_logit_new(model_gen, inputs["input_ids"], inputs["attention_mask"])
                if vocab_logit[id_dict["positive"]] > vocab_logit[id_dict["negative"]]:
                    result = "positive"
                else:
                    result = "negative"
                out_.write(f'{result}\n')
                if result == answer:
                    correct += 1
        print(correct / dataset[self.test_file_name].num_rows)

    def check_logit_new(self, model, input_ids, attention_mask):
        """
        modelのフォワード関数に入れて生成確率を見る
        """
        input_ids = input_ids.to(self.device)  # [1, sentence_tokens]
        attention_mask = attention_mask.to(self.device)  # [1, sentence_tokens]
        model = model.to(self.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        softmax_logits = torch.softmax(outputs.logits[0,-2], dim=0)
        return softmax_logits

    def check_logit(self, model, inputs):
        """
        modelのフォワード関数に入れて生成確率を見る
        """
        inputs = self.tokenizer(inputs, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)  # [1, sentence_tokens]
        attention_mask = inputs["attention_mask"].to(self.device)  # [1, sentence_tokens]
        model = model.to(self.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        softmax_logits = torch.softmax(outputs.logits[0,-2], dim=0)
        return softmax_logits

    
    def check_labels_id(self, classes):
        """
        生成したいラベルのIDを調べる
        """
        id_dict = dict()
        for class_label in classes:
            if self.dataset_name == "gpt2mix/sst2":
                id_dict[class_label] = self.tokenizer(class_label)["input_ids"][1]
            elif self.dataset_name == "sst2":
                a = self.tokenizer(class_label)
                id_dict[class_label] = self.tokenizer(class_label)["input_ids"][0]
        return id_dict

    def get_wte(self, model):
        # 既存の単語に対するベクトルを抽出
        for named_param, value in model.base_model.named_parameters():
            if value.shape[0] == model.base_model.config.vocab_size:
                wte = value
                break
        return wte
################################## メイン ##################################

    def main(self):
        print(type(self.train_model))
        print(self.train_model)
        self.fix_seed()
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.num_virtual_tokens,
            tokenizer_name_or_path=self.model_name_or_path,
        )

        GD = GetDataset(self.dataset_name, self.seed)
        dataset = GD.get_dataset()
        classes = GD.get_classes()

        # data preprocessing
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        target_max_length = max([len(self.tokenizer(class_label)["input_ids"]) for class_label in classes])
        print(dataset)
        PD = PreprocessData(self.tokenizer, self.dataset_name, dataset, self.batch_size, self.temp_mode)
        train_dataloader, eval_dataloader = PD.get_preprocess_dataloader_train_eval()

        if self.train_model:
            # model = self.train(train_dataloader, eval_dataloader, peft_config)
            # model = self.decode_train(train_dataloader, eval_dataloader, peft_config)
            model = self.train(train_dataloader, eval_dataloader, peft_config)
            if self.save_model:
                peft_model_id = f"{self.model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}"
                model.save_pretrained(peft_model_id)


        # peft_model_id = f"{self.model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_{self.num_epochs}_{self.batch_size}"
        # peft_model_id = "/home/kyotaro/peft_test/models/facebook/opt-350M_PROMPT_TUNING_CAUSAL_LM_100_1_2"
        # peft_model_id = "/home/kyotaro/peft_test/models/gpt2-medium_PROMPT_TUNING_CAUSAL_LM_32_1_1_not_gpt3mix"
        # peft_model_id = "/home/kyotaro/peft_test/models/gpt2-medium/PROMPT_TUNING_CAUSAL_LM_32_1_30_not_gpt3mix"
        # peft_model_id = "/home/kyotaro/peft_test/models/gpt2-medium/PROMPT_TUNING_CAUSAL_LM_32_1_32_not_gpt3mix"
        # peft_model_id = "/home/kyotaro/peft_test/models/not_gpt3mix/facebook/opt-125M/PROMPT_TUNING_CAUSAL_LM_100_1_10_not_gpt3mix"
        # peft_model_id = "/home/kyotaro/peft_test/models/not_gpt3mix/facebook/opt-125M/PROMPT_TUNING_CAUSAL_LM_100_1_31_not_gpt3mix"
        # peft_model_id = "/home/kyotaro/peft_test/models/facebook/opt-350M_PROMPT_TUNING_CAUSAL_LM_100_1_60"
        # peft_model_id = "/home/kyotaro/peft_clean/model/facebook/opt-125M/PROMPT_TUNING_CAUSAL_LM_100_1_23_not_gpt3mix"
        peft_model_id = "/home/kyotaro/peft_clean/model/anli/facebook/opt-125M/token_10/lr_0.03/PROMPT_TUNING_CAUSAL_LM_100_1_24_not_gpt3mix"

        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)

        model.to(self.device)
        model.eval()
        print(peft_model_id)

        # harded_prompt = soft_to_hard(self.model_name_or_path, self.num_virtual_tokens, peft_model_id)
        # harded_prompt = "ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ is filmmakingquickShip� TheNitromeÃÂÃÂÃÂÃÂ"
        # harded_prompt = "showc filmmakingÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ���embedreportprint TheNitromereportprint"
        # print("start gen")
        # harded_prompt = ""
        # print(harded_prompt)
        # self.gen_only_labels(dataset, model, classes, peft_config)
        self.soft_prompt_gen(dataset=dataset, model=model, peft_config=peft_config)
        # self.sum_soft_prompt_gen(dataset=dataset, peft_config=peft_config)
        # self.soft_prompt_gen_anli(dataset=dataset, model=model, peft_config=peft_config)
        # self.gen_only_labels_harded(dataset, classes, peft_config, harded_prompt)
        # # self.base_model_gen_hard(dataset, model, peft_config, harded_prompt)
        # # self.gen_only_labels_harded_new(dataset, classes, peft_config, harded_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2-xl",
        help="model name or path",
    )
    parser.add_argument("--num_virtual_tokens", type=int, default=10)  # 要チューニング
    parser.add_argument("--task_type", type=str, default="CAUSAL_LM")
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument("--model_type", default="opt", type=str)
    parser.add_argument("--lr", type=float, default=3e-5)  # 要チューニング
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--dataset_name", type=str, default="sst2")
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--train_model", type=bool, default=False)
    parser.add_argument("--check_output", type=bool, default=True)
    parser.add_argument("--early_stopping", type=int, default=10)
    parser.add_argument("--harded_mode", type=bool, default=True)
    parser.add_argument("--sim_index", type=str, default="eucrid")
    parser.add_argument("--gen_model_name_or_path", type=str, default="gpt2-xl")
    parser.add_argument("--diff_num_mode", type=bool, default=False)
    parser.add_argument("--only_change_mode", type=bool, default=False)
    parser.add_argument("--decode_per_epoch", type=bool, default=False)
    parser.add_argument("--temp_mode", type=bool, default=True)
    parser.add_argument("--test_mode", type=str, default="test")
    args = parser.parse_args()

    if args.dataset_name == "gpt3mix/sst2":
        text_column = "text"
        label_column = "label"
        train_file_name = "train"
        valid_file_name = "validation"
        test_file_name = "test"
        
    elif args.dataset_name == "sst2":
        text_column = "sentence"
        label_column = "label"
        train_file_name = "train"
        valid_file_name = "train"
        test_file_name = "validation"
    
    elif args.dataset_name == "amazon_polarity":
        text_column = "title"
        label_column = "label"
        train_file_name = "train"
        valid_file_name = None
        test_file_name = "test"
    
    elif args.dataset_name == "ag_news":
        text_column = "text"
        label_column = "label"
        train_file_name = "train"
        valid_file_name = None
        test_file_name = "test"
    
    elif args.dataset_name == "anli":  # 生成の時に用いるから後々修正する必要あり！！！！！！
        text_column = "premise"
        label_column = "label"
        train_file_name = "train"
        valid_file_name = "validation"
        test_file_name = "test"
    
    clm_prompt_tuning = CLM_PromptTuning(
        seed=args.seed,
        model_name_or_path=args.model_name_or_path,
        num_virtual_tokens=args.num_virtual_tokens,
        text_column=text_column, 
        label_column=label_column,
        train_file_name=train_file_name,
        valid_file_name=valid_file_name,
        test_file_name=test_file_name,
        max_length=args.max_length, 
        lr=args.lr, 
        num_epochs=args.num_epochs, 
        batch_size=args.batch_size, 
        device="cuda", 
        dataset_name=args.dataset_name,
        save_model=args.save_model,
        train_model=args.train_model,
        check_output=args.check_output,
        early_stopping=args.early_stopping,
        harded_mode=args.harded_mode,
        sim_index=args.sim_index,
        gen_model_name_or_path=args.gen_model_name_or_path,
        diff_num_mode=args.diff_num_mode,
        only_change_mode=args.only_change_mode,
        decode_per_epoch=args.decode_per_epoch,
        temp_mode = args.temp_mode,
        test_mode=args.test_mode
        )
    
    clm_prompt_tuning.main()
