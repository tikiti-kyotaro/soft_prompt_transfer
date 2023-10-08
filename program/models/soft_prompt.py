import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformers import (AutoModelForCausalLM, GPT2LMHeadModel,
#                           GPTNeoForCausalLM)
from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, XGLMForCausalLM, XLMRobertaForMaskedLM
from transformers.utils import ModelOutput


class CustomOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    src_plm_loss: Optional[torch.FloatTensor] = None
    tgt_plm_loss: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PromptTuning:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        soft_prompt_path: str = None,
        n_tokens: int = None,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        use_tgt_data: bool = False,
        mask_token_id: int = None,
        kl_loss_rate: float = 0.0,
        tgt_loss_rate: float = 0.0,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.set_train_sterategy(use_tgt_data, mask_token_id, kl_loss_rate, tgt_loss_rate)
        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        elif n_tokens is not None:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )

        return model

    def set_train_sterategy(
        self,
        use_tgt_data: bool = False,
        mask_token_id: int = None,
        kl_loss_rate: float = 0.0,
        tgt_loss_rate: float = 0.0,
    ) -> None:
        self.use_tgt_data = use_tgt_data
        self.mask_token_id = mask_token_id
        if self.use_tgt_data:
            assert self.model_type == "xlm-roberta", f"Not implemented {self.model_type}"
            self.kl_loss = nn.KLDivLoss(reduction="batchmean")
            self.kl_loss_rate = kl_loss_rate
            self.tgt_loss_rate = tgt_loss_rate
            self.src_loss_rate = 1.0 - kl_loss_rate - tgt_loss_rate
            assert (
                self.src_loss_rate >= 0.0
            ), f"Set the sum of variables kl_loss_rate ({kl_loss_rate}) and tgt_loss_rate ({tgt_loss_rate}) to be less than 1.0."

    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path
        """
        self.soft_prompt = torch.load(soft_prompt_path, map_location=torch.device("cpu"))
        self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def initialize_soft_prompt(
        self,
        n_tokens: int = 20,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            if self.model_type == "xglm":
                init_prompt_value = self.model.embed_tokens.weight[:n_tokens].clone().detach()
                dim = self.config.d_model
            elif self.model_type == "xlm-roberta":
                init_prompt_value = self.roberta.embeddings.word_embeddings.weight[:n_tokens].clone().detach()
                dim = self.config.hidden_size
            else:
                init_prompt_value = self.transformer.wte.weight[:n_tokens].clone().detach()
                dim = self.config.n_embd
        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(-random_range, random_range)

        self.soft_prompt = nn.Embedding(n_tokens, dim)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        if self.model_type == "xglm":
            inputs_embeds = self.model.embed_tokens(input_ids)
        elif self.model_type == "xlm-roberta":
            inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)
        else:
            inputs_embeds = self.transformer.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        if self.model_type != "xglm":
            return inputs_embeds

        return inputs_embeds * self.model.embed_scale

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
            dim=1,
        )

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))
        # print(f"Saved soft prompt: {os.path.join(path, filename)}")

    def get_mask_token_index(self, input_ids: torch.Tensor):
        mask_token_idx_dim1, mask_token_idx_dim2 = torch.where(input_ids == self.mask_token_id)
        mask_token_idx_dim2 += self.n_tokens
        return mask_token_idx_dim1.tolist(), mask_token_idx_dim2.tolist()

    def get_mask_token_logits(
        self, mask_token_idx_dim1: torch.Tensor, mask_token_idx_dim2: torch.Tensor, plm_logits: torch.Tensor
    ):
        mask_token_logits = plm_logits[(mask_token_idx_dim1, mask_token_idx_dim2)]
        return F.softmax(mask_token_logits, dim=-1)

    def cal_kl_loss(
        self,
        src_plm_logits: torch.Tensor,
        src_input_ids: torch.Tensor,
        tgt_plm_logits: torch.Tensor,
        tgt_input_ids: torch.Tensor,
    ):
        src_mask_token_idx_dim1, src_mask_token_idx_dim2 = self.get_mask_token_index(input_ids=src_input_ids)
        tgt_mask_token_idx_dim1, tgt_mask_token_idx_dim2 = self.get_mask_token_index(input_ids=tgt_input_ids)

        # if mask token was not included in src or tgt example,
        # discard the src, tgt examples to calcurate kl loss
        if src_mask_token_idx_dim1 != tgt_mask_token_idx_dim1:
            mask_token_idx_dim1 = list(set(src_mask_token_idx_dim1) & set(tgt_mask_token_idx_dim1))
            src_mask_token_idx_dim2 = [
                idx2
                for idx1, idx2 in zip(src_mask_token_idx_dim1, src_mask_token_idx_dim2)
                if idx1 in mask_token_idx_dim1
            ]
            tgt_mask_token_idx_dim2 = [
                idx2
                for idx1, idx2 in zip(tgt_mask_token_idx_dim1, tgt_mask_token_idx_dim2)
                if idx1 in mask_token_idx_dim1
            ]
            assert (
                len(mask_token_idx_dim1) == len(src_mask_token_idx_dim2) == len(tgt_mask_token_idx_dim2)
            ), "Not match dim1 and dim2"
            src_mask_token_logits = self.get_mask_token_logits(
                mask_token_idx_dim1, src_mask_token_idx_dim2, src_plm_logits
            )
            tgt_mask_token_logits = self.get_mask_token_logits(
                mask_token_idx_dim1, tgt_mask_token_idx_dim2, tgt_plm_logits
            )
        else:
            src_mask_token_logits = self.get_mask_token_logits(
                src_mask_token_idx_dim1, src_mask_token_idx_dim2, src_plm_logits
            )
            tgt_mask_token_logits = self.get_mask_token_logits(
                tgt_mask_token_idx_dim1, tgt_mask_token_idx_dim2, tgt_plm_logits
            )

        return self.kl_loss(tgt_mask_token_logits.log(), src_mask_token_logits)

    def plm_forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(self.device)

        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        outs = super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=return_dict,
        )

        outs = CustomOutput(
            loss=outs.loss,
            src_plm_loss=None,
            tgt_plm_loss=None,
            kl_loss=None,
            logits=outs.logits,
            hidden_states=outs.hidden_states,
            attentions=outs.attentions,
        )

        return outs, labels

    def forward(
        self,
        input_ids=None,
        tgt_input_ids=None,
        past_key_values=None,
        attention_mask=None,
        tgt_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        tgt_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        do_inference: bool = False,
    ):
        src_plm_outs, src_labels = self.plm_forward(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        if not self.use_tgt_data:
            return src_plm_outs
        elif do_inference:
            return src_plm_outs, src_labels

        tgt_plm_outs, tgt_labels = self.plm_forward(
            input_ids=tgt_input_ids,
            labels=tgt_labels,
            attention_mask=tgt_attention_mask,
            return_dict=return_dict,
        )

        kl_loss = self.cal_kl_loss(
            src_input_ids=input_ids,
            src_plm_logits=src_plm_outs.logits,
            tgt_input_ids=tgt_input_ids,
            tgt_plm_logits=tgt_plm_outs.logits,
        )
        loss = (
            self.kl_loss_rate * kl_loss
            + self.tgt_loss_rate * tgt_plm_outs.loss
            + self.src_loss_rate * src_plm_outs.loss
        )

        return CustomOutput(
            loss=loss,
            src_plm_loss=src_plm_outs.loss,
            tgt_plm_loss=tgt_plm_outs.loss,
            kl_loss=kl_loss,
            logits=src_plm_outs.logits,
            hidden_states=src_plm_outs.hidden_states,
            attentions=src_plm_outs.attentions,
        )


class GPT2PromptTuningLM(PromptTuning, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_type = "gpt"


class GPTNeoPromptTuningLM(PromptTuning, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model_type = "gpt"


class XGLMPromptTuningLM(PromptTuning, XGLMForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model_type = "xglm"


class XLMRPromptTuningMLM(PromptTuning, XLMRobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.model_type = "xlm-roberta"
