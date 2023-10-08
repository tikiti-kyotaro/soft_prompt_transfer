from models.soft_prompt import GPT2PromptTuningLM, XGLMPromptTuningLM, XLMRPromptTuningMLM
from peft.src.peft import (
    PeftConfig,
    PeftModel,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM


def load_peft_model(
    model_name_or_path: str,
    model_type: str,
    n_prompt_tokens: int = None,
    load_trained_model: bool = False,
    use_8bit: bool = False,
    init_type: str = "first",
):
    if load_trained_model:
        config = PeftConfig.from_pretrained(model_name_or_path)
        if config.task_type == "MASK_LM":
            model = AutoModelForMaskedLM.from_pretrained(
                config.base_model_name_or_path, load_in_8bit=use_8bit, device_map="auto"
            )
        elif config.task_type == "CAUSAL_LM":
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, load_in_8bit=use_8bit, device_map="auto"
            )
        if use_8bit:
            model = prepare_model_for_int8_training(model)
        model = PeftModel.from_pretrained(model, model_name_or_path)
        model.print_trainable_parameters()
        return model

    init_type = PromptTuningInit.FIRST if init_type == "first" else PromptTuningInit.RANDOM
    if model_type != "xlm-roberta":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=init_type,
            num_virtual_tokens=n_prompt_tokens,
            tokenizer_name_or_path=model_name_or_path,
            base_model_name_or_path=model_name_or_path,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, load_in_8bit=use_8bit, device_map="auto"
        )
        if use_8bit:
            model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, peft_config)
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, load_in_8bit=use_8bit, device_map="auto"
        )
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=init_type,
            num_virtual_tokens=n_prompt_tokens,
            tokenizer_name_or_path=model_name_or_path,
            base_model_name_or_path=model_name_or_path,
            token_dim=model.config.hidden_size,
        )
        if use_8bit:
            model = prepare_model_for_int8_training(model)
        model = PeftModel(model, peft_config)

    model.print_trainable_parameters()
    return model


def load_model(
    model_name: str,
    model_type: str,
    n_prompt_tokens: int = None,
    ft_or_pt_model_path: str = None,
    do_hard: str = None,
    use_tgt_data: bool = False,
    kl_loss_rate: float = 0.0,
    tgt_loss_rate: float = 0.0,
    mask_token_id: int = None,
):
    if do_hard:
        if model_type != "xlm-roberta":
            return AutoModelForCausalLM.from_pretrained(model_name)
        else:
            return AutoModelForMaskedLM.from_pretrained(model_name)

    if ft_or_pt_model_path:
        model = load_soft_model(model_type).from_pretrained(model_name, soft_prompt_path=ft_or_pt_model_path)
    else:
        model = load_soft_model(model_type).from_pretrained(
            model_name,
            n_tokens=n_prompt_tokens,
            initialize_from_vocab=True,
            use_tgt_data=use_tgt_data,
            mask_token_id=mask_token_id,
            kl_loss_rate=kl_loss_rate,
            tgt_loss_rate=tgt_loss_rate,
        )

    return model


def load_soft_model(model_type: str):
    model_mapper = {
        "xglm": XGLMPromptTuningLM,
        "gpt2": GPT2PromptTuningLM,
        "xlm-roberta": XLMRPromptTuningMLM,
    }
    assert model_type in model_mapper, f"Sorry, {model_type} is not implemented..."
    return model_mapper[model_type]
