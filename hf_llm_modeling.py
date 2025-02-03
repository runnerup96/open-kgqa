import os
import sys
import torch
import json
import math
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments, set_seed
)
import hf_llm_args
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.utils import logging
import text2query_llm_dataset
from lmm_mapping_constants import LLM_MAPPING_DICT

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print('No GPU detected!')
        # sys.exit()

    logger = logging.get_logger(__name__)

    parser = HfArgumentParser(hf_llm_args.ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=device)

    print('Loaded model!')

    # read data
    training_sft_dataset, validation_sft_dataset = [], []
    new_rubq_tokens = ['wdt:', 'skos:', 'wd:', 'ps:', 'pq:']
    tokenizer.add_tokens(new_rubq_tokens)
    model.resize_token_embeddings(len(tokenizer))

    training_sft_dataset = json.load(open(args.path_to_training_file, 'r'))
    validation_sft_dataset = json.load(open(args.path_to_testing_file, 'r'))

    if args.try_one_batch:
        training_sft_dataset = training_sft_dataset[:args.per_device_train_batch_size]
        validation_sft_dataset = training_sft_dataset


    tokenized_train_sft_dataset = text2query_llm_dataset.LlmFinetuneDataset(sft_dataset=training_sft_dataset,
                                                                            device=device, tokenizer=tokenizer,
                                                                           max_sft_length=args.max_seq_length)
    tokenized_validation_sft_dataset = text2query_llm_dataset.LlmFinetuneDataset(sft_dataset=validation_sft_dataset,
                                                                           device=device, tokenizer=tokenizer,
                                                                           max_sft_length=args.max_seq_length)

    print('Training samples total size: ', len(tokenized_train_sft_dataset))

    # following https://arxiv.org/pdf/2305.14314
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=128,
            bias="none",
            target_modules=['q_proj', 'v_proj',
                            'k_proj', 'o_proj',
                            'gate_proj',
                            'up_proj', 'down_proj'],
            task_type="CAUSAL_LM",
        )

    batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    num_update_steps_per_epoch = len(training_sft_dataset) // batch_size
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    total_train_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    print('My total train steps: ', total_train_steps)

    num_warmup_steps = int(0.03 * total_train_steps)

    # training setup
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_torch",
        save_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        learning_rate=args.learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=num_warmup_steps,
        lr_scheduler_type="cosine",
        report_to=args.report_to,
        overwrite_output_dir=args.overwrite_output_dir,
        logging_dir=args.logging_dir,
        logging_strategy='steps',
        run_name=args.run_name,
        save_total_limit=1
    )

    response_template = LLM_MAPPING_DICT[args.model_name_or_path]['response_template']

    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    # The assistant answer is ignored during loss calculation
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_sft_dataset,
        eval_dataset=tokenized_validation_sft_dataset,
        peft_config=peft_config,
        dataset_text_field="sft",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        data_collator=collator
    )
    print('Begin training!')
    trainer.train()
    print(f'Training finished, saving to {args.output_dir}')

    output_dir = os.path.join(args.output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
