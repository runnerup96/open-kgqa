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
import training_utils
import hf_llm_args
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.utils import logging
import text2query_llm_dataset

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print('No GPU detected!')
        sys.exit()

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
    model.resize_token_embeddings(len(tokenizer))
    print('Loaded model!')

    # read data
    training_sft_dataset, testing_sft_dataset = [], []
    if args.sparql_dataset_name == "rubq":
        predicate_description_dict = json.load(
            open(args.path_to_predicate_description, 'r'))
        new_rubq_tokens = ['wdt:', 'skos:', 'wd:', 'ps:', 'pq:'] + list(predicate_description_dict.keys())
        graph_entities_str = "|".join(new_rubq_tokens)
        tokenizer.add_tokens(new_rubq_tokens)

        training_sft_dataset = training_utils.read_rubq_sft_file(args.path_to_training_file)
        validation_sft_dataset = training_utils.read_rubq_sft_file(args.path_to_training_file)


        # train_kgqa_dataset = training_utils.format_rubq_dataset_to_kgqa_dataset(args.path_to_training_file)
        # training_sft_dataset = training_utils.form_sft_dataset_llm(train_kgqa_dataset,
        #                                                            graph_entities_str,
        #                                                            tokenizer,
        #                                                            max_length=args.max_seq_length,
        #                                                            phase="train",
        #                                                            try_one_batch=args.try_one_batch,
        #                                                            batch_size=args.per_device_train_batch_size,
        #                                                            language='ru')
        #
        # test_kgqa_dataset = training_utils.format_rubq_dataset_to_kgqa_dataset(args.path_to_testing_file)
        # testing_sft_dataset = training_utils.form_sft_dataset_llm(test_kgqa_dataset,
        #                                                           graph_entities_str,
        #                                                           tokenizer,
        #                                                           max_length=args.max_seq_length,
        #                                                           phase="train",
        #                                                           try_one_batch=args.try_one_batch,
        #                                                           batch_size=args.per_device_train_batch_size,
        #                                                           language='ru')

    elif args.sparql_dataset_name == "salute":
        predicate_vocab_list = json.load(
            open(args.path_to_predicate_description, 'r'))
        new_salute_tokens = predicate_vocab_list

        graph_entities_str = "|".join(new_salute_tokens)
        tokenizer.add_tokens(new_salute_tokens)
        model.resize_token_embeddings(len(tokenizer))

        kgqa_train_dataset_list = training_utils.format_salute_to_kgqa_dataset(args.path_to_training_file)
        training_sft_dataset = training_utils.form_sft_dataset_llm(kgqa_train_dataset_list,
                                                               graph_entities_str,
                                                               tokenizer,
                                                               max_length=args.max_seq_length,
                                                               phase="train",
                                                               language='ru',
                                                               try_one_batch=args.try_one_batch,
                                                               batch_size=args.per_device_train_batch_size)

        kgqa_test_dataset_list = training_utils.format_salute_to_kgqa_dataset(args.path_to_testing_file)
        testing_sft_dataset = training_utils.form_sft_dataset_llm(kgqa_test_dataset_list,
                                                              graph_entities_str,
                                                              tokenizer,
                                                              max_length=args.max_seq_length,
                                                              phase="train",
                                                              language='ru',
                                                              try_one_batch=args.try_one_batch,
                                                              batch_size=args.per_device_eval_batch_size)

    tokenized_train_sft_dataset = text2query_llm_dataset.LlmFinetuneDataset(sft_dataset=training_sft_dataset,
                                                                            device=device, tokenizer=tokenizer,
                                                                           max_sft_length=args.max_seq_length)
    tokenized_test_sft_dataset = text2query_llm_dataset.LlmFinetuneDataset(sft_dataset=testing_sft_dataset,
                                                                           device=device, tokenizer=tokenizer,
                                                                           max_sft_length=args.max_seq_length)

    if args.try_one_batch:
        tokenized_test_sft_dataset = tokenized_train_sft_dataset

    print('Training samples total size: ', len(tokenized_train_sft_dataset))

    # following https://arxiv.org/pdf/2305.14314
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=16,
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

    response_template = f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    if args.model_name_or_path.contains("Qwen2.5-Coder"):
        response_template = "<|im_start|>assistant\n"


    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    # The assistant answer is ignored during loss calculation
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # import text2sql_dataset
    # from torch.utils.data import DataLoader
    # ds_train = text2sql_dataset.Text2SQLDataset(training_sft_dataset, tokenizer, args.max_seq_length, 'cuda')
    # dl_train = DataLoader(ds_train, batch_size=args.per_device_train_batch_size, collate_fn=collator)
    #
    # tokenizer.padding_side = "left"
    # ds_test = text2sql_dataset.Text2SQLDataset(testing_sft_dataset, tokenizer, args.max_seq_length, 'cuda')
    # dl_test = DataLoader(ds_test, batch_size=args.per_device_train_batch_size)

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_sft_dataset,
        eval_dataset=tokenized_test_sft_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        data_collator=collator,
    )
    print('Begin training!')
    trainer.train()
    print(f'Training finished, saving to {args.output_dir}')

    output_dir = os.path.join(args.output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
