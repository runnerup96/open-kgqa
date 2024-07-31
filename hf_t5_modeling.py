import logging
import os
import math
import sys
import json
import torch

from transformers import HfArgumentParser, T5ForConditionalGeneration, AutoTokenizer, Adafactor, \
    set_seed, get_cosine_schedule_with_warmup, EarlyStoppingCallback, Seq2SeqTrainingArguments

from transformers import GenerationConfig
import hf_t5_args
import text2query_t5_dataset
import training_utils
from sp_t5_trainer import SemanticParsingSeq2SeqTrainer


def main():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print('No GPU detected!')
        sys.exit()

    logger = logging.getLogger(__name__)

    parser = HfArgumentParser((hf_t5_args.ScriptArguments, Seq2SeqTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, model_max_length=script_args.max_seq_length, use_fast=False)
    # Prepare model & optimizer
    model = T5ForConditionalGeneration.from_pretrained(script_args.model_name_or_path)

    training_dataset_list, testing_dataset_list = [], []
    if script_args.sparql_dataset_name == "rubq":
        predicate_description_dict = json.load(
            open(script_args.path_to_predicate_description, 'r'))
        new_rubq_tokens = ['wdt:', 'skos:', 'wd:', 'ps:', 'pq:'] + list(predicate_description_dict.keys())
        tokenizer.add_tokens(new_rubq_tokens)
        model.resize_token_embeddings(len(tokenizer))

        kgqa_train_dataset_list = training_utils.format_rubq_dataset_to_kgqa_dataset(script_args.path_to_training_file)
        training_dataset_list = training_utils.form_t5_dataset(kgqa_train_dataset_list,
                                                           predicate_description_dict,
                                                           tokenizer,
                                                           input_max_length=script_args.max_seq_length,
                                                          output_max_length=script_args.max_output_length,
                                                           phase="train",
                                                            language=script_args.language,
                                                           try_one_batch=script_args.try_one_batch,
                                                           batch_size=training_args.per_device_train_batch_size)

        kgqa_test_dataset_list = training_utils.format_rubq_dataset_to_kgqa_dataset(script_args.path_to_testing_file)
        testing_dataset_list = training_utils.form_t5_dataset(kgqa_test_dataset_list,
                                                        predicate_description_dict,
                                                          tokenizer,
                                                          input_max_length=script_args.max_seq_length,
                                                          output_max_length=script_args.max_output_length,
                                                          phase="test",
                                                          language=script_args.language,
                                                          try_one_batch=script_args.try_one_batch,
                                                          batch_size=training_args.per_device_train_batch_size)

    train_dataset = text2query_t5_dataset.T5FinetuneDataset(training_dataset_list, tokenizer)
    test_dataset = text2query_t5_dataset.T5FinetuneDataset(testing_dataset_list, tokenizer)

    if script_args.try_one_batch:
        test_dataset = train_dataset
        testing_dataset_list = training_dataset_list



    optimizer = Adafactor(model.parameters(), lr=training_args.learning_rate,
                          scale_parameter=False, relative_step=False, clip_threshold=1.0,
                          warmup_init=False)
    batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    num_update_steps_per_epoch = len(train_dataset) // batch_size
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    total_train_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    print('My total train steps: ', total_train_steps)

    callbacks_list = []

    num_warmup_steps = int(0.1 * total_train_steps)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=200,
                                                    early_stopping_threshold=0.01)
    callbacks_list.append(early_stopping_callback)

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=total_train_steps)

    # prepare evaluation class
    evaluator = training_utils.Evaluator()

    # https://huggingface.co/docs/transformers/main_classes/text_generation
    my_generation_config = GenerationConfig()
    #

    my_generation_config.max_length = script_args.max_output_length
    my_generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    my_generation_config.eos_token_id = tokenizer.eos_token_id
    my_generation_config.pad_token_id = tokenizer.pad_token_id



    # my_generation_config.max_new_tokens = script_args.max_output_length
    #
    # my_generation_config.decoder_start_token_id = tokenizer.pad_token_id
    # # my_generation_config.bos_token_id = tokenizer.bos_token_id
    # my_generation_config.pad_token_id = tokenizer.pad_token_id
    #
    training_args.generation_config = my_generation_config


    trainer = SemanticParsingSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        eval_examples=testing_dataset_list,
        tokenizer=tokenizer,
        compute_metrics=evaluator.evaluate,
        optimizers=(optimizer, lr_scheduler),
        post_process_function=training_utils.model_post_processing_function,
        callbacks=callbacks_list
    )

    if training_args.do_train:
        checkpoint = None
        # means that we are training from last checkpoint, including the state of optimizer
        if script_args.phase == 'finetune':
            last_checkpoint = training_utils.get_last_checkpoint(script_args.model_name_or_path)

            if last_checkpoint is not None:
                checkpoint = last_checkpoint

                print(f'Starting from from {last_checkpoint}')

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        trainer.save_state()

    if training_args.do_eval and script_args.phase != 'pretrain':
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=script_args.max_seq_length,

                                   num_beams=training_args.generation_num_beams, metric_key_prefix="eval",
                                   output_save_dir=training_args.output_dir)

        if 'predictions' in metrics:
            output_dir = training_args.output_dir
            filename = os.path.basename(output_dir).split('.')[0]
            filename = f"{filename}_prediction.txt"
            save_path = os.path.join(output_dir, filename)
            with open(save_path, 'w') as f:
                for pred in metrics['predictions']:
                    f.write(f"{pred} \n")
            metrics.pop('predictions')

        metrics["eval_samples"] = len(test_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        res = trainer.predict(test_dataset, tokenizer=tokenizer)
        # Save the prediction files for spider evaluation
        prediction_list = []
        for pred_idx, pred_id in enumerate(res.predictions):
            prediction_list.append(pred_id)

        output_dir = training_args.output_dir
        filename = os.path.basename(script_args.path_to_testing_file).split('.')[0]
        filename = f"{filename}_prediction.txt"
        save_path = os.path.join(output_dir, filename)

        logger.info("Writing model predictions to txt file...")
        with open(save_path, 'w') as f:
            for line in prediction_list:
                f.write(f"{line}\n")


if __name__ == "__main__":
    main()
