import logging
import os
import pickle
import sys
from tqdm import tqdm
import torch
from transformers import HfArgumentParser, T5ForConditionalGeneration, AutoTokenizer, AutoConfig, Adafactor, \
    set_seed, Seq2SeqTrainingArguments, GenerationConfig
from torch.utils.data import DataLoader
import hf_t5_args
import text2query_t5_dataset
import json
import training_utils

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print('No GPU detected!')
        sys.exit()

    logger = logging.getLogger(__name__)

    parser = HfArgumentParser((hf_t5_args.ScriptArguments, Seq2SeqTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path,
                                              model_max_length=script_args.max_seq_length, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(script_args.model_name_or_path).to(device)
    model.resize_token_embeddings(len(tokenizer))

    testing_dataset_list = []
    if script_args.sparql_dataset_name == "rubq":
        predicate_description_dict = json.load(
            open(script_args.path_to_predicate_description, 'r'))
        new_rubq_tokens = ['wdt:', 'skos:', 'wd:', 'ps:', 'pq:'] + list(predicate_description_dict.keys())
        graph_entities_str = "|".join(new_rubq_tokens)
        tokenizer.add_tokens(new_rubq_tokens)

        kgqa_test_dataset_list = training_utils.format_rubq_dataset_to_kgqa_dataset(script_args.path_to_testing_file)
        testing_dataset_list = training_utils.form_t5_dataset(kgqa_test_dataset_list,
                                                              graph_entities_str,
                                                              tokenizer,
                                                              input_max_length=script_args.max_seq_length,
                                                              output_max_length=script_args.max_output_length,
                                                              phase="test",
                                                              language=script_args.language,
                                                              try_one_batch=script_args.try_one_batch,
                                                              batch_size=training_args.per_device_eval_batch_size)

    elif script_args.sparql_dataset_name == "salute":
        predicate_vocab_list = json.load(
            open(script_args.path_to_predicate_description, 'r'))
        new_salute_tokens = ['wdt:', 'skos:', 'wd:', 'ps:', 'pq:'] + predicate_vocab_list

        graph_entities_str = "|".join(new_salute_tokens)
        tokenizer.add_tokens(new_salute_tokens)
        model.resize_token_embeddings(len(tokenizer))

        kgqa_test_dataset_list = training_utils.format_salute_to_kgqa_dataset(script_args.path_to_testing_file)
        testing_dataset_list = training_utils.form_t5_dataset(kgqa_test_dataset_list,
                                                              graph_entities_str,
                                                              tokenizer,
                                                              input_max_length=script_args.max_seq_length,
                                                              output_max_length=script_args.max_output_length,
                                                              phase="test",
                                                              language='ru',
                                                              try_one_batch=script_args.try_one_batch,
                                                              batch_size=training_args.per_device_train_batch_size)


    test_dataset = text2query_t5_dataset.T5FinetuneDataset(testing_dataset_list, tokenizer)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=training_args.eval_batch_size)


    # TODO: зачем оно тут, если не используется?
    my_generation_config = GenerationConfig()
    my_generation_config.max_length = script_args.max_output_length
    my_generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    my_generation_config.eos_token_id = tokenizer.eos_token_id
    my_generation_config.pad_token_id = tokenizer.pad_token_id

    ids_list = []
    prediction_list = []
    scores_list = []
    hidden_scores = []
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        ids = batch['id']
        source_ids = batch["input_ids"].to(device)
        outputs = model.generate(input_ids=source_ids, max_length=script_args.max_seq_length,
                                 num_beams=script_args.num_beams,
                                 output_scores=True, return_dict_in_generate=True)

        generated_sequences = outputs["sequences"].cpu() if "cuda" in device else outputs["sequences"]

        entropy_scores = training_utils.maximum_entropy_confidence_score_method(generation_scores=outputs["scores"],
                                                                                device=device)
        entropy_scores = training_utils.truncate_scores(generated_sequences=generated_sequences,
                                                        scores=entropy_scores,
                                                        tokenizer=tokenizer)
        max_entropy_scores = [max(score_list) for score_list in entropy_scores]
        scores_list += max_entropy_scores

        decoded_preds = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        ids_list += ids
        predictions = [training_utils.generated_query_simple_processor(pred) for pred in decoded_preds]
        prediction_list += predictions

        batch_logits = torch.stack(outputs["scores"], dim=1)
        for sample_idx in range(len(batch_logits)):
            sequence_scores = batch_logits[sample_idx, :, :].cpu()
            hidden_scores.append(sequence_scores)

    result_dict = dict()
    for id_, pred_query, score, hiddens in zip(ids_list, prediction_list, scores_list, hidden_scores):
        result_dict[id_.item()] = {
            "query": pred_query,
            "score": score
            # "hidden_states": hiddens
        }

    output_dir = training_args.output_dir
    filename = os.path.basename(script_args.path_to_testing_file).split('.')[0]
    if script_args.try_one_batch:
        filename = f"{script_args.sparql_dataset_name}_{filename}_one_batch_inference_result.pkl"
    else:
        filename = f"{script_args.sparql_dataset_name}_{filename}_inference_result.pkl"
    save_path = os.path.join(output_dir, filename)
    logger.info("Writing model predictions to json file...")

    pickle.dump(result_dict, open(save_path, 'wb'))
