import pandas as pd
from transformers import EvalPrediction, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import EvalLoopOutput
from datasets import Dataset
import os
import re
import torch
import evaluate
import random
import json
from tqdm import tqdm
import preprocessing_utils


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


class Evaluator:
    def __init__(self):
        self.exact_match = evaluate.load(os.path.join(os.environ['PROJECT_PATH'], 'exact_match.py'))

    def evaluate(self, p: EvalPrediction):
        metrics_dict = dict()
        exact_match_metric = self.exact_match.compute(predictions=p.predictions, references=p.label_ids,
                                                      ignore_case=True, ignore_punctuation=True, regexes_to_ignore=' ')
        metrics_dict.update(exact_match_metric)
        return metrics_dict



def generated_query_simple_processor(query):
    query = query.replace('<extra_id_0>', '')
    query = query.strip()
    return query


def original_query_simple_processor(query):
    # pass
    return query


def model_post_processing_function(examples: list, outputs: EvalLoopOutput, tokenizer):
    # Decode the predicted tokens.
    preds = outputs.predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    predictions = [generated_query_simple_processor(pred) for pred in decoded_preds]
    raw_references = [sample['target'] for sample in examples]

    return EvalPrediction(predictions=predictions, label_ids=raw_references)


# TODO: нужно унифицировать чтение и создание промта для обоих моделей
#       датасет должен просто брать из подготолвленных токенов


def format_salute_to_kgqa_dataset(dataset_path):
    data = json.load(open(dataset_path, 'r'))
    result_list = list()
    for idx, sample in enumerate(data):
        sample_id = idx
        ru_question = sample['question']
        sparql = sample['masked_query']

        sample_dict = {
            "id": sample_id,
            "ru_question": ru_question,
            "sparql": sparql,
        }
        result_list.append(sample_dict)
    random.shuffle(result_list)
    return result_list


def format_rubq_dataset_to_kgqa_dataset(dataset_path):
    data = json.load(open(dataset_path, 'r'))
    result_list = list()
    for sample in data:
        en_question = sample['question_eng']
        ru_question = sample['question_text']
        sparql = sample['query']
        sample_id = sample['uid']
        answers = sample['answers']
        expected_answer_id_list = []
        for ans in answers:
            entity_id = ans['value'].split('/')[-1]
            expected_answer_id_list.append(entity_id)
        if sparql:
            sample_dict = {
                "id": sample_id,
                "ru_question": ru_question,
                "en_question": en_question,
                "sparql": sparql,
                "answer_ids": expected_answer_id_list
            }
            result_list.append(sample_dict)
    random.shuffle(result_list)
    return result_list


def form_sft_dataset_llm(kgqa_dataset_list, graph_entities_str, tokenizer, phase='train',
                         max_length=1024,
                         try_one_batch=False, batch_size=4, language=None):
    sft_dataset_list = []
    for sample in tqdm(kgqa_dataset_list):
        sample_id = sample["id"]
        if language:
            if language == 'ru':
                input_lang_key = 'ru_question'
                instruction = RU_INSTRUCTION
            else:
                instruction = EN_INSTRUCTION
                input_lang_key = 'en_question'
        else:
            input_lang_key = 'question'
            instruction = RU_INSTRUCTION
        input_text = sample[input_lang_key]
        sparql = sample["sparql"]

        user_task = f"QUESTION: {input_text}\n" \
                    f"GRAPH ENTITIES: {graph_entities_str}\n"
        preprocessed_sparql = preprocessing_utils.preprocess_sparql(sparql)
        masked_sparql = preprocessing_utils.form_masked_query(preprocessed_sparql)['masked_query']

        if phase == 'train':

            chat = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_task},
                {"role": "assistant", "content": masked_sparql}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        else:
            chat = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_task}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        tokenized_prompts = tokenizer(formatted_prompt, max_length=max_length,
                                      truncation=True, padding='max_length', add_special_tokens=False,
                                      return_tensors='pt')
        sft_dataset_list.append(
            {'id': sample_id, 'tokenized_prompt': tokenized_prompts, "masked_sparql": masked_sparql})
    if try_one_batch:
        sft_dataset_list = sft_dataset_list[:batch_size]
    return sft_dataset_list


def form_t5_dataset(kgqa_dataset_list, graph_entities_str, tokenizer,
                    phase='train', input_max_length=512,
                    output_max_length=256, try_one_batch=False, batch_size=4, language=None):
    dataset_list = []
    for sample in tqdm(kgqa_dataset_list):
        sample_id = sample["id"]
        if language:
            if language == 'ru':
                input_lang_key = 'ru_question'
            else:
                input_lang_key = 'en_question'
        else:
            input_lang_key = 'question'
        input_text = sample[input_lang_key]

        sparql = sample["sparql"]

        preprocessed_sparql = preprocessing_utils.preprocess_sparql(sparql)
        masked_sparql = preprocessing_utils.form_masked_query(preprocessed_sparql)

        if "fred" in tokenizer.name_or_path.lower():
            formatted_source = '<SC6>Человек: ' + input_text + " | " + graph_entities_str + '<extra_id_0>'
            # TODO: At first predict predicates, then the whole query
            # as in RESDSQL - https://arxiv.org/abs/2302.05965
            formatted_target = '<extra_id_0>' + masked_sparql['masked_query']
        else:
            formatted_source = input_text + " | " + graph_entities_str
            formatted_target = masked_sparql['masked_query']

        source_tokens = tokenizer.encode(formatted_source, add_special_tokens=False, truncation=True,
                                         max_length=input_max_length)

        target_tokens = tokenizer.encode(formatted_target, add_special_tokens=False, truncation=True,
                                         max_length=output_max_length)

        dataset_list.append({'id': sample_id,
                             'source_tokens': source_tokens,
                             'target_tokens': target_tokens,
                             'source': input_text,
                             'target': masked_sparql['masked_query']})
    if try_one_batch:
        dataset_list = dataset_list[:batch_size]
    return dataset_list


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


class TrainingStopCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def __init__(self, steps):
        self.total_training_steps = steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == self.total_training_steps:
            control.should_training_stop = True


def maximum_entropy_confidence_score_method(generation_scores, device):
    # TODO: Work with beans to samples ratio here
    logits = torch.stack(generation_scores, dim=1)[:: 1]
    logits = logits.cpu() if "cuda" in device else logits
    probs = torch.softmax(logits, dim=2).float()
    log_probs = torch.log_softmax(logits, dim=2).float()
    entropies = (torch.sum(probs * log_probs, axis=2) * (-1)).numpy()

    return entropies


def truncate_scores(generated_sequences, scores, tokenizer):
    scores_list = []
    for idx in range(len(generated_sequences)):
        pred_tensor = generated_sequences[idx][1:]
        scores_truncated = scores[idx].tolist()

        # Truncate the prediction at the end-of-sequence token, if present.
        if tokenizer.eos_token_id in pred_tensor:
            pred_eos_idx = torch.nonzero(pred_tensor == tokenizer.eos_token_id)[0].item()
            scores_truncated = scores_truncated[: pred_eos_idx + 1]

        scores_list.append(scores_truncated)

    return scores_list


if __name__ == "__main__":
    import os
    dataset_path = '/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/RuBQ_2.0_dev.json'
    pred_map = "/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/rubq_predicate_mapping.json"

    from transformers import AutoTokenizer

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              # use_auth_token=True,
                                              token=os.environ['hf_token'],
                                              force_download=True,
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    predicate_description_dict = json.load(
        open(pred_map, 'r'))
    new_rubq_tokens = ['wdt:', 'skos:', 'wd:', 'ps:', 'pq:'] + list(predicate_description_dict.keys())
    tokenizer.add_tokens(new_rubq_tokens)

    train_kgqa_dataset = format_rubq_dataset_to_kgqa_dataset(dataset_path)
    training_sft_dataset = form_sft_dataset_llm(train_kgqa_dataset,
                                                predicate_description_dict,
                                                tokenizer,
                                                max_length=1024,
                                                phase="train",
                                                try_one_batch=False,
                                                batch_size=8,
                                                language='ru')
