import sys
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import training_utils
from transformers import HfArgumentParser
import hf_llm_args
import json
import logging
import pickle
import os
from transformers import set_seed
from tqdm import tqdm
import text2query_llm_dataset
from lmm_mapping_constants import LLM_MAPPING_DICT
from torch.utils.data import DataLoader

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == 'cpu':
        print('No GPU detected!')
        sys.exit()

    logger = logging.getLogger(__name__)

    parser = HfArgumentParser(hf_llm_args.ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("}"),
        335  # 'Ġ}'
    ]

    if args.use_lora:
        model = AutoPeftModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                         torch_dtype=torch.float16,
                                                         device_map=device)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     torch_dtype=torch.float16,
                                                     device_map=device)
    model.generation_config.pad_token_ids = tokenizer.pad_token_id

    # read test data
    testing_sft_dataset = []

    # predicate_description_dict = json.load(
    #     open(args.path_to_predicate_description, 'r'))

    new_rubq_tokens = ['wdt:', 'skos:', 'wd:', 'ps:', 'pq:']
    tokenizer.add_tokens(new_rubq_tokens)
    model.resize_token_embeddings(len(tokenizer))

    testing_sft_dataset = json.load(open(args.path_to_testing_file, 'r'))

    if args.try_one_batch:
        testing_sft_dataset = testing_sft_dataset[:args.per_device_eval_batch_size]

    tokenized_test_sft_dataset = text2query_llm_dataset.LlmFinetuneDataset(sft_dataset=testing_sft_dataset,
                                                                           device=device, tokenizer=tokenizer,
                                                                           max_sft_length=args.max_seq_length)

    print(f'Total testing samples = {len(tokenized_test_sft_dataset)}')

    test_dataloader = DataLoader(tokenized_test_sft_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size)

    ids_list, prediction_list, scores_list = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            sample_id = batch['id']
            input_length = batch['input_ids'].shape[1]
            outputs = model.generate(input_ids=batch['input_ids'].to(device),
                                     attention_mask=batch['attention_mask'].to(device),
                                     max_new_tokens=args.max_new_tokens,
                                     num_beams=args.num_beams,
                                     eos_token_id=terminators,
                                     output_logits=True,
                                     return_dict_in_generate=True,
                                     pad_token_id=tokenizer.eos_token_id)

            generated_sequences = outputs["sequences"].cpu() if "cuda" in device else outputs["sequences"]

            entropy_scores = training_utils.maximum_entropy_confidence_score_method(generation_scores=outputs["logits"],
                                                                                    device=device)
            entropy_scores = training_utils.truncate_scores(generated_sequences=generated_sequences,
                                                            scores=entropy_scores,
                                                            tokenizer=tokenizer)
            max_entropy_scores = [max(score_list) for score_list in entropy_scores]
            scores_list += max_entropy_scores
            decoded_preds = tokenizer.batch_decode(generated_sequences[:, input_length:],
                                                   skip_special_tokens=True, clean_up_tokenization_spaces=False)
            predictions = [pred for pred in decoded_preds]
            prediction_list += predictions
            ids_list += sample_id

    print('Inference completed!')

    result_dict = dict()
    for id_, pred_query, score in zip(ids_list, prediction_list, scores_list):
        result_dict[id_] = {
            "query": pred_query,
            "score": score
        }

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(args.path_to_testing_file).split('.')[0]
    if args.try_one_batch:
        filename = f"{args.sparql_dataset_name}_{filename}_one_batch_inference_result.pkl"
    else:
        filename = f"{args.sparql_dataset_name}_{filename}_inference_result.pkl"
    save_path = os.path.join(output_dir, filename)
    logger.info("Writing model predictions to json file...")

    pickle.dump(result_dict, open(save_path, 'wb'))

    filename = filename.split('.')[0]

    if args.try_one_batch:
        filename = f"{filename}_one_batch_query_predictions.txt"
    else:
        filename = f"{filename}_query_predictions.txt"
    save_path = os.path.join(output_dir, filename)
    with open(save_path, 'w') as f:
        for query in prediction_list:
            f.write(f"{query}\n")

    # model_id = "/home/etutubalina/somov-od/llama3/Meta-Llama-3-8B-Instruct/hf_converted"
    #
    # pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16},
    #                                  device_map="auto")
    # print(pipeline("Hey how are you doing today? Answer in Russian"))

    # # Example of using the fine-tuned model
    # messages = [
    #     {"role": "system", "content": "You are a helpful medical chatbot."},
    #     {"role": "user", "content": "I have a headache and fever."},
    # ]
    #
    # input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
    #     model.device)
    # outputs = model.generate(input_ids, max_new_tokens=256, do_sample=True, temperature=0.7)
    # response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    # print(response)
