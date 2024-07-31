import torch
from torch.utils.data import Dataset


class T5FinetuneDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.tokenizer = tokenizer

        self.max_input_len = 0
        self.max_output_len = 0
        self.samples = []

        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

        for sample in samples:
            id_ = sample['id']
            input_ids = sample['source_tokens']
            if sample['target_tokens']:
                output_ids = sample['target_tokens'] + [self.eos_token_id]
                self.max_output_len = max(self.max_output_len, len(output_ids))
            else:
                output_ids = None

            self.max_input_len = max(self.max_input_len, len(input_ids))
            self.samples.append((id_, input_ids, output_ids))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        id_, tokenized_input, tokenized_output = self.samples[index]

        input_npad = self.max_input_len - len(tokenized_input)
        attention_mask = [1] * len(tokenized_input) + [0] * input_npad
        input_ids = torch.LongTensor(tokenized_input + input_npad * [self.pad_token_id])

        if tokenized_output:
            output_npad = self.max_output_len - len(tokenized_output)

            labels = torch.LongTensor(tokenized_output + output_npad * [-100])
        else:
            labels = []

        return {'id': id_,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                }
