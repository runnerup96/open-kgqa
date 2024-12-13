from torch.utils.data import Dataset


class LlmFinetuneDataset(Dataset):
    def __init__(self, sft_dataset, device, tokenizer, max_sft_length):
        self.sft_dataset = sft_dataset
        self.device = device
        self.tokenizer = tokenizer
        self.max_sft_length = max_sft_length

    def __len__(self):
        return len(self.sft_dataset)

    def __getitem__(self, idx):
        sample_id = self.sft_dataset[idx]
        id_ = sample_id['id']
        sft = sample_id['sft']
        tokenized_sft = self.tokenizer(sft, max_length=self.max_sft_length,
                                  truncation=True, padding='max_length', add_special_tokens=False,
                                  return_tensors='pt')

        input_ids, attention_mask = tokenized_sft['input_ids'][0].to(self.device), \
                                    tokenized_sft['attention_mask'][0].to(self.device)

        return {"id": id_, "input_ids": input_ids, "attention_mask": attention_mask}