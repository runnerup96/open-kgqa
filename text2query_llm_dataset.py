from torch.utils.data import Dataset


class LlmFinetuneDataset(Dataset):
    def __init__(self, sft_dataset, device):
        self.sft_dataset = sft_dataset
        self.device = device

    def __len__(self):
        return len(self.sft_dataset)

    def __getitem__(self, idx):
        sample_id = self.sft_dataset[idx]
        id_ = sample_id['id']
        input_ids, attention_mask = sample_id['tokenized_prompt']['input_ids'][0].to(self.device), \
                                    sample_id['tokenized_prompt']['attention_mask'][0].to(self.device)

        return {"id": id_, "input_ids": input_ids, "attention_mask": attention_mask}