from torch.utils.data import Dataset
import torch
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, context_size, stride):
        self.input_ids = []
        self.output_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - context_size, stride):
            input_chunk = token_ids[i:i + context_size]
            output_chunk = token_ids[i + 1:i + 1 + context_size]
            self.input_ids.append(torch.tensor(input_chunk))
            self.output_ids.append(torch.tensor(output_chunk))

    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]

    def __len__(self):
        return len(self.input_ids)