from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="augmented_prompt", label_key=None, apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data.get(input_key, [])
        if isinstance(chat, dict):
            chat = [chat]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label, data["augmented_prompt"], data["base_prompt"], data["rubric"]


class SelfPlayDataset(Dataset):
    """
    Dataset for Summary-PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        self.augmented_prompts = []
        self.base_prompts = []
        self.rubrics = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label, augmented_prompt, base_prompt, rubric = preprocess_data(data, input_template, input_key, label_key, apply_chat_template)
            self.prompts.append(prompt)
            self.labels.append(label)
            self.augmented_prompts.append(augmented_prompt)
            self.base_prompts.append(base_prompt)
            self.rubrics.append(rubric)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.labels[idx], self.augmented_prompts[idx], self.base_prompts[idx], self.rubrics[idx]
