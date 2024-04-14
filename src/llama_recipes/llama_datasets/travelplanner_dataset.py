# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/jfleg
# For download and preparation see: recipes/ft_datasets/grammar_dataset/grammar_dataset_process.ipynb


from datasets import load_dataset
from pathlib import Path

from torch.utils.data import Dataset
from transformers import LlamaTokenizerFast


class travelplanner(Dataset):
    def __init__(
        self,
        tokenizer,
        csv_name=None,
    ):

        try:
            self.dataset = load_dataset(
                "csv",
                data_files={"train": "travelplanner_train.csv"},  # "eval": "grammar_validation.csv"},
                delimiter=",",
            )
        except Exception as e:
            print("Loading of travelplanner dataset failed!")
            print(e)

        # self.dataset = load_dataset("wikihow", "all", data_dir="data/", split=type_path)
        # if num_samples:
        #    self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.tokenizer = tokenizer
        self.print_text = False  # print_text

    def __len__(self):
        return self.dataset["train"].shape[0]

    def convert_to_features(self, example_batch):

        # Create prompt and tokenize contexts and questions

        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch["text"]))

        input_ = example_batch["query"]
        target_ = example_batch["annotated_plan"]
        context_ = example_batch["reference_information"]

        prompt = f"""You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).                ***** Example Ends *****
                Given information: {context_}
                Query: {input_}
                Travel Plan:"""
        prompt_ids = self.tokenizer.encode(self.tokenizer.bos_token + prompt, add_special_tokens=False)
        label_ids = self.tokenizer.encode(target_ + self.tokenizer.eos_token, add_special_tokens=False)
        print(len(prompt_ids), len(label_ids))
        sample = {
            "input_ids": prompt_ids + label_ids,
            "attention_mask": [1] * len(prompt_ids + label_ids),
            "labels": [-100] * len(prompt_ids) + label_ids
        }

        return sample

    def __getitem__(self, index):
        return self.convert_to_features(self.dataset["train"][int(index)])


def get_dataset(
    dataset_config, tokenizer, csv_name=None
):
    """cover function for handling loading the working dataset"""
    """dataset loading"""
    if csv_name is None:
        currPath = Path.cwd() / "datasets_grammar" / "grammar_train.csv"
        print(f"Loading dataset {currPath}")
        csv_name = str(currPath)
    dataset = travelplanner(
        tokenizer=tokenizer,
        csv_name=csv_name,
    )

    return dataset

if(__name__ == "__main__"):
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    dataset = get_dataset(None, tokenizer, "travelplanner_train.csv")
    for(i, data) in enumerate(dataset):
        data