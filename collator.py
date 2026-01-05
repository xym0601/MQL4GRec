import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)

        labels = self.tokenizer(label_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100

        interest_texts = [d.get("interest", "") for d in batch]

        interest_inputs = self.tokenizer(
            interest_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,  # 或者你自己设短一点
            truncation=True,
            return_attention_mask=True
        )

        inputs["interest_input_ids"] = interest_inputs["input_ids"]
        inputs["interest_attention_mask"] = interest_inputs["attention_mask"]

        # print(inputs.input_ids[0])
        # print(inputs.labels[0])
        
        if not hasattr(self, "_debug_printed"):
            self._debug_printed = 0
        if self._debug_printed < 3:
            print("[CHECK collator] interest_input_ids shape:",
                inputs["interest_input_ids"].shape,
                "interest_attention_mask shape:",
                inputs["interest_attention_mask"].shape,
                flush=True)
            print("[CHECK collator] first interest ids:",
                inputs["interest_input_ids"][0][:20].tolist(),
                flush=True)
            self._debug_printed += 1

        return inputs


    
class TestCollatorSave(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.prefix_token = vars(args).get('prefix_token', '')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
            
        if isinstance(self.tokenizer, LlamaTokenizer):
            # Allow batched inference
            self.tokenizer.padding_side = "left"

    def __call__(self, batch):

        input_texts = [d["input_ids"] + self.prefix_token for d in batch]
        targets = [d["labels"] for d in batch]
        users = [d["label"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        
        interest_texts = [d.get("interest", "") for d in batch]

        interest_inputs = self.tokenizer(
            interest_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,  # 或者你自己设短一点
            truncation=True,
            return_attention_mask=True
        )

        inputs["interest_input_ids"] = interest_inputs["input_ids"]
        inputs["interest_attention_mask"] = interest_inputs["attention_mask"]


        return (inputs, targets, users)
    



