import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
from typing import List

import torch
import transformers
import torch.nn as nn
from swanlab.integration.huggingface import SwanLabCallback
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.models.t5.modeling_t5 import T5LayerCrossAttention, T5LayerNorm
from transformers.modeling_outputs import BaseModelOutput
from utils import *
from collator import Collator

from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5LayerCrossAttention, T5LayerNorm
from transformers.modeling_outputs import BaseModelOutput
import torch

class InterestCrossAttnT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.interest_cross_attn = T5LayerCrossAttention(config)
        self.interest_ln = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        interest_input_ids=None,
        interest_attention_mask=None,
        encoder_outputs=None,        # ✅ 关键：接住 generate 传进来的 encoder_outputs
        past_key_values=None,
        use_cache=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ---------------------------------------------------------
        # 1) encoder：两种情况
        #    A) generate 后续 step：encoder_outputs 已经有了 -> 直接用
        #    B) 训练/推理第一步：encoder_outputs 没有 -> 用 input_ids 编码
        # ---------------------------------------------------------
        if encoder_outputs is None:
            if input_ids is None:
                raise ValueError("Need input_ids when encoder_outputs is None.")
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            # 兼容 encoder_outputs 可能是 tuple / BaseModelOutput
            if isinstance(encoder_outputs, tuple):
                encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0])
            elif not hasattr(encoder_outputs, "last_hidden_state"):
                encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)

        hist_hidden = encoder_outputs.last_hidden_state  # [B, Lh, D]

        # ---------------------------------------------------------
        # 2) interest 融合：只在“有 interest 输入”时做
        #    训练/推理第一步：一般会有 interest
        #    generate 后续 step：通常 interest 不再传，但 encoder_outputs 已经融合过
        # ---------------------------------------------------------
        if interest_input_ids is not None and interest_attention_mask is not None:
            interest_outputs = self.encoder(
                input_ids=interest_input_ids,
                attention_mask=interest_attention_mask,
                return_dict=True,
            )
            int_hidden = interest_outputs.last_hidden_state  # [B, Li, D]

            # additive mask: [B, 1, 1, Li], pad -> -1e9
            int_mask = (1.0 - interest_attention_mask[:, None, None, :].float()) * -1e9

            fused = self.interest_cross_attn(
                hidden_states=hist_hidden,
                key_value_states=int_hidden,
                attention_mask=int_mask,
                position_bias=None,
                layer_head_mask=None,
                past_key_value=None,
                use_cache=False,     
                query_length = hist_hidden.size(1),      
                output_attentions=False,
            )[0]

            hist_hidden = self.interest_ln(hist_hidden + fused)

            # 把融合后的 hidden 写回 encoder_outputs
            encoder_outputs = BaseModelOutput(last_hidden_state=hist_hidden)

        # ---------------------------------------------------------
        # 3) decode/loss：关键点
        #    generate 后续 step input_ids 会是 None，所以这里不要依赖 input_ids
        #    把 encoder_outputs 传给 super.forward 即可
        # ---------------------------------------------------------
        return super().forward(
            input_ids=input_ids,  # 训练/第一步可以传；后续 step 可能为 None，也没关系
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs
        )

    # ✅ 让 generate 正确把 encoder_outputs/past_key_values 往后传
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        encoder_outputs=None,
        **kwargs
    ):
        return {
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    # ✅ beam search 需要（更稳）
    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return None
        reordered = ()
        for layer_past in past_key_values:
            # layer_past: (self_attn_k, self_attn_v, cross_attn_k, cross_attn_v) for T5
            reordered_layer_past = tuple(
                p.index_select(0, beam_idx.to(p.device)) if p is not None else None
                for p in layer_past
            )
            reordered += (reordered_layer_past,)
        return reordered
        
def train(args):

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)

    if args.load_model_name is not None:
        config = T5Config.from_pretrained(args.load_model_name)
        tokenizer = T5Tokenizer.from_pretrained(
            args.load_model_name,
            model_max_length=512,
        )
    else:
        config = T5Config.from_pretrained(args.base_model)
        tokenizer = T5Tokenizer.from_pretrained(
            args.base_model,
            model_max_length=512,
        )
    if local_rank == 0:
        print("d_model:", config.d_model,
            "num_heads:", getattr(config, "num_heads", None),
            "num_layers:", config.num_layers)
        
    if args.tie_encoder_decoder:
        config.tie_encoder_decoder = True
        
    args.deepspeed = None
    gradient_checkpointing= False
    
    tasks = args.tasks.split(",")
    soft_prompts = {}
    for i, task in enumerate(tasks):
        if task == 'fgfusionseqrec':
            token_ids = list(range(100 * (i + 1), 100 * (i + 1) + args.prompt_num * 4))
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            prompt = []
            for k in range(4):
                p = ''.join(tokens[args.prompt_num * k : args.prompt_num * (k+1)])
                prompt.append(p)
            
        else:
            token_ids = list(range(100 * (i + 1), 100 * (i + 1) + args.prompt_num))
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            prompt = ''.join(tokens)
            
        if local_rank == 0:
            print(tokens)
            
        soft_prompts[task] = prompt
        
    args.soft_prompts = soft_prompts

    train_data, valid_data = load_datasets(args)
    
    add_num = 0
    # item / image tokens
    for dataset in train_data.datasets:
        add_num += tokenizer.add_tokens(dataset.get_new_tokens())

    # user interest tokens (ROBUST for T5/SentencePiece)
    interest_path = os.path.join(args.data_path, args.dataset, "User_Interest_IDs.json")
    print("[CHECK] interest_path:", interest_path, "exists:", os.path.exists(interest_path), flush=True)

    if local_rank == 0:
        print("[CHECK] interest_path =", interest_path, "exists =", os.path.exists(interest_path), flush=True)

    if os.path.exists(interest_path):
        import json
        with open(interest_path, "r") as f:
            user_interest = json.load(f)
            print("[CHECK] #users in interest:", len(user_interest), flush=True)
            first_uid = next(iter(user_interest.keys()))
            print("[CHECK] sample uid:", first_uid, "tokens:", user_interest[first_uid], flush=True)

        # 清洗 token（非常关键：去掉不可见空格/换行）
        interest_tokens = sorted({str(t).strip() for toks in user_interest.values() for t in toks})

        if local_rank == 0:
            print("[CHECK] #interest_tokens =", len(interest_tokens), flush=True)
            print("[CHECK] sample tokens repr:", [repr(x) for x in interest_tokens[:5]], flush=True)

        # 关键：作为 additional_special_tokens 加入，保证不拆分
        interest_add_num = tokenizer.add_special_tokens({"additional_special_tokens": interest_tokens})
        add_num += interest_add_num

        if local_rank == 0:
            print(f"[CHECK] added interest special tokens: {interest_add_num}", flush=True)
            for t in ["<o_70>", "<p_131>", "<q_131>"]:
                print("[CHECK vocab]", t, t in tokenizer.get_vocab(), flush=True)
            print("[CHECK tokenize spaced]", tokenizer.tokenize("<o_70> <p_131> <q_131>"), flush=True)


    collator = Collator(args, tokenizer)
    
    # if args.load_model_name is not None:
    #     model = T5ForConditionalGeneration.from_pretrained(args.load_model_name, config=config)
    # else:
    #     model = T5ForConditionalGeneration(config)
    if args.load_model_name is not None:
        model = InterestCrossAttnT5.from_pretrained(args.load_model_name, config=config)
    else:
        model = InterestCrossAttnT5(config)
            
    # ===== DEBUG: check interest tokenization =====
    if local_rank == 0:
        test_interest = "<o_70> <p_131> <q_131>"
        print("\n[DEBUG] tokenizer.tokenize(test_interest):")
        print(tokenizer.tokenize(test_interest))

        enc = tokenizer(test_interest, add_special_tokens=False)
        print("[DEBUG] input_ids:", enc["input_ids"])
        print("[DEBUG] back to tokens:",
            tokenizer.convert_ids_to_tokens(enc["input_ids"]))
        print("============================================\n")
    # ===== END DEBUG =====
    if local_rank == 0:
        for t in ["<o_70>", "<p_131>", "<q_131>"]:
            print("[CHECK vocab]", t, t in tokenizer.get_vocab(), flush=True)

    model.resize_token_embeddings(len(tokenizer))

    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print('train sequence')
        for dataset in train_data.datasets:
            print(dataset[100])
        print('train sequence')
        print(valid_data[100])
        print(model)
        print_trainable_parameters(model)


    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    early_stop = EarlyStoppingCallback(early_stopping_patience=args.patient)
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            fp16=args.fp16,
            # bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            gradient_checkpointing=args.gradient_checkpointing,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            # deepspeed=args.deepspeed,
            ddp_find_unused_parameters=True if ddp else None,
            report_to=None,
            eval_delay= 1 if args.save_and_eval_strategy=="epoch" else 2000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[early_stop, SwanLabCallback(project="MQL4GRec"),],
    )
    model.config.use_cache = False


    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)
