from transformers import (
  AutoTokenizer, 
  AutoConfig, 
  AdamW, 
  get_scheduler, 
  TrainingArguments, 
  Trainer, 
  DataCollatorWithPadding,
  AutoModelForSequenceClassification,
  GenerationConfig
) 
from trl import AutoModelForCausalLMWithValueHead
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate
import torch.nn as nn
import torch
import numpy as np
import os
import tqdm
from peft import PeftModel, PeftConfig
import huggingface_hub

def setup():
  huggingface_hub.login("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")
  model_name = "justshao/llama2-7b-with-confidence"
  lora_model_name = "/data/chenran/llama_data_collect/value_head_training/final-checkpoint"
  device = "cuda" if torch.cuda.is_available() else "cpu"
  torch.set_default_device(device)
  
  tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                            trust_remote_code=True, 
                                            padding_side="left")

  num_labels = 2
  model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    torch_dtype=torch.bfloat16
  )

  if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
  
  #print(model)
  model = PeftModel.from_pretrained(model, lora_model_name)
  model.base_model.resize_token_embeddings(len(tokenizer))
  model.base_model.config.pad_token_id = tokenizer.pad_token_id

  model.base_model.push_to_hub("justshao/llama2-7b-with-confidence")
  model.base_model.config.push_to_hub("justshao/llama2-7b-with-confidence")
  model.push_to_hub("justshao/llama2-7b-with-confidence")
  tokenizer.push_to_hub("justshao/llama2-7b-with-confidence")

def main():
  huggingface_hub.login("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")
  division = 'test'
  input_data_dir = "/data/chenran/llama_data_collect/value_head_training/llama_data/arithmetic/" + division
  with_predicted_IK_dir ="/data/chenran/llama_data_collect/value_head_training/llama_data/arithmetic" + division + "_with_IK"
  model_name = "justshao/llama2-test"
  lora_model_name = "/data/chenran/llama_data_collect/value_head_training/final-checkpoint_5shot"
  device = "cuda" if torch.cuda.is_available() else "cpu"
  torch.set_default_device(device)
  
  tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                            trust_remote_code=True, 
                                            padding_side="left")

  num_labels = 2
  model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    torch_dtype=torch.bfloat16
  )
  
  #print(model)
  model = PeftModel.from_pretrained(model, lora_model_name)
  #model.push_to_hub(model_name)
  
  example_prompt = "Question: What is 4721 + 18397?\n\nAnswer: 23118\nQuestion: What is 13656 + 45155?\n\nAnswer: 58811\nQuestion: What is 4785 - 8417?\n\nAnswer: -3632\nQuestion: What is 6 + 0?\n\nAnswer: 6\nQuestion: What is 7486 + 1057?\n\nAnswer: 8543"
  def tokenize_function(examples):
    full_prompts = list(example_prompt + question for question in examples["context"])
    return tokenizer(
        full_prompts, 
        return_attention_mask=True
        )
  
  
  raw_dataset = load_from_disk(input_data_dir)
  tokenzied_dataset = raw_dataset.map(tokenize_function, batched=True)
  tokenzied_dataset = tokenzied_dataset.remove_columns(raw_dataset.column_names)
  print(raw_dataset.column_names)

  data_collator  = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
  train_dataloader = DataLoader(
    tokenzied_dataset, shuffle=False, batch_size=8, collate_fn=data_collator
  )

  IK_pred = []
  for batch in tqdm.tqdm(train_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
      outputs = model(**batch, return_dict=True)
    #output keys: [loss, logits, past_key_values]
    predicted_ik = nn.functional.sigmoid(outputs['logits'][:, 1]).tolist()
    IK_pred.extend(predicted_ik)
  
  dataset_with_predictde_IK = raw_dataset.add_column("predicted IK", IK_pred)
  dataset_with_predictde_IK.save_to_disk(with_predicted_IK_dir)


if __name__ == "__main__":
  main()
