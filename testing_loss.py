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
  division = 'val'
  dataset_dir = "/data/chenran/llama_data_collect/value_head_training/llama_data/" + division + "/MMLU_5shot_postprocess/all"
  dataset_with_prediction = "/data/chenran/llama_data_collect/value_head_training/llama_data/" + division + "/MMLU_5shot_with_IK"
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

  def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        return_attention_mask=True
        )
  
  def prepare_dataset_entry(example):
    #include up to "Answer:" -> reflects P(IK)
    #includes up to "Answer: C" -> reflects P(correct)
    example['text'] = example['prompt']
    correct_prob = example["correct_prob"]
    example['label'] = [1.0 - correct_prob, correct_prob]
    return example
  
  
  raw_dataset = load_from_disk(dataset_dir)
  dataset = raw_dataset.map(prepare_dataset_entry)
  dataset = dataset.map(tokenize_function, batched=True)
  dataset = dataset.remove_columns(['prompt', 'ABCD_probs', 'is_correct', 'ABCD_entropy', 'correct_prob','model_answers', 'question', 'subject', 'choices', 'text', 'answer'])
  print(dataset.column_names)

  data_collator  = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
  train_dataloader = DataLoader(
    dataset, shuffle=False, batch_size=8, collate_fn=data_collator
  )
  losses = []
  IK_pred = []
  for batch in tqdm.tqdm(train_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
      outputs = model(**batch, return_dict=True)
    loss = outputs['loss'].detach()
    #output keys: [loss, logits, past_key_values]

    losses.append(loss.item())
    predicted_ik = nn.functional.sigmoid(outputs['logits'][:, 1]).tolist()
    IK_pred.extend(predicted_ik)
    """ 
    print(outputs['logits']) 
    print(nn.BCEWithLogitsLoss()(outputs['logits'], batch['labels']))
    print(loss.item())
    print(batch['labels'].dtype)
    print(model.base_model.config.problem_type)
    return """
  
  dataset_with_predicted_IK = raw_dataset.add_column("predicted IK", IK_pred)
  dataset_with_predicted_IK.save_to_disk(dataset_with_prediction)
  

if __name__ == "__main__":
  main()
