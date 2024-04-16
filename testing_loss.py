from transformers import (
  AutoTokenizer, 
  AutoConfig, 
  AdamW, 
  get_scheduler, 
  TrainingArguments, 
  Trainer, 
  DataCollatorWithPadding,
  AutoModelForSequenceClassification
) 
from trl import AutoModelForCausalLMWithValueHead
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from confidence_head_training import CausalLMWithValueHeadForValueHeadTraining
import evaluate
import torch.nn as nn
import torch
import numpy as np
import os

  
def main():
  model_name = "justshao/llama2-7b-with-confidence"
  device = "cuda" if torch.cuda.is_available() else "cpu"
  torch.set_default_device(device)

  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
  model = CausalLMWithValueHeadForValueHeadTraining.from_pretrained(
    model_name,
    load_in_8bit=True
  )
  
  PATH = "misc_train_checkpoint"
  checkpoint = torch.load(PATH, map_location=device)
  model_state_read = checkpoint["model_state_dict"]
  # For some reasons, the saved keys are in format: model.[...]
  # While the model expects keys in format: pretrained_model.model.[...]
  state_dict = dict()
  for k, v in model_state_read.items():
    new_key = "pretrained_model." + k
    if "v_head" in k:
      new_key = k
    state_dict[new_key] = v

  model.load_state_dict(state_dict)

  loss = checkpoint['loss']
  model.to(device)
  print(loss)

  val_dataset = load_from_disk("/data/chenran/llama_data_collect/value_head_training/data/val/MMLU_miscellaneous_1713117765655231364")
  
  def get_predicted_logit(example):
    with torch.no_grad():
      inputs = tokenizer(
        example['question'],
        return_tensors="pt",
        return_attention_mask=True
      ) 
      for k, v in inputs.items():
        print(v)
        inputs[k] = v.to(device)
      print(inputs)
      with torch.no_grad():
        outputs = model(**inputs)
      example['IK prediction'] = outputs[1][-1]
    return example
  
  for i in model.named_parameters():
    print(f"{i[0]} -> {i[1].device}")
  val_dataset = val_dataset.map(get_predicted_logit)
  val_dataset.save_to_disk("data/val/misc_with_IK_prediction")


if __name__ == "__main__":
  main()
