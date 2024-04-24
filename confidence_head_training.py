from transformers import (
  AutoTokenizer, 
  AutoConfig, 
  AdamW, 
  get_scheduler, 
  TrainingArguments, 
  Trainer, 
  DataCollatorWithPadding,
  AutoModelForSequenceClassification,
) 
import huggingface_hub
from trl import AutoModelForCausalLMWithValueHead
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate
import torch.nn as nn
import torch
import numpy as np
import os
from peft import get_peft_model, LoraConfig, TaskType


def initial_setup():
  model_name = "meta-llama/Llama-2-7b-hf"
  new_model_name = "justshao/llama2-7b-with-confidence"
  torch.set_default_device("cuda:0")
  config = AutoConfig.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
  if '<pad>' not in tokenizer.get_vocab():
      # Add the pad token
      tokenizer.add_special_tokens({"pad_token":"<pad>"})

  model_w_valuehead = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
  model_w_valuehead.pretrained_model.resize_token_embeddings(len(tokenizer))
  model_w_valuehead.config.pad_token_id = tokenizer.pad_token_id

  model_w_valuehead.config.push_to_hub("llama2-7b-with-confidence") 
  model_w_valuehead.push_to_hub("llama2-7b-with-confidence")
  tokenizer.push_to_hub("llama2-7b-with-confidence")


def example():
  #how to generate:
  model_name = "justshao/llama2-7b-with-confidence"
  torch.set_default_device("cuda:0")
  config = AutoConfig.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")

  test_input_text = ["Hello", "Hello World"]
  inputs = tokenizer(test_input_text,
                       padding=True,
                       return_tensors="pt",
                       return_attention_mask=True)
    
  with torch.no_grad():
    model_w_valuehead = AutoModelForSequenceClassification.from_pretrained(model_name)
    value_output = model_w_valuehead(**inputs, labels=torch.tensor([1.0, 0.5]))

    sample_generation = model_w_valuehead.generate(**inputs, 
                                                   max_new_tokens=5, 
                                                   pad_token_id=model_w_valuehead.config.pad_token_id)
    value_output = model_w_valuehead(**inputs, labels=torch.tensor([1.0, 0.5]))
    print(value_output)


def main():
  test_dataset_dir = "/data/chenran/llama_data_collect/value_head_training/llama_data/test/MMLU_5shot_postprocess/all"
  val_dataset_dir ="/data/chenran/llama_data_collect/value_head_training/llama_data/val/MMLU_postprocess/all"
  os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
  device = "cuda" if torch.cuda.is_available() else "cpu"

  use_llama = False
  if use_llama:
    model_name = "meta-llama/Llama-2-7b-hf"
    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(
      model_name,
      num_labels = num_labels,
      torch_dtype=torch.bfloat16
    )
  else:
    model_name = "justshao/llama2-test"
    model = AutoModelForSequenceClassification.from_pretrained(
      model_name,
      num_labels=2,
      vocab_size=32001,
      torch_dtype=torch.bfloat16
    ).to(device) 
    print(model)
  
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left") 
  if use_llama or getattr(tokenizer, "pad_token_id") is None:
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    print(model)
    #return
    huggingface_hub.login("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")
    model.push_to_hub("justshao/llama2-test")
    model.config.push_to_hub("justshao/llama2-test")
    tokenizer.push_to_hub("justshao/llama2-test")
    return

  def prepare_dataset_entry(example):
    #TODO: extend this logic for responses with length > 1. Can generate a soft_label
    """ correct_letter_choice = "ABCD"[example['answer']]
    example['correct_count'] = sum(i == correct_letter_choice for i in example['model_answers'])
    example['label'] = example['correct_count'] / len(example['model_answers']) """
    #include up to "Answer:" -> reflects P(IK)
    #includes up to "Answer: C" -> reflects P(correct)
    example['text'] = example['prompt']
    correct_prob = example["correct_prob"]
    example['label'] = [1.0 - correct_prob, correct_prob]
    return example

  def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        return_attention_mask=True
        )
  
  # Step 1: format dataset
  raw_dataset = load_from_disk(test_dataset_dir)
  dataset = raw_dataset.map(prepare_dataset_entry)
  dataset = dataset.map(tokenize_function, batched=True)
  dataset = dataset.remove_columns(['model_answers', 'question', 'subject', 'choices', 'text', 'answer'])

  raw_val_dataset = load_from_disk(val_dataset_dir)
  val_dataset = raw_val_dataset.map(prepare_dataset_entry)
  val_dataset = val_dataset.map(tokenize_function, batched=True)
  val_dataset = val_dataset.remove_columns(['model_answers', 'question', 'subject', 'choices', 'text', 'answer'])

  print(dataset.features)
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
  
  # Step 2: setup training arguments
  # Using half the LR for llama2
  peft_config = LoraConfig(
        r=16,  
        lora_alpha=64, 
        lora_dropout=0.1, 
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=['v_proj', 'down_proj', 'up_proj', 'q_proj', 'gate_proj', 'k_proj', 'o_proj']
  )
  model.config.use_cache = False 
  model = get_peft_model(model, peft_config)

  model.save_pretrained("output_dir_5shot")
  print(model)
  model.push_to_hub("justshao/llama2-test")
  
  
  print(model.print_trainable_parameters()) 

  training_args = TrainingArguments(
    output_dir="LORA_trained_on_MMLU_5shot_test",
    num_train_epochs=5,
    save_steps=200,
    save_total_limit=5,
    logging_strategy='epoch',
    logging_first_step=True,
    evaluation_strategy='epoch',
    bf16=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    push_to_hub=True
  ) 

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=val_dataset
  )
  trainer.train(resume_from_checkpoint = False)
  trainer.save_model("final-checkpoint_5shot")

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    # forward pass
    outputs = model(**inputs)
    #loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    logits = outputs['logits'] if isinstance(outputs, dict) else outputs[1]
    # compute custom loss
    # only use the second logit, aka logit of IK
    loss_fct = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([0, 1]))
    loss = loss_fct(logits, labels)
    return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
  main()