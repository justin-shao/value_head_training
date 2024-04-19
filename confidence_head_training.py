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
import evaluate
import torch.nn as nn
import torch
import numpy as np
import os
from peft import get_peft_model, LoraConfig, TaskType


class CausalLMWithValueHeadForValueHeadTraining(AutoModelForCausalLMWithValueHead):
  """
  Simple wrapper around AutoModelForCausalLMWithValueHead.
  Change made to forward() method is to return loss calculated on last token's value prediction
  """
  def __init__(self, pretrained_model, **kwargs):
    super().__init__(pretrained_model, **kwargs)
    self.loss_fnc = nn.BCEWithLogitsLoss()
     
  def forward(self, input_ids=None, past_key_values=None, attention_mask=None, **kwargs):
    """
    No need for lm_logits. Keep loss at first element to use Trainer API.
    
    Returns: (BCE loss, logits evaluated on last token)
    """
    # outputs = tuple(lm_logits, loss, value_head_results)
    # lm_logits shape: (batch, max_seq_len, vocab_size)
    # value_head_results shape: (batch, max_seq_len)
    labels = None
    if "labels" in kwargs:
     labels = kwargs.pop('labels', None)
    
    outputs = super().forward(input_ids, past_key_values, attention_mask, **kwargs)
    
    logits = outputs[2]
    batch_size = outputs[2].shape[0]

    # Determines the index before the first padding_token
    sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
    sequence_lengths = sequence_lengths % input_ids.shape[-1]
    sequence_lengths = sequence_lengths.to(logits.device)
      
    pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

    loss = None
    if not labels is None:
      loss = self.loss_fnc(pooled_logits.squeeze(), labels.squeeze())

    # TODO: error message, "pooled_logits used before reference". Check what is being passed in?
    return (loss, pooled_logits)
      

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
    model_w_valuehead = CausalLMWithValueHeadForValueHeadTraining.from_pretrained(model_name)
    value_output = model_w_valuehead(**inputs, labels=torch.tensor([1.0, 0.5]))

    sample_generation = model_w_valuehead.generate(**inputs, 
                                                   max_new_tokens=5, 
                                                   pad_token_id=model_w_valuehead.config.pad_token_id)
    value_output = model_w_valuehead(**inputs, labels=torch.tensor([1.0, 0.5]))
    print(value_output)


def main():
  os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
  device = "cuda" if torch.cuda.is_available() else "cpu"

  use_llama = True
  if use_llama:
    model_name = "meta-llama/Llama-2-7b-hf"
    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(
      model_name,
      num_labels = num_labels
    )

  else:
    model_name = "justshao/llama2-7b-with-confidence"
    model = CausalLMWithValueHeadForValueHeadTraining.from_pretrained(
      model_name
    ).to(device) 
  
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left") 
  if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

  def prepare_dataset_entry(example):
    #TODO: extend this logic for responses with length > 1. Can generate a soft_label
    correct_letter_choice = "ABCD"[example['answer']]
    example['correct_count'] = sum(i == correct_letter_choice for i in example['model_answers'])
    example['label'] = example['correct_count'] / len(example['model_answers'])

    #include up to "Answer:" reflects P(IK)
    #if includes up to "Answer: C" reflects P(correct)
    #TODO: should inlucde the answer options, though context length might be problematic.
    #TODO: prompting only question right now
    #template = "Question: {0}\nA. {1}\nB. {2}\nC. {3}\nD. {4}\nAnswer:"
    #example['text'] = template.format(*([example['question']] + example['choices']))
    example['text'] = example['question']
    example["ABCD_probs"] = example["ABCD_probs"][0]
    is_correct = [0] * 4
    is_correct[example["answer"]] = 1
    example["is_correct"] = is_correct
    correct_prob = example["ABCD_probs"][example["answer"]]
    example['label'] = [1.0 - correct_prob, correct_prob]
    return example

  def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        return_attention_mask=True
        )
  
  # Step 1: format dataset
  raw_dataset = load_from_disk("/data/chenran/llama_data_collect/value_head_training/data/MMLU_miscellaneous_1713211091682062920")
  dataset = raw_dataset.map(prepare_dataset_entry)
  dataset = dataset.map(tokenize_function, batched=True)
  dataset = dataset.remove_columns(['model_answers', 'question', 'subject', 'choices', 'text', 'correct_count', 'answer'])

  raw_val_dataset = load_from_disk("/data/chenran/llama_data_collect/value_head_training/data/val/MMLU_miscellaneous_1713348212672635832")
  val_dataset = raw_val_dataset.map(prepare_dataset_entry)
  val_dataset = val_dataset.map(tokenize_function, batched=True)
  val_dataset = val_dataset.remove_columns(['model_answers', 'question', 'subject', 'choices', 'text', 'correct_count', 'answer'])

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

  model.save_pretrained("output_dir")
  """
  print(model.print_trainable_parameters()) 
  print(AutoModelForSequenceClassification.from_pretrained(model_name))
  """

  training_args = TrainingArguments(
    output_dir="misc_test_trainer",
    num_train_epochs=10,
    save_steps=300,
    save_total_limit=5,
    logging_strategy='epoch',
    logging_first_step=True,
    evaluation_strategy='epoch'
  ) 

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=val_dataset
  )
  trainer.train()
  trainer.save_model("final-checkpoint")

"""   train_dataloader = DataLoader(
    dataset, shuffle=True, batch_size=2, collate_fn=data_collator
  )
  
  optimizer = AdamW(model.parameters(), lr=2.5e-5)
  num_epochs = 5
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
  )

  progress_bar = tqdm(range(num_training_steps))
  model.train()
  loss_values = []
  for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in train_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      outputs = model(**batch)
      loss = outputs[0]
      running_loss += loss.detach()
      loss.backward()

      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
      progress_bar.update(1) 
    loss_values.append(running_loss / len(dataset))

  print(loss_values)
  save_dir = "misc_train_checkpoint"
  torch.save({
              'epoch': num_epochs,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': loss_values,
              }, save_dir) """

if __name__ == "__main__":
  main()