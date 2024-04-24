from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from datasets import load_from_disk
import numpy as np
import time

def main():
  # IMPORTANT: have no space at the end of prompt "...\nAnswer:"
  dataset_dir = "arithmetic_MC_dataset"
  results_dir = "/data/chenran/llama_data_collect/value_head_training/llama_data/arithmetic_MC"
  dataset = load_from_disk(dataset_dir)
  indices = np.random.choice(len(dataset), 0, replace=False).tolist()
  example_prompt = ""
  for index in indices:
    example_prompt += dataset[index]['context'] + " " + "ABCD"[dataset[index]['answer_choice']] + '\n\n'
  print(example_prompt)

  torch.set_default_device("cuda")
  model_name = "meta-llama/Llama-2-7b-hf"
  model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

  def get_model_answer(example):
    full_prompt = example_prompt + example['context']    
    tokenized_input = tokenizer(
      full_prompt,
      return_tensors="pt"
    )

    with torch.no_grad():
      outputs = model.generate(
        **tokenized_input,
        do_sample=True,
        temperature=1.0,
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True
      )
    
    input_len = tokenized_input.input_ids.shape[-1]
    generated_outputs = outputs.sequences[:, input_len:]
    generated_texts = tokenizer.batch_decode(generated_outputs)

    #scores: tuple with length = generated tokens, element of shape(batch, vocab_size)
    last_token_scores = outputs.scores[-1]
    # step 1: run softmax on logits
    probs = nn.functional.softmax(last_token_scores, dim=-1)

    # step 2: get probs corresponding to ["A", "B", "C", "D"]
    choice_indices = tokenizer("A B C D").input_ids
    if tokenizer.add_bos_token:
      choice_indices = choice_indices[1:5]
    
    example['prompt'] = full_prompt
    example['model answers'] = generated_texts
    example["ABCD_probs"] = probs[:, choice_indices]
    return example
  

  results = dataset.map(get_model_answer)
  results.save_to_disk(results_dir)
    
if __name__ == "__main__":
  main()

