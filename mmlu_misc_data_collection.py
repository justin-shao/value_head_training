import torch
import numpy as np
import os
import random
import time
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from trl import AutoModelForCausalLMWithValueHead
from datasets import load_dataset, load_from_disk, concatenate_datasets

#prevents memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

subset = "miscellaneous"
mmlu_dataset = load_dataset("cais/mmlu", subset)
cwd = os.getcwd()
path_to_save = Path(os.getcwd() + "/data/MMLU_" + subset + "_" + str(time.time_ns()))

def get_k_shot_examples(dataset, k, balanced=True, class_count=None, shuffle=False, save=True):
  prompt = ""
  template = "Question: {0}\nA. {1}\nB. {2}\nC. {3}\nD. {4}\nAnswer: {5}\n\n"
  example_indices = []

  letter_choice_labels = 'ABCD'

  if balanced and class_count:
    counts = {}
    upper_limit_per_class = k / class_count
    i = 0
    while i < len(dataset) and sum(counts.values()) < k:
      if counts.get(dataset[i]['answer'], 0) < upper_limit_per_class:
        counts[dataset[i]['answer']] = counts.get(dataset[i]['answer'], 0) + 1
        example_indices.append(i)
      i += 1

    # if we shuffle the order
    if shuffle:
        random.shuffle(example_indices)
    
    if save:
      if not os.path.exists(path_to_save):
          path_to_save.mkdir(parents=True, exist_ok=True)
      with open(str(path_to_save) + "/indices.txt", 'w+') as fp:
          fp.write(str(example_indices) + '\n') 
  else:
    example_indices = range(k)

  for i in example_indices:
    entry = dataset[i]
    prompt += template.format(*([entry['question']] + entry['choices'] + [letter_choice_labels[entry['answer']]]))
  return prompt


def get_prompt(examples_prefix, entry):
  template = "Question: {0}\nA. {1}\nB. {2}\nC. {3}\nD. {4}\nAnswer:"
  prompt = examples_prefix
  arg_list = [entry['question']] + entry['choices']
  prompt += template.format(*arg_list)
  return prompt

save = True
val_and_dev_dataset = concatenate_datasets([mmlu_dataset['validation'], mmlu_dataset['dev']])
eight_shot_examples = get_k_shot_examples(val_and_dev_dataset, 8, True, 4, save=save)

if save:
  with open(str(path_to_save) + "/indices.txt", 'a') as fp:
      fp.write('\nPrompt:\n' + eight_shot_examples + '\n')


model_name = "meta-llama/Llama-2-7b-hf"
torch.set_default_device("cuda")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def evaluate(example):
  def get_answers(entry, num_samples, temp=1.0, new_token_limit=10):
    # Generate multiple samples in one batch

    prompt = get_prompt(eight_shot_examples, entry)
    inputs = tokenizer(
      [prompt] * num_samples,
      return_tensors="pt",
      return_attention_mask=True)

    # the forward method will automatically set global attention on question tokens
    # The scores for the possible start token and end token of the answer are retrived
    # wrap the function in torch.no_grad() to save memory
    with torch.no_grad():
      outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temp,
        max_new_tokens=new_token_limit,
        output_scores=True,
        return_dict_in_generate=True
      )
    # Need post-processing to extract the answers. Most because phi-2 produces verbose response

    # input_ids shape: [batch_size, max_seq_len]
    input_len = inputs.input_ids.shape[-1]
    generated_outputs = outputs.sequences[:, input_len:]
    generated_texts = tokenizer.batch_decode(generated_outputs)

    ''' # The generated response is the substring after inputs and before the first '\n'
    for gen_text in generated_texts:
      if '\n' in gen_text:
        gen_text = gen_text[:gen_text.index('\n')]
      gen_text = gen_text.strip()

      answers.append(gen_text) '''
    answers = generated_texts
    return (answers, outputs.scores)

  #Saving logits of ["A", "B", "C", "D"], so only need one forward/generation step
  sample_number = 1
  #scores: tuple with length = generated tokens , element of shape(batch, vocab_size)
  answers, scores = get_answers(example, sample_number, 1.0, 1)

  choice_indices = tokenizer("A B C D").input_ids
  if tokenizer.add_bos_token:
     choice_indices = choice_indices[1:5]

  # save the results
  example["model_answers"] = answers
  #scores: tuple with length = generated tokens , element of shape(batch, vocab_size)
  last_token_scores = scores[-1]
  probs = nn.functional.softmax(last_token_scores, dim=-1)

  """ print(choices)
  print(answers)
  print(torch.argmax(scores[0], dim=-1))
  print(last_token_scores)
  print(last_token_scores[:, choices])
  print(probs[:, choice_indices]) """

  # step 1: run softmax on logits
  # step 2: get probs corresponding to ["A", "B", "C", "D"]
  example["ABCD_probs"] = probs[:, choice_indices]
  print(probs[:, choice_indices])
  return example


#evaluate(mmlu_dataset['test'][1])
results_dataset = mmlu_dataset['test'].map(evaluate)
if save:
  results_dataset.save_to_disk(str(path_to_save))

def main():
  pass

if __name__ == "__main__":
  main()
