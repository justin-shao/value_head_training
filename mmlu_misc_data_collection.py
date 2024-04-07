import torch
import torch.nn as nn
import numpy as np
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset, load_from_disk

mmlu_misc_dataset = load_dataset("cais/mmlu", 'miscellaneous')


def get_k_shot_examples(dataset, k, balanced=True, class_count=None):
  prompt = ""
  template = "Question: {0}\nA. {1}\nB. {2}\nC. {3}\nD. {4}\nAnswer: {5}\n\n"
  example_indices = []

  letter_choice_labels = 'ABCDEFGH'

  if balanced and class_count:
    counts = {}
    upper_limit_per_class = k / class_count
    i = 0
    while i < len(dataset) and sum(counts.values()) < k:
      if counts.get(dataset[i]['answer'], 0) < upper_limit_per_class:
        counts[dataset[i]['answer']] = counts.get(dataset[i]['answer'], 0) + 1
        example_indices.append(i)
      i += 1

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


eight_shot_examples = get_k_shot_examples(mmlu_dataset['validation'], 8, True, 4)

model_name = "meta-llama/Llama-2-7b-hf"
torch.set_default_device("cuda：0")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')


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
        max_new_tokens=new_token_limit
      )
    # Need post-processing to extract the answers. Most because phi-2 produces verbose response

    # input_ids shape: [batch_size, max_seq_len]
    input_len = inputs.input_ids.shape[-1]
    generated_outputs = outputs[:, input_len:]
    generated_texts = tokenizer.batch_decode(generated_outputs)

    ''' # The generated response is the substring after inputs and before the first '\n'
    for gen_text in generated_texts:
      if '\n' in gen_text:
        gen_text = gen_text[:gen_text.index('\n')]
      gen_text = gen_text.strip()

      answers.append(gen_text) '''
    answers = generated_texts
    return answers

  sample_number = 20
  answers = get_answers(example, sample_number, 1.0, 1)
  # save the results
  example["model_answers"] = answers
  return example

path_to_save = "/data/chenran/llama_data_collect/misc_responses"
results_dataset = mmlu_dataset['test'].map(evaluate)
results_dataset.save_to_disk(path_to_save)