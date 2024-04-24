import torch
import numpy as np
import os
import random
import time
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset, load_from_disk, concatenate_datasets


def get_k_shot_examples(dataset, k, balanced=True, class_count=None, shuffle=False, save=True, path_to_save=None):
  prompt = ""
  template = "Question: {0}\nA. {1}\nB. {2}\nC. {3}\nD. {4}\nAnswer: {5}\n\n"
  example_indices = []

  letter_choice_labels = 'ABCD'

  if balanced:
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
    
  else:
    example_indices = range(k)

  if save:
      if not os.path.exists(path_to_save):
          path_to_save.mkdir(parents=True, exist_ok=True)
      with open(str(path_to_save) + "/indices.txt", 'w+') as fp:
          fp.write(str(example_indices) + '\n') 

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


def main():
  topics = ['abstract_algebra', 'anatomy', 'astronomy', 
           'business_ethics', 'clinical_knowledge', 'college_biology', 
           'college_chemistry', 'college_computer_science', 'college_mathematics', 
           'college_medicine', 'college_physics', 'computer_security', 
           'conceptual_physics', 'econometrics', 'electrical_engineering', 
           'elementary_mathematics', 'formal_logic', 'global_facts', 
           'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
           'high_school_european_history', 'high_school_geography', 
           'high_school_government_and_politics', 'high_school_macroeconomics', 
           'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 
           'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 
           'high_school_world_history', 'human_aging', 'human_sexuality', 
           'international_law', 'jurisprudence', 'logical_fallacies', 
           'machine_learning', 'management', 'marketing', 'medical_genetics', 
           'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 
           'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 
           'professional_medicine', 'professional_psychology', 'public_relations', 
           'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

  torch.set_default_device("cuda")

  model_name = "meta-llama/Llama-2-7b-hf"
  model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  for t in topics:
    collect(t, model, tokenizer)


def collect(subset, model, tokenizer):
  #prevents memory fragmentation
  os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

  dataset_name = "cais/mmlu"
  mmlu_dataset = load_dataset(dataset_name, subset)
  test_set = "validation"
  if test_set == "test":
    path_to_save = Path(os.getcwd() + "/llama_data/test/MMLU_5shot/" + subset)
  elif test_set == "validation":
    path_to_save = Path(os.getcwd() + "/llama_data/val/MMLU_5shot/" + subset)
  else:
    raise Exception("Unkown set \"" + test_set + "\" in dataset " + dataset_name)
  
  save = True
  divisions = mmlu_dataset.keys()
  if test_set not in divisions:
    raise Exception("Cannot find \"" + test_set + "\" set in dataset: " + "mmlu/" + subset)
  
  use_dev_set = True
  if use_dev_set:
    prompt_dataset = mmlu_dataset['dev']
  else:
    non_test_sets = list(mmlu_dataset[subset] for subset in divisions if subset != test_set)
    prompt_dataset = concatenate_datasets(non_test_sets)

  k = 5
  options = 4
  balanced = True
  k_shot_examples = get_k_shot_examples(prompt_dataset, k, balanced, options, save=save, path_to_save=path_to_save)

  if save:
    with open(str(path_to_save) + "/indices.txt", 'a') as fp:
      fp.write('\nPrompt:\n' + k_shot_examples + '\n')

  def evaluate(example):
    def get_answers(entry, num_samples, temp=1.0, new_token_limit=10):
      # Generate multiple samples in one batch

      prompt = get_prompt(k_shot_examples, entry)
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
      return (answers, outputs.scores, prompt)

    #Saving logits of ["A", "B", "C", "D"], so only need one forward/generation step
    sample_number = 1
    #scores: tuple with length = generated tokens , element of shape(batch, vocab_size)
    answers, scores, prompt = get_answers(example, sample_number, 1.0, 1)

    # save the results
    example["model_answers"] = answers
    example["prompt"] = prompt
    
    #scores: tuple with length = generated tokens , element of shape(batch, vocab_size)
    last_token_scores = scores[-1]
    # step 1: run softmax on logits
    probs = nn.functional.softmax(last_token_scores, dim=-1)

    # step 2: get probs corresponding to ["A", "B", "C", "D"]
    choice_indices = tokenizer("A B C D").input_ids
    if tokenizer.add_bos_token:
      choice_indices = choice_indices[1:5]
    
    example["ABCD_probs"] = probs[:, choice_indices]
    #print(probs[:, choice_indices])
    return example

  results_dataset = mmlu_dataset[test_set].map(evaluate)
  if save:
    results_dataset.save_to_disk(str(path_to_save))


if __name__ == "__main__":
  main()
