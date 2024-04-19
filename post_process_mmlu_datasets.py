from datasets import load_from_disk
from pathlib import Path
from scipy.stats import entropy
import os

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
   
   for sub_task in topics:
       dataset_dir = Path(os.getcwd() + "/data/test/MMLU/" + sub_task)
       dataset = load_from_disk(dataset_dir)
       dataset = dataset.map(post_process)
       dataset.save_to_disk(os.getcwd() + "/data/test/MMLU_postprocess/" + sub_task)

def post_process(example):
   is_correct = [0] * 4
   is_correct[example["answer"]] = 1
   example["is_correct"] = is_correct

   example["ABCD_probs"] = example["ABCD_probs"][0]
   assert(len(example["ABCD_probs"]) == 4)

   example["ABCD_entropy"] = entropy(example['ABCD_probs'], base=2)
   example["correct_prob"] = example["ABCD_probs"][example["answer"]]
   return example


if __name__ == "__main__":
  main()
