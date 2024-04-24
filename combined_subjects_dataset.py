from datasets import load_from_disk, Dataset, concatenate_datasets
from pathlib import Path
from scipy.stats import entropy
import pandas as pd
import numpy as np
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
    
    all_datasets = []
    split = "val"
    for topic in topics:
       dataset_dir = Path(os.getcwd() + "/llama_data/"+ split + "/MMLU_5shot_postprocess/" + topic)
       all_datasets.append(load_from_disk(dataset_dir))

    total_dataset = concatenate_datasets(all_datasets)
    total_dataset.save_to_disk(os.getcwd() + "/llama_data/"+ split +"/MMLU_5shot_postprocess/all")

if __name__ == "__main__":
  main()
