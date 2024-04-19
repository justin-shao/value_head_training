from datasets import load_from_disk, Dataset
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
    
    # 3 columns: task name, avg accuracy, ECE
    benchmark_results = pd.DataFrame({'subject':[], "avg_accuracy": [], "ECE": []})
    for subtask in topics:
       dataset_dir = Path(os.getcwd() + "/data/test/MMLU_postprocess/" + subtask)
       dataset = load_from_disk(dataset_dir)

       df_pandas = pd.DataFrame(dataset).drop(["model_answers"], axis=1)
       exploded_df = df_pandas.explode(["ABCD_probs", "is_correct"])
       exploded_df = exploded_df.drop(["question", 'subject', 'choices',"answer", "ABCD_entropy"], axis=1)

       bins = pd.qcut(pd.to_numeric(exploded_df["ABCD_probs"]), 10, duplicates="drop")
       exploded_df["group"] = bins
       grouped_df = exploded_df.groupby("group", observed=False).agg({
          'is_correct': 'mean',     # the average chances of the answer being correct
          'ABCD_probs': 'mean',     # the average probabilities of this cateory
          'prompt': 'count'
       }).rename(columns={"prompt": "count"})

       avg_acc = np.mean(df_pandas['correct_prob'])
       total_counts = len(exploded_df)
       bin_differences = np.abs(grouped_df["is_correct"] - grouped_df["ABCD_probs"])
       ece = np.sum((grouped_df['count'] / total_counts) * bin_differences)

       df_row = pd.DataFrame({
          'subject': subtask, 
          "avg_accuracy": avg_acc, 
          "ECE": ece
       }, index=[0])
       benchmark_results = pd.concat([benchmark_results, df_row], ignore_index=True)
    
    print(benchmark_results)
    results_path = str(os.getcwd() + "/mmlu_bench_results")
    benchmark_results_dataset = Dataset.from_pandas(benchmark_results)
    benchmark_results_dataset.save_to_disk(results_path)

if __name__ == "__main__":
  main()
