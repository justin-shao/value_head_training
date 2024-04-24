import numpy as np
from datasets import Dataset
import math

def count_digits(n):
    if n > 0:
        digits = int(math.log10(n)) + 1
    elif n == 0:
        digits = 1
    else:
        digits = int(math.log10(-n)) + 2 
    return digits

def main():
    add_dataset = {"context": [], "answer":[], "answer_digits": []}
    dataset_size = 100
    # 1-5 digit addition
    context_template = "Question: What is {0} + {1}?\n\nAnswer: "
    for digit in range(1, 6):
        for i in range(dataset_size):
            a = np.random.randint(0, 10**digit)
            b = np.random.randint(0, 10**digit)
            result = a + b
            digits = count_digits(result)
            context = context_template.format(a, b)
            add_dataset['context'].append(context)
            add_dataset['answer'].append(result)
            add_dataset['answer_digits'].append(digits)

    # 1-5 digit substraction
    context_template = "Question: What is {0} - {1}?\n\nAnswer: "
    for digit in range(1, 6):
        for i in range(dataset_size):
            a = np.random.randint(0, 10**digit)
            b = np.random.randint(0, 10**digit)
            result = a - b
            digits = count_digits(result)
            context = context_template.format(a, b)
            add_dataset['context'].append(context)
            add_dataset['answer'].append(result)
            add_dataset['answer_digits'].append(digits)
    
    # 1-3 digit multiplication
    context_template = "Question: What is {0} * {1}?\n\nAnswer: "
    for digit in range(1, 4):
        for i in range(dataset_size):
            a = np.random.randint(0, 10**digit)
            b = np.random.randint(0, 10**digit)
            result = a * b
            digits = count_digits(result)
            context = context_template.format(a, b)
            add_dataset['context'].append(context)
            add_dataset['answer'].append(result)
            add_dataset['answer_digits'].append(digits)

    dataset = Dataset.from_dict(add_dataset)
    dataset.save_to_disk("arithmetic_dataset")
       

if __name__ == "__main__":
  main()
