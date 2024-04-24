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
    dataset_dir = "arithmetic_MC_dataset"
    add_dataset = {"context": [], "answer_choice":[], "choices": [], "answer_digits": []}
    dataset_size = 100
    # 1-5 digit addition
    context_template = "Question: Find the value of {0} + {1}?\nA. {2}\nB. {3}\nC. {4}\nD. {5}\nAnswer:"
    for digit in range(1, 6):
        for i in range(dataset_size):
            a = np.random.randint(10**(digit-1), 10**digit)
            b = np.random.randint(10**(digit-1), 10**digit)
            result = a + b
            digits = count_digits(result)
            offsets = np.random.choice(np.arange(-result-5, result+5), 4, replace=False)
            while 0 in offsets:
                offsets = np.random.choice(np.arange(-result-5, result+5), 4, replace=False)
            
            #select a choice
            answer_choice = np.random.randint(0,4)
            offsets[answer_choice] = 0
            choices = (offsets + result).tolist()
                    
            context = context_template.format(a, b, *choices)
            add_dataset['context'].append(context)
            add_dataset['answer_choice'].append(answer_choice)
            add_dataset['choices'].append(choices)
            add_dataset['answer_digits'].append(digits)

    # 1-5 digit substraction
    context_template = "Question: Find the value of {0} - {1}? \nA. {2}\nB. {3}\nC. {4}\nD. {5}\nAnswer:"
    for digit in range(1, 6):
        for i in range(dataset_size):
            a = np.random.randint(10**(digit-1), 10**digit)
            b = np.random.randint(10**(digit-1), 10**digit)
            result = a - b

            digits = count_digits(result)
            #Make sure all offsets are non-zero
            offsets = np.random.choice(np.arange(-(a+b)-5, (a+b)+5), 4, replace=False)
            while 0 in offsets:
                offsets = np.random.choice(np.arange(-(a+b)-5, (a+b)+5), 4, replace=False)

            #select a choice
            answer_choice = np.random.randint(0,4)
            offsets[answer_choice] = 0
            choices = (offsets + result).tolist()
                    
            context = context_template.format(a, b, *choices)
            add_dataset['context'].append(context)
            add_dataset['answer_choice'].append(answer_choice)
            add_dataset['choices'].append(choices)
            add_dataset['answer_digits'].append(digits)
    
    # 1-3 digit multiplication
    context_template = "Question: Find the value of {0} * {1}?\nA. {2}\nB. {3}\nC. {4}\nD. {5}\nAnswer:"
    for digit in range(1, 4):
        for i in range(dataset_size):
            a = np.random.randint(10**(digit-1), 10**digit)
            b = np.random.randint(10**(digit-1), 10**digit)
            result = a * b

            digits = count_digits(result)
            #Make sure all offsets are non-zero
            offsets = np.random.choice(np.arange(-result-5, result+5), 4, replace=False)
            while 0 in offsets:
                offsets = np.random.choice(np.arange(-result-5, result+5), 4, replace=False)
            #select a choice
            answer_choice = np.random.randint(0,4)
            offsets[answer_choice] = 0
            choices = (offsets + result).tolist()
                    
            context = context_template.format(a, b, *choices)
            add_dataset['context'].append(context)
            add_dataset['answer_choice'].append(answer_choice)
            add_dataset['choices'].append(choices)
            add_dataset['answer_digits'].append(digits)
    
    dataset = Dataset.from_dict(add_dataset)
    dataset.save_to_disk(dataset_dir)
       

if __name__ == "__main__":
  main()
