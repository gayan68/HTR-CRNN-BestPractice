import os
import re
import pickle

def natural_key(text):
    return [int(chunk) if chunk.isdigit() else chunk.lower() 
            for chunk in re.split('(\d+)', text)]

def find_best_model(model0, results):
    bst_model_accuracy_cer = 100
    bst_model_accuracy_wer = 100
    bst_model_cer = ""
    bst_model_wer = ""

    for idx, vs_mdl in enumerate(model0):
        if results[idx][0] < bst_model_accuracy_cer:
            bst_model_cer = vs_mdl
            bst_model_accuracy_cer = results[idx][0]

        if results[idx][1] < bst_model_accuracy_wer:
            bst_model_wer = vs_mdl
            bst_model_accuracy_wer = results[idx][1]

    print(f"CER: {bst_model_cer} | {round(bst_model_accuracy_cer,4)}")
    print(f"WER: {bst_model_wer} | {round(bst_model_accuracy_wer,4)}")

def print_res(result0, pos, model0):
    for idx, result1 in enumerate(result0):
        print(f"{model0[idx]} : {round(result1[pos],4)}")

dir = "800_51"

saved_directory = f'./results/saved_models_{dir}'
saved_models = []
for filename in os.listdir(saved_directory):
    if os.path.isfile(os.path.join(saved_directory, filename)):
        if (filename != f'results_test_{dir}.pkl') and (filename != f'results_val_{dir}.pkl') and (filename != 'logfile.txt'):
            saved_models.append(filename)
saved_models = sorted(saved_models, key=natural_key)

with open(f'{saved_directory}/results_test_{dir}_fft4.pkl', 'rb') as file:
    results_test = pickle.load(file)

with open(f'{saved_directory}/results_val_{dir}_fft4.pkl', 'rb') as file:
    results_val = pickle.load(file)

print("\n## Val Dataset ##")
find_best_model(saved_models, results_val)
print("CER")
print_res(results_val, 0, saved_models) #CER on Test Dataset
print("WER")
print_res(results_val, 1, saved_models) #WER on Test Dataset

print("\n## Test Dataset ##")
find_best_model(saved_models, results_test)
print("CER")
print_res(results_test, 0, saved_models) #CER on Test Dataset
print("WER")
print_res(results_test, 1, saved_models) #WER on Test Dataset


