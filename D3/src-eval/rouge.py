"""
This is the sample code for implementing ROUGE using the package.
"""
from rouge import Rouge

def calculate_rouge(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores

# Example usage
reference_summary = "This is the reference summary."
hypothesis_summary = "This is the hypothesis summary."

rouge_scores = calculate_rouge(reference_summary, hypothesis_summary)

print("ROUGE-1 Precision:", rouge_scores[0]['rouge-1']['p'])
print("ROUGE-1 Recall:", rouge_scores[0]['rouge-1']['r'])
print("ROUGE-1 F1 Score:", rouge_scores[0]['rouge-1']['f'])

print("\nROUGE-2 Precision:", rouge_scores[0]['rouge-2']['p'])
print("ROUGE-2 Recall:", rouge_scores[0]['rouge-2']['r'])
print("ROUGE-2 F1 Score:", rouge_scores[0]['rouge-2']['f'])

print("\nROUGE-L Precision:", rouge_scores[0]['rouge-l']['p'])
print("ROUGE-L Recall:", rouge_scores[0]['rouge-l']['r'])
print("ROUGE-L F1 Score:", rouge_scores[0]['rouge-l']['f'])
