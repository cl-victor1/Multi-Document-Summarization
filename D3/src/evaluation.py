from rouge_score import rouge_scorer
import sys 
import os 
import csv

# hypothesis_summary_filename = sys.argv[1] 
# model_summary_filename = sys.argv[2] 



def get_rouge_score(hypothesis_summary_filename, model_summary_filename):
    with open(hypothesis_summary_filename) as hypothesis_file:
        hypothesis_summary = hypothesis_file.read() 

    with open(model_summary_filename) as model_file:
        model_summary = model_file.read() 

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(model_summary, hypothesis_summary) 

    return scores


def main():

    model_summary_dir = sys.argv[1]
    hypothesis_summary_dir = sys.argv[2]


    hypothesis_summary_filenames = []
    
    model_summaries_A = []
    # model_summaries_B = []
    # model_summaries_C = []
    # model_summaries_D = []


    model_summaries = {}


    for file in os.listdir(hypothesis_summary_dir):
        hypothesis_summary_filenames.append(hypothesis_summary_dir+"/"+file)

    for file in os.listdir(model_summary_dir):
        if file.endswith('.A'):
            model_summaries_A.append(model_summary_dir+"/"+file)
        # if file.endswith('.B'):
        #     model_summaries_B.append(model_summary_dir+file)
        # if file.endswith('.C'):
        #     model_summaries_C.append(model_summary_dir+file)
        # if file.endswith('.D'):
        #     model_summaries_D.append(model_summary_dir+file)
            
        topicID = file.split("-")[0]
        if topicID not in model_summaries:
            model_summaries[topicID] = []
        model_summaries[topicID].append(model_summary_dir+"/"+file)

    # print(model_summaries)


    out_dict = [] 
    fields = ['topicID', 'ROUGE-1-avgP', 'ROUGE-1-avgR', 'ROUGE-1-avgF', 'ROUGE-2-avgP', 'ROUGE-2-avgR', 'ROUGE-2-avgF', 'ROUGE-L-avgP', 'ROUGE-L-avgR', 'ROUGE-L-avgF']

    name = hypothesis_summary_dir.split("/")[-1]

    outfile_name = f'rouge_output_{name}.csv'

    with open(outfile_name, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()

        for i in range(len(hypothesis_summary_filenames)):
            row = {}
            filename = hypothesis_summary_filenames[i].split("/")[-1]
            topicID = filename.split("-")[0]
            row['topicID'] = topicID

            model_summs = model_summaries[topicID]
            scores = []
            for summary in model_summs:
                scores.append(get_rouge_score(hypothesis_summary_filenames[i], summary)) 
            # scoreA = get_rouge_score(model_summaries_A[i], hypothesis_summary_filenames[i])
            # scoreB = get_rouge_score(model_summaries_B[i], hypothesis_summary_filenames[i])
            # scoreC = get_rouge_score(model_summaries_C[i], hypothesis_summary_filenames[i])
            # scoreD = get_rouge_score(model_summaries_D[i], hypothesis_summary_filenames[i])


            # scores = [scoreA,scoreB,scoreC,scoreD]


            
            print(f'ROUGE for {filename} is:', scores[0])

            row['ROUGE-1-avgP'] = sum([score['rouge1'][0] for score in scores]) / len(scores)
            row['ROUGE-2-avgP'] = sum([score['rouge2'][0] for score in scores]) / len(scores)
            row['ROUGE-L-avgP'] = sum([score['rougeL'][0] for score in scores]) / len(scores)

            row['ROUGE-1-avgR'] = sum([score['rouge1'][1] for score in scores]) / len(scores)
            row['ROUGE-2-avgR'] = sum([score['rouge2'][1] for score in scores]) / len(scores)
            row['ROUGE-L-avgR'] = sum([score['rougeL'][1] for score in scores]) / len(scores)
            
            row['ROUGE-1-avgF'] = sum([score['rouge1'][2] for score in scores]) / len(scores)
            row['ROUGE-2-avgF'] = sum([score['rouge2'][2] for score in scores]) / len(scores)
            row['ROUGE-L-avgF'] = sum([score['rougeL'][2] for score in scores]) / len(scores)

            out_dict.append(row)

        writer.writerows(out_dict)
main()
