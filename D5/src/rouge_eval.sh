python3 ./rouge_eval.py /D5/data/model_devtest /D5/outputs/D5_devtest/LLR_devtest > rouge_scores_LLR_devtest.out
python3 ./rouge_eval.py /D5/data/model_devtest /D5/outputs/D5_devtest/lexrank_compressed_devtest > rouge_scores_lexrank_devtest.out
python3 ./rouge_eval.py /D5/data/model_devtest /D5/outputs/D5_devtest/sumbasic_devtest > rouge_scores_sumbasic_devtest.out
python3 ./rouge_eval.py /D5/data/model_devtest /D5/data/TAC_sharedtask_devset > rouge_scores_peers_devtest.out


python3 ./rouge_eval.py /D5/data/model_evaltest /D5/outputs/D5_evaltest/LLR_evaltest > rouge_scores_LLR_evaltest.out
python3 ./rouge_eval.py /D5/data/model_evaltest /D5/outputs/D5_evaltest/lexrank_compressed_evaltest > rouge_scores_lexrank_evaltest.out
python3 ./rouge_eval.py /D5/data/model_evaltest /D5/outputs/D5_evaltest/sumbasic_evaltest > rouge_scores_sumbasic_evaltest.out
