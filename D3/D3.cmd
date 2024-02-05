executable = ./src/content_select.sh
error = content_select.error
log = content_select.log
getenv = true
notification = complete
arguments = "D3/data/back_corpus_file D3/data/back_corpus D2/outputs/devtest_output D3/outputs/LLR_summaries_no_stopwords_0.05 0 D3/outputs/LLR_summaries_stopwords_0.05 1 0.05 100"
request_memory = 512
request_GPUs = 1
Queue