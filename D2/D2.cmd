executable = ./src/proc_docset.sh
error = proc_docset.error
log = proc_docset.log
getenv = true
notification = complete
arguments = "./data/2009_TAC/devtest/GuidedSumm10_test_topics.xml ./data/2009_TAC/evaltest/GuidedSumm11_test_topics.xml ./data/2009_TAC/training/2009/UpdateSumm09_test_topics.xml ./data/selected_files/training ./outputs/training_output ./data/selected_files/devtest ./outputs/devtest_output ./data/selected_files/evaltest ./outputs/evaltest_output"
request_memory = 512
request_GPUs = 1
Queue