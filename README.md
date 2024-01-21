# LING-575-project

## D1
Team Setup


## D2
### Project 1: Process a docSet
```
proc_docset.sh input_xml_file1 input_xml_file2 input_xml_file3 training_files_path training_files_output_dir dev_files_path dev_files_output_dir eval_files_path eval_files_output_dir
```
### Structure Overview

#### src/ 
Location of all source code
- `extract.py`
  - Extract topic ids and corresponding doc id from DocSetA (into format such as `topic.docSetA.id`)
  - Transform doc id into path
- `{training|dev|eval}_process.py`
  - Use `XML.etree.ElementTree` library to parse XML
    - If the file is **NOT** a standard XML (e.g. `/corpora/LDC/LDC02T31/apw/1999/19990914_APW_ENG`) -> modify the file by adding `<DOCSTREAM>` tag
  - Find doc id by indexing DOCNO keyword
    - If the file is standard XML, find the doc by looking for id keyword.
  - Extract headline and dateline from HEADLINE and DATELINE keyword.
  - Get individual sentence by `doc.TEXT.findall(“P”)`


#### Documents/
- Including the 3 guided summary xml files for `training`, `dev` and `eval` set
- The data come from **TAC 2010 Guided Summarization task**
- `devtest` and `evaltest` each has an accompanying `categories.txt` file that captures the 5 types of topics occurred in the docset

#### outputs
- Including the results from each of the 3 sets respectively under `{training|devtest|evaltest}_output/`
  - Create sub directories named by `docsetA_id` (e.g., D0901A-A) by joining `output_directory` with `docID_docsetID_pairs[docID]`
    -  Store the outputs in each sub file of the above by the `doc_id` (e.g.,  XIN_ENG_20041113.0001)
  -  In total, 71 docsetA sub directories under `training_output/`, 88 under `devtest_output` and 44 under `evaltest_output`


### Anomalies & Missing Files Handling
- XIN docs only exist after 2000. For these files, we use Fei's convention to find th path.
- XIN docs are actually XIE docs before 2000. XIN's between the period 1996-2000, we use Fei's convention to find th path.
- NYT docs before the year 2000 correspond to files without `_ENG` in the `doc_id`.


### Other Materials for D2
- [Notebook Demo](https://colab.research.google.com/drive/12O_-mGa7kY9bDnpg7UTk68-pzOXtouhh?usp=sharing)
- [Slides](https://docs.google.com/presentation/d/1SA4BHlqPNocj633CXlnUCiJl5jbSAbeq9EIMPmwoDfY/edit?usp=sharing)
- [Report]

### Misc
- Use `XML.etree.ElementTree library` to parse XML in `extract.py` and `{training|dev|eval}_process.py`
- Use `nltk.word_tokenize` to tokenize the sentences in `{training|dev|eval}_process.py`



## D3
TBA


## D4
TBA


## D5
TBA


## D6
TBA

