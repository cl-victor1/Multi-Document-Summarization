
#################################################
### STEP 1:
#################################################


#pathDoc = "/dropbox/dropbox/23-24/575x/Data/Documents"
#files  = ["training/2009/UpdateSumm09_test_topics.xml", 
#          "devtest/GuidedSumm10_test_topics.xml", 
#          "evaltest/GuidedSumm11_test_topics.xml"]

import sys
import xml.etree.ElementTree as ET

def extract_ids(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract topic IDs and their corresponding document IDs from docsetA
    topic_docs = []

    for topic in root.iter('topic'):
      topic_cat = topic.get('category')

      topic_id   = topic.get('id')
      docsetA    = topic.find('docsetA')
      docsetA_id = docsetA.get('id')

      if docsetA is not None:
          for doc in docsetA.findall('doc'):
              doc_id = doc.get('id')
              if doc_id:
                  topic_docs.append((topic_id, topic_cat, docsetA_id, doc_id))

    return topic_docs

# File path to your XML file

arg1 = sys.argv[1]

training_docset = ""
devtest_docset  = ""
evaltest_docset = ""

if "training" in arg1:
    training_docset = arg1
elif "devtest" in arg1:
    devtest_docset = arg1
elif "evaltest" in arg1:
    evaltest_docset = arg1
else:
    print("the path seems to be invalid, please check again.")
    sys.exit(0)

docset = {'training': training_docset, 'evaltest': evaltest_docset, 'devtest': devtest_docset}


# Extract and sort the document IDs
training_result_ids = []
evaltest_result_ids = []
devtest_result_ids  = []

if training_docset != "":
    training_result_ids = extract_ids(docset['training'])
elif evaltest_docset != "":
    evaltest_result_ids = extract_ids(docset['evaltest'])
elif devtest_docset != "":
    devtest_result_ids  = extract_ids(docset['devtest'])

# print(training_result_ids)
# print(evaltest_result_ids)
# print(devtest_result_ids)


#################################################
### STEP 2:
#################################################



def transform_doc_id_to_path(doc_id):
    # Case 1: Format like "APW19990914.0234"
    if "_" not in doc_id:
      dir_name = doc_id[:3].lower() # apw
      year     = doc_id[3:7]        # 1999
      month    = doc_id[7:9]        # 09
      day      = doc_id[9:11]       # 14
      rest     = doc_id[-4:-1]      # 0234
      file_name = f"{year}{month}{day}_{dir_name.upper()}_ENG"
      path = f"/corpora/LDC/LDC02T31/{dir_name}/{year}/{file_name}"

    # Case 2: Format like "APW_ENG_20050609.0625"
    else:
      parts = doc_id.split('_')
      dir_name = '_'.join(parts[:2]).lower() # apw_eng
      year     = parts[2][:4]                # 2005
      month    = parts[2][4:6]               # 06
      day      = parts[2][6:8]               # 09
      rest     = parts[2][-4:-1]             # 0625
      file_name = f"{dir_name}_{year}{month}"
      path = f"/corpora/LDC/LDC08T25/data/{dir_name}/{file_name}.xml"

    if year in ["1996", "1997", "1998", "1999", "2000"] and dir_name == 'xie':
      path = path.replace("XIE", "XIN")

    if year in ["1996", "1997", "1998", "1999"] and dir_name == 'nyt':
      path = path[:-4]

    if year not in ["1996", "1997", "1998", "1999", "2000", "2004", "2005", "2006"]:
      path = f"/dropbox/23-24/575x/TAC_2010_KBP_Source_Data/data/2009/nw/{dir_name}/{year}{month}{day}/{doc_id}.LDC2009T13.sgm"

    return path

# for item in training_result_ids:
#   path = transform_doc_id_to_path(item[3])
#   print(f"Doc ID: {item[3]} -> Path: {path}")

# for item in devtest_result_ids:
#   path = transform_doc_id_to_path(item[3])
#   print(f"Doc ID: {item[3]} -> Path: {path}")

# for item in evaltest_result_ids:
#   path = transform_doc_id_to_path(item[3])
#   print(f"Doc ID: {item[3]} -> Path: {path}")

training_paths = []
devtest_paths  = []
evaltest_paths = []

if training_result_ids != []:
    training_paths = [transform_doc_id_to_path(item[3]) for item in training_result_ids]
elif devtest_result_ids != []:
    devtest_paths  = [transform_doc_id_to_path(item[3]) for item in devtest_result_ids]
elif evaltest_result_ids != []:
    evaltest_paths = [transform_doc_id_to_path(item[3]) for item in evaltest_result_ids]

# print(training_paths)

if training_paths != []:
    with open('training_paths', 'w') as f:
        for line in training_paths:
            f.write(f"{line}\n")
elif devtest_paths != []:
    with open('devtest_paths', 'w') as f:
        for line in devtest_paths:
            f.write(f"{line}\n")
elif evaltest_paths != []:
    with open('evaltest_paths', 'w') as f:
        for line in evaltest_paths:
            f.write(f"{line}\n")

########################################
## check whether there are bad paths:
########################################

# import subprocess



# args = sys.argv
# if len(args) == 3:
#     arg2 = sys.argv[-1]

# def check_command_output(command):
#     try:
#         # Run the command and capture the output
#         result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

#         # If the command has output, return True along with the output
#         if result.stdout:
#             return True, result.stdout
#         else:
#             return False, "No output"
#     except subprocess.CalledProcessError as e:
#         # If the command fails, return False along with the error message
#         return False, e.stderr

# # Example usage
# command = "head -10 {}"

# def evaluate_paths(a):

#     if int(a) == 0:
#         iter   = devtest_paths
#         source = devtest_result_ids 
#     elif int(a) == 1:
#         iter = evaltest_paths
#         source = evaltest_result_ids
#     elif int(a) == 2:
#         iter = training_paths
#         source = training_result_ids
    
#     success_count = 0
#     failure_count = 0
#     expected_count = len(source)
#     broken_paths = []

#     for i,p in enumerate(iter):
#         success, output = check_command_output(command.format(p))
#         if success:
#             print("Command executed successfully with output:")
#             #print(output)
#             success_count += 1
#         else:
#             print("Command failed or returned no output:")
#             print(output)
#             failure_count += 1
#             print(source[i])
#             broken_paths.append((source[i], p))
#             # break

#     print(broken_paths)
#     return (success_count, failure_count, expected_count)


# ##### comment out these two lines to disable checking.

# # if len(args) == 3:
# #     c = evaluate_paths(arg2)
# #     print(f'final result : {c}')
