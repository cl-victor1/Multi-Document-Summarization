import sys
import os
import xml.etree.ElementTree as ET
import nltk

def write_content(headline, dateline, docID, block, output_directory, docID_docsetID_pairs):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.join(output_directory, docID_docsetID_pairs[docID]), exist_ok=True)
    with open(os.path.join(output_directory, docID_docsetID_pairs[docID], docID), 'w') as output:                    

        # write the extracted information
        output.write(f"HEADLINE: {headline}\n")
        output.write(f"DATE_TIME: {dateline}\n")
        # write tokenized sentences in TEXT
        output.write("TEXT:\n")

        for sentence in block.find('TEXT').findall('P'):
            sentence = sentence.text.replace('\n', ' ')
            # Tokenize the sentence into a list of words
            tokenized_words = " ".join(nltk.word_tokenize(sentence))
            output.write(tokenized_words + "\n")
        output.write("\n")

def process(input_directory, output_directory, docID_docsetID_pairs):
    files = os.listdir(input_directory)
    # Iterate through each file
    for file_name in files:
        # Create the full path to the file
        file_path = os.path.join(input_directory, file_name)
        
        # if the file is not a standard XML file
        if file_name[-3:] != "xml":
            # Read the content from the file
            with open(file_path, 'r') as file:
                xml_content = file.read()
            # Modify the content by adding <DOCSTREAM> tags        
            modified_content = f'<DOCSTREAM>{xml_content}</DOCSTREAM>'
            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.write(modified_content)
            
            # parse the xml file after wrapping all <DOC> elements in a single root element
            with open(file_path, 'r') as file:
                xml_data = file.read()
                # edge cases
                doc_xml_escaped = xml_data.replace("&", "and")
                # Parse the XML data
                root = ET.fromstring(doc_xml_escaped)
                # Iterate over each DOC element
                for doc in root.findall('DOC'):
                    docID = doc.find("DOCNO").text.strip()
                    if docID not in docID_docsetID_pairs:
                        continue                    
                    if doc.find('DATE_TIME') is not None:
                        dateline = doc.find('DATE_TIME').text.strip()
                    else:
                        dateline = ""                        
                    # Find the BODY element
                    body_element = doc.find('BODY')
                    # Extract HEADLINE, DATELINE, CATEGORY
                    if body_element.find('HEADLINE') is not None:
                        headline = body_element.find('HEADLINE').text.strip()
                    else:
                        headline = ""      
                                  
                    write_content(headline, dateline, docID, body_element, output_directory, docID_docsetID_pairs)
                    
        # if the file is a standard XML file
        else:
            with open(file_path, 'r') as file:
                xml_data = file.read()            
                # Parse the XML data
                root = ET.fromstring(xml_data)
                # Iterate over each DOC element
                for doc in root.findall('DOC'):
                    docID = doc.get("id")
                    if docID not in docID_docsetID_pairs:
                        continue
                    # Extract HEADLINE, DATELINE
                    if doc.find('HEADLINE') is not None:
                        headline = doc.find('HEADLINE').text.strip()
                    else:
                        headline = ""
                    if doc.find('DATELINE') is not None:
                        dateline = doc.find('DATELINE').text.strip()
                    else:
                        dateline = ""

                    write_content(headline, dateline, docID, doc, output_directory, docID_docsetID_pairs)                         
                          
def get_docsetID_docID_match(data, docID_docsetID_pairs):
    xml_data = data.read()
    # Parse the XML data
    root = ET.fromstring(xml_data)
    # Iterate over each topic
    for topic in root.findall('topic'):
        # Iterate over each docsetA within the current topic
        for docsetA in topic.findall('docsetA'):
            docsetA_id = docsetA.get('id')            
            for doc in docsetA.findall('doc'):
                # Save the data in the dictionary
                docID_docsetID_pairs[doc.get('id')] = docsetA_id          
            
                
if __name__ == "__main__":
    training_xml = sys.argv[1]
    dev_xml = sys.argv[2]
    eval_xml = sys.argv[3]
    input_directory = sys.argv[4]
    output_directory = sys.argv[5]
    
    docID_docsetID_pairs = {} # The name of the subdirectory is the same as the docsetA id

    with open(training_xml, "r") as training, open(dev_xml, "r") as dev, open(eval_xml, "r") as eval:
        get_docsetID_docID_match(training, docID_docsetID_pairs)
        get_docsetID_docID_match(dev, docID_docsetID_pairs)
        get_docsetID_docID_match(eval, docID_docsetID_pairs)
        
    process(input_directory, output_directory, docID_docsetID_pairs)
    
    