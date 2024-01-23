import sys
import os
import xml.etree.ElementTree as ET
import nltk

def process(input_directory, output_directory, docID_docsetID_pairs):
    files = os.listdir(input_directory)

    # Iterate through each file
    for file_name in files:
        # Create the full path to the file
        file_path = os.path.join(input_directory, file_name)
        with open(file_path, 'r') as file:
            xml_data = file.read()
            
            # Parse the XML data
            root = ET.fromstring(xml_data)

            # Iterate over each DOC element
            for doc in root.findall('DOC'):
                docID = doc.get("id")
                if docID not in docID_docsetID_pairs:
                    continue
                # Extract HEADLINE, DATELINE, CATEGORY
                if doc.find('HEADLINE') is not None:
                    headline = doc.find('HEADLINE').text.strip()
                else:
                    headline = ""
                if doc.find('DATELINE') is not None:
                    dateline = doc.find('DATELINE').text.strip()
                else:
                    dateline = ""

                # Create the directory if it doesn't exist
                os.makedirs(os.path.join(output_directory, docID_docsetID_pairs[docID]), exist_ok=True)
                with open(os.path.join(output_directory, docID_docsetID_pairs[docID], docID), 'w') as output:                    

                    # write the extracted information
                    output.write(f"HEADLINE: {headline}\n")
                    output.write(f"DATELINE: {dateline}\n")
                    # write tokenized sentences in TEXT
                    output.write("\n")
        
                    for paragraph in doc.find('TEXT').findall('P'):
                        paragraph = paragraph.text.replace('\n', ' ')
                        sentences = nltk.sent_tokenize(paragraph)
                        for sentence in sentences:
                            # Tokenize the sentence into a list of words
                            tokenized_words = " ".join(nltk.word_tokenize(sentence))
                            output.write(tokenized_words + "\n")
                        output.write("\n")
                    output.write("\n")
                
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
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]
    
    docID_docsetID_pairs = {} # The name of the subdirectory is the same as the docsetA id

    with open(training_xml, "r") as training:
        get_docsetID_docID_match(training, docID_docsetID_pairs)
        
    process(input_directory, output_directory, docID_docsetID_pairs)
    
    