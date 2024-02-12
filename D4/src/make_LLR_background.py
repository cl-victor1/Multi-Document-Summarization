'''
make the background_corpus based on files of /dropbox/23-24/575x/TAC_2010_KBP_Source_Data/data/2009/nw/cna_eng (19970921-20070609)
'''
import sys
import xml.etree.ElementTree as ET
import nltk
from pathlib import Path


def process(input_directory, output_file):
    with open(output_file, 'w') as output:
        for file_path in input_directory.glob('**/*'):
            if file_path.is_file():
                with open(file_path, 'r') as file:
                    xml_data = file.read()
                    
                    # Parse the XML data
                    root = ET.fromstring(xml_data)           
                        
                    # Find the BODY element
                    body_element = root.find('BODY')      
                    
                    first_paragraph_skipped = False # skip information like "Tokyo,  Oct.  5  (CNA) (By Sofia Wu)"
                    for paragraph in body_element.find('TEXT').findall('P'):
                        if not first_paragraph_skipped:
                            first_paragraph_skipped = True
                            continue  
                        paragraph = paragraph.text.replace('\n', ' ')
                        sentences = nltk.sent_tokenize(paragraph)
                        for sentence in sentences:
                            # Tokenize the sentence into a list of words
                            tokenized_words = " ".join(nltk.word_tokenize(sentence))
                            output.write(tokenized_words + "\n")
                    output.write("\n\n\n")
                
                
if __name__ == "__main__":   
    output_file = sys.argv[1]
    input_directory = sys.argv[2]      
    process(input_directory, output_file)
    
    