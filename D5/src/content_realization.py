"""
Author: Long Cheng
Organization: University of Washington, the Linguistics Department
Last Update: 1 March, 2024
"""
import re
import string
import sys
import os
import spacy
import nltk

def content_enhance(text): # each text is only one sentence in this approach
    # Remove bylines and editorial content
    cleaned_text = re.sub(r'Byline:\s*[^.,]*\.|Editorial:\s*[^.,]*\.', '', text, flags=re.IGNORECASE)

    # Remove sentence-initial adverbials and conjunct phrases up to the first comma
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)  # Split text into sentences
    cleaned_sentences = []
    for sentence in sentences:
        # Remove adverbials and conjunct phrases up to the first comma
        cleaned_sentence = re.sub(r'^[\w\s]*?,', '', sentence)
        if cleaned_sentence.strip():  # Check if the sentence is not empty
            cleaned_sentences.append(cleaned_sentence)
    cleaned_text = ' '.join(cleaned_sentences)

    # Remove relative clause attributives and attributions without quotes
    cleaned_text = re.sub(r',\s*[^,.]+,|,\s*[^,.]+,', ',', cleaned_text)
    
    # Use regular expression to remove leading and trailing punctuation
    cleaned_text = re.sub(r'^[^\w]+|[^\w]+$', '', cleaned_text) 
    
    # combine the final word and punctuation of each sentence.
    cleaned_text = '. '.join(sentence.strip(string.punctuation + " ") for sentence in cleaned_text.split('.') if sentence.strip()) + "."

    # turn `` and ' into "
    cleaned_text = re.sub(r'``\s([^`]+)\s\'\'', r'"\1"', cleaned_text)
    
    # remove the space before 's
    cleaned_text = re.sub(r'\s\'s', r"'s", cleaned_text)
    
    # remove space after decimal point
    pattern = r'(\d+\.)\s+(\d+)'
    cleaned_text = re.sub(pattern, r'\1\2', cleaned_text)
    
    # capitalize the first letter
    cleaned_text = cleaned_text[0].capitalize() + cleaned_text[1:]
    
    return cleaned_text.strip()  # Strip leading/trailing spaces

# # Example usage
# text = """
# The Bronx district attorney 's office has declined to comment on details of the evidence or how it might be used in the trial .
# Bronx District Attorney Robert Johnson noted New York law does not require him to prove premeditation on the part of the four officers .
# The shooting led to 15 days of protests outside Police Headquarters and reignited the gnawing debate about police in New York City .
# A second set of bullets that struck Diallo 's right leg and foot also figure to be a source of debate .
# """
# cleaned_text = content_realization(text)
# print(cleaned_text)

def resolve_coref(summary):
    nlp = spacy.load("en_core_web_md")  # Load the English language model
    coref_entities = set()
    refined_summary = []
    for line in summary.readlines():
        sentence = line.strip()
        nlp_sentence = nlp(sentence) # spacy object
        for chunk in nlp_sentence.noun_chunks:
            if chunk.root.pos_ == "PRON": # ignore pronouns
                    continue
            head = chunk.root.text # get the head of the NP
            # Find all tokens attached to the noun chunk's root token
            modifiers = []
            for token in chunk.root.subtree:
                # Check if the token is not the root token itself and is not the noun chunk's text
                if token != chunk.root and token.text != chunk.text:
                    modifiers.append(token.text)
            if head not in coref_entities: # this implies the first occurrence of the entity
                coref_entities.add(head)
            else: # if this entity has appeared before, replace the original NP by the head and remove the modifier
                sentence = sentence.replace(chunk.text, head) 
                words = nltk.word_tokenize(sentence)
                head_index = words.index(head)
                modifier = ""
                for i in range(head_index + 1, len(words)):
                    if words[i] in modifiers: # update the modifier sequence
                        modifier += words[i]
                    else: # remove the modifier
                        sentence = sentence.replace(modifier, "")
                        break                
        refined_summary.append(sentence)         
    return refined_summary            
            
def main():
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    # Create the output_directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)
        with open(file_path, 'r') as file, open(os.path.join(output_directory, filename), "w") as output:
            refined_summary = resolve_coref(file) # resolve coreference
            for sentence in refined_summary:
                cleaned_text = content_enhance(sentence)
                output.write(cleaned_text + "\n")
            
if __name__ == "__main__":   
    main()