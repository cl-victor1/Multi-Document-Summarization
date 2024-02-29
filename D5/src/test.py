import spacy

# Load SpaCy model with neuralcoref extension
nlp = spacy.load("en_core_web_sm")
neuralcoref = spacy.load('en_coref_sm')
nlp.add_pipe(neuralcoref, name='neuralcoref')

# Text to be analyzed
text = """
Barack Obama was born in Hawaii. He is the president of the United States.
The capital of France is Paris. It is a beautiful city.
"""

# Process the text using SpaCy
doc = nlp(text)

# Iterate over each sentence and print the resolved coreferences
for sentence in doc.sents:
    print("Sentence:", sentence.text)
    print("Resolved coreferences:", sentence._.coref_resolved)
    print()
