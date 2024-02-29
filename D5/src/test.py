import spacy

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample sentence
sentence = "The fox in the forest jumps over the lazy dog."

# Process the sentence using SpaCy
doc = nlp(sentence)

# Iterate over noun chunks in the sentence
for chunk in doc.noun_chunks:
    print("Noun Chunk:", chunk.text)

    # Find all tokens attached to the noun chunk's root token
    modifiers = []
    for token in chunk.root.subtree:
        # Check if the token is not the root token itself and is not the noun chunk's text
        if token != chunk.root and token.text != chunk.text:
            modifiers.append(token.text)

    # If there are any modifiers, print them
    if modifiers:
        print("Modifiers after noun chunk:", modifiers)
    else:
        print("No modifiers after noun chunk")

