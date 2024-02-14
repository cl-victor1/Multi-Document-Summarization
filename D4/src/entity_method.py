import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define two sample sentences
sentence1 = "Apple is looking at buying U.K. startup for $1 billion"
sentence2 = "Google is considering purchasing a British company for a hefty amount"

# Process the sentences using spaCy
doc1 = nlp(sentence1)
doc2 = nlp(sentence2)

# Extract entities from the processed sentences
entities1 = set([(entity.text) for entity in doc1.ents])
entities2 = set([(entity.text) for entity in doc2.ents])

# Calculate the Jaccard similarity between the sets of entities
similarity = len(entities1.intersection(entities2)) / len(entities1.union(entities2))

print("Jaccard similarity:", similarity)
