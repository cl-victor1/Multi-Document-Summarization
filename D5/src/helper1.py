def get_coref_class(data):
    coref_classes = {};

    for docSetA in data:
        coref_classes[docSetA] = {};
        for file in data[docSetA]:
            coref_classes[docSetA][file] = {};
            sentences_list = data[docSetA][file];

            ## round 1, get all possible classes keys
            for i, nlp_sent in enumerate(sentences_list):
                for np in nlp_sent.noun_chunks:
                    # Use the root token of the noun phrase as the head noun
                    # head_noun_lemma = np.root.lemma_
                    if np.root.pos_ == "PRON":
                        continue;
                    if not any([token_in_np.text[0].isupper() or token_in_np.pos_ == "PROPN" for token_in_np in np]):
                        break;
                    head_noun_text = np.root.text
                    if head_noun_text in coref_classes[docSetA][file]:
                        coref_classes[docSetA][file][head_noun_text].append(np.text)
                    else:
                        coref_classes[docSetA][file][head_noun_text] = [np.text]

            ## round 2, get postmodifiers, appos, relcl, etc.
            for i, nlp_sent in enumerate(sentences_list):
                for np in nlp_sent.noun_chunks:

                    if np.text not in coref_classes[docSetA][file]:
                        continue;

                    # Initialize span with the noun phrase itself
                    start, end = np.start, np.end

                    # Only extend to immediate children to avoid spanning across sentences
                    for token in np:
                        for child in token.children:
                            # Check if the child is a postmodifier and ensure it's within the same sentence
                            cond1 = child.dep_ in ['prep', 'appos', 'relcl', 'acl'];
                            cond2 = child.dep_ in ['pobj', 'compound', 'poss'];
                            cond3 = 'mod' in child.dep_;
                            if (cond1 or cond2 or cond3) and child.sent.start == np.sent.start:
                                start = min(start, child.left_edge.i)
                                end = max(end, child.right_edge.i + 1)

                    # Return the corrected span ensuring it's within the bounds of the sentence
                    start = max(start, np.sent.start)
                    end = min(end, np.sent.end)
                    coref_classes[docSetA][file][np.text].append(spacy.tokens.Span(nlp_sent, start, end, label='NP').text)


    return coref_classes;

coref_classes_by_file = get_coref_class(data = spacy_nlp_results_by_file);

coref_classes_by_file

# supervisor,
# name, same name, different persons
