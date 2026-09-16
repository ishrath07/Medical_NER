import spacy

def load_model():
    return spacy.load("en_core_sci_sm")

def extract_entities(nlp, text: str):
    doc = nlp(text)
    return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

if __name__ == "__main__":
    nlp = load_model()
    sample_text = (
        "The patient was prescribed 500mg of Metformin twice daily "
        "for management of Type 2 Diabetes Mellitus, diagnosed on 03/14/2024."
    )
    entities = extract_entities(nlp, sample_text)
    for text, label, start, end in entities:
        print(f"{text!r:40} {label:15} [{start}:{end}]")