from PIL import Image
import pytesseract
import spacy

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open the image file
img = Image.open(r"C:\Users\Rojus\Desktop\Test_creator\pdf_images\output_0.png")

# Use pytesseract to convert the image into text
text = pytesseract.image_to_string(img)

# Save the extracted text into a .txt file
with open(r"C:\Users\Rojus\Desktop\Test_creator\extracted_text.txt", 'w') as f:
    f.write(text)

def generate_questions_from_entities(doc):
    questions = set()  # Using a set to avoid duplicates
    
    for ent in doc.ents:
        # Find the sentence containing this entity
        sentence = next((sent for sent in doc.sents if ent.start >= sent.start and ent.end <= sent.end), None)
        if not sentence:
            continue
        
        sent_text = sentence.text.strip()
        
        # Remove unwanted characters like "�"
        clean_ent_text = ent.text.replace("�", "").strip()

        if ent.label_ == "PERSON":
            if sent_text.startswith(clean_ent_text) and "," in sent_text:
                verb_part = sent_text.split(",")[1].strip().split(" ")[0]
                questions.add(f"Who {verb_part} {clean_ent_text}?")
            else:
                questions.add(f"What did {clean_ent_text} do?")
                
        elif ent.label_ == "ORG":
            if "theory" in sent_text.lower():
                questions.add(f"What is the {clean_ent_text} theory about?")
            else:
                questions.add(f"What is {clean_ent_text}?")
            
        elif ent.label_ == "GPE":
            if sent_text.startswith(clean_ent_text):
                verb_part = sent_text[len(clean_ent_text):].strip().split(" ")[0]
                questions.add(f"What {verb_part} in {clean_ent_text}?")
            else:
                questions.add(f"What happened in {clean_ent_text}?")
                
    return list(questions)

# Load the spacy models
nlp_sm = spacy.load('en_core_web_sm')

def analyze_text(input_file_path, output_file_path):
    # Read the input file
    with open(input_file_path, 'r') as file:
        text = file.read()

    # Process the text
    doc_sm = nlp_sm(text)

    # Open the output file in write mode
    with open(output_file_path, 'w') as file:
        # Sentence Splitting
        sentences = [sent.text for sent in doc_sm.sents]
        file.write(f'Sentences: {sentences}\n\n')

        # Tokenization
        tokens = [token.text for token in doc_sm]
        file.write(f'Tokens: {tokens}\n')

        # Lemmatization
        lemmas = [token.lemma_ for token in doc_sm]
        file.write(f'Lemmas: {lemmas}\n')

        # POS Tagging
        pos_tags = [(token.text, token.pos_) for token in doc_sm]
        file.write(f'POS Tags: {pos_tags}\n')

        # Dependency Parsing
        dependencies = [(token.text, token.dep_) for token in doc_sm]
        file.write(f'Dependencies: {dependencies}\n')

        # Named Entity Recognition
        entities_sm = {(entity.text, entity.label_) for entity in doc_sm.ents}
        file.write(f'Entities: {list(entities_sm)}\n')

analyze_text(r"C:\Users\Rojus\Desktop\Test_creator\extracted_text.txt", r"C:\Users\Rojus\Desktop\Test_creator\tokinized_text.txt")