import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json

# Load the trained model and tokenizer
model_directory = r"outputs/1/trained_t5_model"

model = T5ForConditionalGeneration.from_pretrained(model_directory)
tokenizer = T5Tokenizer.from_pretrained(model_directory)

# Ensure the model is in evaluation mode and move it to the appropriate device
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_question(model, tokenizer, statement, device):
    # Prefix the statement with the task-specific prefix
    input_text = "generate question: " + statement
    input_encoding = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Move the tensor to the device where the model is
    input_encoding = {k: v.to(device) for k, v in input_encoding.items()}
    
    # Generate the output sequence, using max_new_tokens instead of max_length
    with torch.no_grad():
        output = model.generate(**input_encoding, max_new_tokens=100)
    
    # Decode and return the generated question
    question = tokenizer.decode(output[0], skip_special_tokens=True)
    return question


# Test the function
if __name__ == '__main__':
    #statement = input("Enter a statement to generate a question: ")
    
    # 1. Load the JSON dataset
    with open(r"datasets/ParaphraseRC_dev.json", "r") as file:
        statements = json.load(file)

    statements = statements[:10]
	
    # 2. Extract plot-question pairs
    plots = []

    for entry in statements:
        plots.append(entry['plot'])

    for plot in plots:
        generated_question = generate_question(model, tokenizer, plot, device)
        print("-----------------------------------------------------------")
        print(plot)
        print(f"Generated Question: {generated_question}")
