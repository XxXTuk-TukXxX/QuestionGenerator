import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the trained model and tokenizer
model_directory = r"C:\Users\Rojus\Desktop\Test_creator\trained_t5_model"
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
    statement = input("Enter a statement to generate a question: ")
    generated_question = generate_question(model, tokenizer, statement, device)
    print(f"Generated Question: {generated_question}")
