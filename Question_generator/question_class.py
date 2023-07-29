import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader

# 1. Load the JSON dataset
with open(r"datasets\Rojaus.json", "r") as file:
    raw_dataset = json.load(file)

# 2. Extract question-answer pairs
input_statements = []
target_questions = []
for entry in raw_dataset:
    plot = entry['plot']
    for qa_entry in entry['qa']:
        if not qa_entry['no_answer'] and qa_entry['answers']:
            for answer in qa_entry['answers']:
                input_statements.append(plot)
                target_questions.append(qa_entry['question'])

# 3. Tokenize the data
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
input_encodings = tokenizer(input_statements, return_tensors="pt", padding=True, truncation=True, max_length=512)
target_encodings = tokenizer(target_questions, return_tensors="pt", padding=True, truncation=True, max_length=512)

# 4. Create a PyTorch Dataset and DataLoader
class QuestionGenerationDataset(Dataset):
    def __init__(self, input_encodings, target_encodings):
        self.input_encodings = input_encodings
        self.target_encodings = target_encodings

    def __len__(self):
        return len(self.input_encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_encodings["input_ids"][idx],
            "attention_mask": self.input_encodings["attention_mask"][idx],
            "labels": self.target_encodings["input_ids"][idx]
        }

dataset = QuestionGenerationDataset(input_encodings, target_encodings)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# 5. Load the T5 model

# Check GPU setup
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

# Load model
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

# Verify model's location
print(next(model.parameters()).device)

# 6. Define the training loop and fine-tune the model
optimizer = AdamW(model.parameters(), lr=1e-4)
num_epochs = 3
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    for batch_idx, batch in enumerate(dataloader):
        # Move batch tensors to the device
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}")

save_directory = r"C:\Users\Rojus\Desktop\Test_creator\trained_t5_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)