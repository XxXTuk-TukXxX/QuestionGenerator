import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import time
from graph_plot import generate_loss_graph


start_time_load_data = time.time()

# 1. Load the JSON dataset
with open(r"datasets/ParaphraseRC_test.json", "r") as file:
    raw_dataset = json.load(file)

# split dataset into training and validation
split_index = int(len(raw_dataset) * 0.7)

train_raw_dataset = raw_dataset[:split_index]
val_raw_dataset = raw_dataset[split_index:]

# 2. Extract plot-question pairs
input_statements = []
target_questions = []
for entry in train_raw_dataset:
    plot = entry['plot']
    for qa_entry in entry['qa']:
        if not qa_entry['no_answer'] and qa_entry['answers']:
            for answer in qa_entry['answers']:
                input_statements.append(plot)
                target_questions.append(qa_entry['question'])

val_input_statements = []
val_target_questions = []

for entry in val_raw_dataset:
    plot = entry['plot']
    for qa_entry in entry['qa']:
        if not qa_entry['no_answer'] and qa_entry['answers']:
            for answer in qa_entry['answers']:
                val_input_statements.append(plot)
                val_target_questions.append(qa_entry['question'])
                
# 3. Tokenize the data

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# training
input_encodings = tokenizer(input_statements, return_tensors="pt", padding=True, truncation=True, max_length=512)
target_encodings = tokenizer(target_questions, return_tensors="pt", padding=True, truncation=True, max_length=512)

# validation
val_input_encodings = tokenizer(val_input_statements, return_tensors="pt", padding=True, truncation=True, max_length=512)
val_target_encodings = tokenizer(val_target_questions, return_tensors="pt", padding=True, truncation=True, max_length=512)

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

# Create a PyTorch Dataset and DataLoader for training and validation data

dataset = QuestionGenerationDataset(input_encodings, target_encodings)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

validation_dataset = QuestionGenerationDataset(val_input_encodings, val_target_encodings)
validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=True) # NOTE: changed shuffle from false to true

end_time_load_data = time.time()

# Check GPU setup
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

# 5. Load the T5 model

model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Verify model's location
print(next(model.parameters()).device)

# 6. Define the training loop and fine-tune the model
optimizer = AdamW(model.parameters(), lr=1e-4)
num_epochs = 30 # NOTE: change epochs 

train_losses = []
val_losses = []

start_time_train = time.time()

for epoch in range(num_epochs):
    #print(f"Starting epoch {epoch + 1}/{num_epochs}")
    print("Starting epoch {}/{}".format(epoch + 1, num_epochs))
    
    model.train()  # Set the model to training mode

    total_train_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # Move batch tensors to the device
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

        #if batch_idx % 100 == 0:
            #print("Epoch {}, Batch {}/{} Loss: {}".format(epoch + 1, batch_idx, len(dataloader), loss.item()))
            #print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}")

	# Calculate average training loss for this epoch
    avg_train_loss = total_train_loss / len(dataloader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0

    with torch.no_grad():
        for val_batch in validation_dataloader:
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            val_outputs = model(input_ids=val_batch["input_ids"], attention_mask=val_batch["attention_mask"], labels=val_batch["labels"])
            val_loss = val_outputs.loss
            total_val_loss += val_loss.item()

	# Calculate average validation loss for this epoch
    avg_val_loss = total_val_loss / len(validation_dataloader)
    val_losses.append(avg_val_loss)
    
    print("Epoch {}, Average Training Loss: {:.4f}, Average Validation Loss: {:.4f}".format(epoch + 1, avg_train_loss, avg_val_loss))

end_time_train = time.time()

# Calculate the durations
time_load_data = end_time_load_data - start_time_load_data
time_train = end_time_train - start_time_train

# Print the durations
print(f"Time taken to load and tokenize data: {time_load_data:.4f} seconds")
print(f"Time taken to train the model: {time_train:.4f} seconds")	

print('Saving results')
torch.save({
    'all_losses': train_losses,
    'all_val_losses': val_losses,
    'model': model.state_dict()
}, 'training_results.pt')
     
save_directory = r"trained_t5_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Plot the training vs. validation loss graph
generate_loss_graph()
