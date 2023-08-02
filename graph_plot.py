import torch
import matplotlib
matplotlib.use("Agg")  # Set the backend to Agg (non-interactive)
import matplotlib.pyplot as plt

def generate_loss_graph():
    # Load the training results from the saved file and map to CPU
    checkpoint = torch.load('training_results.pt', map_location=torch.device('cpu'))

    # Extract the training and validation losses
    train_losses = checkpoint['all_losses']
    val_losses = checkpoint['all_val_losses']

    # Plot the training vs. validation loss graph
    num_epochs = len(train_losses)
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot as an image (e.g., PNG)
    plt.savefig("loss_graph.png")

    # Alternatively, if you still want to show the plot interactively, you can call plt.show()
    # plt.show()
