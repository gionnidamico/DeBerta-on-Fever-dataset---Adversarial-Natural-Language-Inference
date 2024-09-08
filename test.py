from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import os

# combine the elements from a fever entry to prepare a feasible input for 
def preprocess_input(entry, tokenizer, use_POS=False):

    # Sample premise and hypothesis with POS tagging
    premise = entry["premise"]
    hypothesis = entry["hypothesis"]

    if use_POS:
        # Corresponding POS tags (this is just a placeholder for the actual POS tags)
        premise_pos = " ".join([wsd['pos'] for wsd in entry["wsd"]["premise"]])             # create a sentence with the list of all the POS tags sepaated by space
        hypothesis_pos = " ".join([wsd['pos'] for wsd in entry["wsd"]["hypothesis"]])       # 

        # Combine text with POS tags
        premise_with_pos = f"{premise} POS: {premise_pos}"
        hypothesis_with_pos = f"{hypothesis} POS: {hypothesis_pos}"
        
        # Prepare the combined input for the model
        combined_input = f"[CLS] {premise_with_pos} [SEP] {hypothesis_with_pos}"
        encoding = tokenizer(combined_input, truncation=True, padding='max_length', max_length=438)
    
    else:
        combined_input = f"[CLS] {premise} [SEP] {hypothesis}"
        encoding = tokenizer(combined_input, truncation=True, padding='max_length', max_length=256)

    # Map labels as integers representing the classes
    label_map = {'ENTAILMENT': 0, 'CONTRADICTION': 1, 'NEUTRAL': 2}
    int_label = label_map[entry['label']]


    return {**encoding, 'label': int_label}


path = './models/nopos_base_model'

# Load the saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(path+'/saved_model')
tokenizer = AutoTokenizer.from_pretrained(path+'/saved_model')

# Load datasets
from datasets import load_dataset

fever_plus = load_dataset("tommasobonomo/sem_augmented_fever_nli")

# Apply preprocessing to datasets
test_dataset = fever_plus['test'].map(preprocess_input, fn_kwargs={'tokenizer': tokenizer, 'use_POS':True})

# Set format for PyTorch
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


# Initialize the Trainer
trainer = Trainer(
    model=model,
    train_dataset=None,
    eval_dataset=None
)


# Ensure the result directory exists
os.makedirs('./results', exist_ok=True)

# Assuming you have a Trainer and test_dataset already defined
# Get predictions
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

# Compute F1 score
f1 = f1_score(labels, preds, average='weighted')
print(f"F1 Score on the test set: {f1}")

# Compute confusion matrix
cm = confusion_matrix(labels, preds)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - F1 Score: ' + f"{f1 * 100:.1f} %")

# Save the plot
plt.savefig('./results/confusion_matrix.png')
plt.show()
plt.close()

print("Confusion matrix saved to ./results/confusion_matrix.png")
