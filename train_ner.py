import json
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer

# This function will load your ANNOTATED data from Label Studio
def load_dataset_from_json(filepath):
    """
    Loads data from a JSON file exported from Label Studio.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please make sure you have exported your labeled data from Label Studio and saved it with this name.")
        return None

    # --- Dynamically discover all labels from the data ---
    # Start with 'O' for "Outside" any entity
    label_list = ["O"] 
    for entry in raw_data:
        # An entry might not have annotations if it was skipped
        if 'annotations' in entry and entry['annotations']:
            # The first result list contains the labels
            for annotation in entry['annotations'][0]['result']:
                # The label name is in a list, get the first element
                label_name = annotation['value']['labels'][0]
                if label_name not in label_list:
                    label_list.append(label_name)
    
    # Create the label-to-ID mappings
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    print(f"Discovered labels: {label_list}")
    print(f"ID to label mapping: {id2label}")

    processed_data = {'id': [], 'tokens': [], 'ner_tags': []}
    
    for i, entry in enumerate(raw_data):
        # FIX 2: Make the check more robust. Skip if no annotations or empty results.
        if 'annotations' not in entry or not entry['annotations'] or not entry['annotations'][0]['result']:
            continue
    
        text = entry['data']['text']
        # Use simple whitespace tokenization for alignment
        tokens = text.split()
        
        # Initialize all token labels to 'O' (Outside)
        labels = [label2id['O']] * len(tokens)

        char_pos = 0
        for token_idx, token in enumerate(tokens):
            start_token_char = char_pos
            end_token_char = char_pos + len(token)

            for annotation in entry['annotations'][0]['result']:
                label_start_char = annotation['value']['start']
                label_end_char = annotation['value']['end']
                label_name = annotation['value']['labels'][0]

                # Check if the token falls within the labeled entity span
                if start_token_char >= label_start_char and end_token_char <= label_end_char:
                    labels[token_idx] = label2id[label_name]
                    break
            
            char_pos += len(token) + 1 # Account for the space

        processed_data['id'].append(str(i))
        processed_data['tokens'].append(tokens)
        processed_data['ner_tags'].append(labels)

    return Dataset.from_dict(processed_data), label_list, id2label, label2id

# --- Main Script ---

# FIX 1: Load the CORRECT file exported from Label Studio 
result = load_dataset_from_json("labeled_data.json")

if result is None:
    # Exit if the data file wasn't found
    exit()

dataset, label_list, id2label, label2id = result

MODEL_CHECKPOINT = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Fine Tune Model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT, 
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    output_dir="my_ner_model", # Save to the final model directory
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    push_to_hub=False,
    # No need for remove_unused_columns=False if the data is pre-processed correctly
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Split dataset into training and evaluation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    # The tokenizer is passed via the data_collator, so this is not needed
    # tokenizer=tokenizer, 
)

print("Starting training...")
trainer.train()
print("Training complete.")

print("Saving model and tokenizer...")
trainer.save_model("./my_ner_model")
tokenizer.save_pretrained("./my_ner_model")
print("Model and tokenizer saved to ./my_ner_model")