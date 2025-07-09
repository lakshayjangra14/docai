import os
import json
from transformers import pipeline
from ocr_pipeline import extract_text_from_document # We import the function from our first script

# --- 1. Load the fine-tuned model from the local directory ---
# The 'pipeline' function from Hugging Face is the easiest way to use a trained model for inference.
print("Loading custom NER model...")

# Make sure the path './my_ner_model' matches the directory where your model was saved.
MODEL_PATH = "./my_ner_model" 

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model directory not found at '{MODEL_PATH}'")
    print("Please make sure you have run the training script and the model was saved correctly.")
    exit()

# 'ner' is the task. aggregation_strategy="simple" groups word pieces (e.g., "Corp", "##oration")
# into a single entity ("Corporation").
ner_pipeline = pipeline(
    "ner",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    aggregation_strategy="simple"
)
print("Model loaded successfully.")


# --- 2. Define the main extraction function ---
def extract_entities_from_file(file_path):
    """
    Takes a file path, performs OCR, runs it through the NER model,
    and returns a clean dictionary of extracted entities.
    """
    # Step A: Get raw text from the document using our existing OCR function
    print(f"Step 1: Running OCR on '{file_path}'...")
    raw_text = extract_text_from_document(file_path)
    
    if "Error:" in raw_text or not raw_text.strip():
        print("OCR failed or returned no text.")
        return {"error": raw_text}
    
    print("OCR completed. Text extracted.")
    # In inference.py

# ... (keep the top part of the file the same)

def extract_entities_from_file(file_path):
    # ...
    print(f"Step 1: Running OCR on '{file_path}'...")
    raw_text = extract_text_from_document(file_path)
    
    # <<<--- ADD THIS BLOCK TO DEBUG --- >>>
    print("\n" + "="*20 + " OCR OUTPUT " + "="*20)
    print(raw_text)
    print("="*52 + "\n")
    # <<<--- END OF DEBUG BLOCK --- >>>

    if "Error:" in raw_text or not raw_text.strip():
        # ...
    # ... (the rest of the file stays the same)
    
    # Step B: Run the text through our NER pipeline
        print("Step 2: Running NER model on the extracted text...")
    ner_results = ner_pipeline(raw_text)
    print("NER processing complete.")

    # Step C: Process the results into a clean dictionary
    # The output from the pipeline looks like:
    # [{'entity_group': 'COMPANY_NAME', 'score': 0.99, 'word': 'Acme Inc'}, ...]
    print("Step 3: Formatting results...")
    extracted_data = {}
    for entity in ner_results:
        entity_group = entity['entity_group']
        entity_value = entity['word'].strip()

        # If the entity is already in our dictionary, append the new value.
        # This can be useful for multi-line addresses, but for single-value entities,
        # you might want to just keep the one with the highest score.
        # For simplicity, we'll combine them.
        if entity_group in extracted_data:
            extracted_data[entity_group] += " " + entity_value
        else:
            extracted_data[entity_group] = entity_value

    return extracted_data


# --- This is the main execution block to test the full pipeline ---
if __name__ == "__main__":
    # IMPORTANT: Change this to a file that the model has NEVER seen before.
    # It should be a PDF or image file located in your 'data' folder.
    test_file_name = "invoice_02.jpeg" # <--- CHANGE THIS FILENAME
    
    test_file_path = os.path.join("data", test_file_name) 
    
    if os.path.exists(test_file_path):
        print("-" * 50)
        print(f"Running full extraction pipeline on: {test_file_path}")
        print("-" * 50)
        
        # Call our main function
        final_data = extract_entities_from_file(test_file_path)
        
        # Pretty-print the final JSON output
        print("\n--- EXTRACTION COMPLETE ---")
        print("Final structured data:")
        print(json.dumps(final_data, indent=2))
        print("-" * 50)
    else:
        print(f"\nERROR: Test file not found at '{test_file_path}'.")
        print("Please make sure the file exists in the 'data' folder and the filename is correct.")