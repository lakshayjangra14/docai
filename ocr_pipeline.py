import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import os
import json

# --- This function is the same as before ---
def extract_text_from_document(file_path):
    """Extracts text from a PDF or image file."""
    if not os.path.exists(file_path):
        return "Error: File not found."

    file_extension = os.path.splitext(file_path)[1].lower()
    extracted_text = ""

    if file_extension == ".pdf":
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                extracted_text += pytesseract.image_to_string(img) + "\n\n"
            doc.close()
        except Exception as e:
            return f"Error processing PDF: {e}"
    elif file_extension in [".png", ".jpg", ".jpeg", ".tiff"]:
        try:
            img = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(img)
        except Exception as e:
            return f"Error processing image: {e}"
    else:
        return "Error: Unsupported file type."

    return extracted_text

# --- Main script logic ---
if __name__ == "__main__":
    DATA_DIR = "data"
    OUTPUT_JSON_FILE = "ocr_output_for_label_studio.json"

    # This list will hold one dictionary for each document
    tasks_for_label_studio = []

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and add your documents.")
    else:
        # Loop through every file in the 'data' directory
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.isfile(file_path):
                print(f"Processing: {filename}...")

                # Get the raw text from the document
                raw_text = extract_text_from_document(file_path)

                # Skip if there was an error
                if "Error:" in raw_text:
                    print(f"  -> Skipped due to error: {raw_text}")
                    continue

                # Create the JSON structure that Label Studio expects for each task
                # The key 'text' must match the <Text name="text" ...> tag in the Labeling Interface
                task = {
                    "data": {
                        "text": raw_text
                    }
                }
                tasks_for_label_studio.append(task)
                print(f"  -> Added to task list.")

        # Write the entire list of tasks to a single JSON file
        with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks_for_label_studio, f, indent=2)

        print(f"\nSuccessfully created '{OUTPUT_JSON_FILE}' with {len(tasks_for_label_studio)} tasks.")
        print("You can now import this file into Label Studio.")