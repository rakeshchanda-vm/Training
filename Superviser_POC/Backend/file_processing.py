import json
import pymupdf4llm
import os
import fitz
from prompts import OCR_MODEL_PROMPT
from config import OCR_MODEL, PATH_PROCESSED, PATH_STRUCTURED, PATH_RAW
# import ollama

def save_as_markdown(path:str, text:str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Created {path}.md")

def process_ocr_pdf(path:str):
    markdown = ""
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        
        img = f"/tmp/page_{i}.png"
        page.get_pixmap(dpi=150).save(img)

        MODEL = "minicpm-v:latest"
        response = ollama.chat(
            model=OCR_MODEL,
            messages=[{ "role": "user",
                        "content": OCR_MODEL_PROMPT,
                        "images": [img]
                }])

        markdown += f"\n\n## Page {i+1}\n\n"
        markdown += response["message"]["content"]
        os.remove(img)
    return markdown

def process_files():

    source_path = PATH_RAW
    files = [name for name in os.listdir(source_path)
            if os.path.isfile(os.path.join(source_path, name))]

    print(files)

    for file in files:
        PDF_FILE = f"{source_path}/{file}"
        print("Extracting PDF...",PDF_FILE)
        markdown_text = pymupdf4llm.to_markdown(PDF_FILE)
        process_file_path = f"{PATH_PROCESSED}/{file}.md"

        if len(markdown_text) < 50:
            markdown_text = process_ocr_pdf(PDF_FILE)
            save_as_markdown(process_file_path,markdown_text)
        else:
            save_as_markdown(process_file_path,markdown_text)

if __name__=="__main__":
    process_files()