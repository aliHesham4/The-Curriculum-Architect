import os
import fitz
from dotenv import load_dotenv
from groq import Groq
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract
import google.generativeai as genai


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Paths ──
PDF_PATH     = r"D:\GUC\The Curriculum Architect\Dataset\Math Curriculum For Children.pdf"
OUTPUT_FILE  = r"D:\GUC\The Curriculum Architect\Python Files\Debugging\extracted_text.txt"
CONCEPTS_FILE = r"D:\GUC\The Curriculum Architect\Python Files\Debugging\concepts_found_in_document.json"
CLEAN_RELATIONS_FILE = r"D:\GUC\The Curriculum Architect\Python Files\Debugging\clean_relations.json"
DETERMINSITIC_VALIDATOR_OUTPUT = r"D:\GUC\The Curriculum Architect\Python Files\Debugging\deterministic_validator_output.json"

# ── PDF ──
doc          = fitz.open(PDF_PATH)
total_pages  = len(doc)

# ── Models ──
print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",local_files_only=True)
blip_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.eval()

print("Loading embedder + KeyBERT...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedder)

# ── Groq client ──
load_dotenv()
print("Loading Gemini API...")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")
# response = model.generate_content("Say hello")

# print(response.text)