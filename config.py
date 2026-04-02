import os
import fitz
from dotenv import load_dotenv
from groq import Groq
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Paths ──
PDF_PATH     = r"D:\GUC\The Curriculum Architect\Dataset\Math Curriculum For Children.pdf"
OUTPUT_FILE  = r"D:\GUC\The Curriculum Architect\Python Files\extracted_text.txt"
CONCEPTS_FILE = r"D:\GUC\The Curriculum Architect\Python Files\concepts.json"

# ── PDF ──
doc          = fitz.open(PDF_PATH)
total_pages  = len(doc)

# ── Models ──
print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.eval()

print("Loading embedder + KeyBERT...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedder)

# ── Groq client ──
load_dotenv()
print("Loading LLaMA API...")
groq_client = Groq(api_key=os.getenv("GROQ_KEY"))
response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Say hello"}]
)
print(response.choices[0].message.content)