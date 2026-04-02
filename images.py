import re
import io
from PIL import Image
import pytesseract
import torch
from config import doc, blip_processor, blip_model


def describe_image_locally(image_bytes):
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = []

        ocr_text = pytesseract.image_to_string(pil_image).strip()
        ocr_text = re.sub(r'\s+', ' ', ocr_text)
        if len(ocr_text) > 5:
            results.append(f"Text in image: {ocr_text}")

        inputs = blip_processor(pil_image, return_tensors="pt")
        with torch.no_grad():
            output = blip_model.generate(**inputs, max_new_tokens=60)
        caption = blip_processor.decode(output[0], skip_special_tokens=True).strip()
        if caption:
            results.append(f"Visual: {caption}")

        return " | ".join(results) if results else "[Image: no content detected]"
    except Exception as e:
        return f"[Image analysis failed: {e}]"


def extract_page_images(page, page_number):
    image_descriptions = []
    image_list = page.get_images(full=True)
    if not image_list:
        return ""

    for img_index, img in enumerate(image_list):
        xref, width, height = img[0], img[2], img[3]
        if width < 100 or height < 100:
            continue
        try:
            base_image  = doc.extract_image(xref)
            image_bytes = base_image["image"]
            print(f"  → Analyzing image {img_index + 1} on page {page_number + 1} ({width}x{height}px)...")
            description = describe_image_locally(image_bytes)
            image_descriptions.append(f"[IMAGE {img_index + 1}: {description}]")
        except Exception as e:
            image_descriptions.append(f"[IMAGE {img_index + 1}: Could not extract — {e}]")

    return "\n" + "\n".join(image_descriptions) + "\n" if image_descriptions else ""