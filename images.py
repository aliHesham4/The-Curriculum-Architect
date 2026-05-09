import re
import io
import numpy as np
from PIL import Image
import pytesseract
import torch
from config import doc, blip_processor, blip_model


# ── Heuristic pre-filter (runs before any heavy model) ───────────────────────
def _is_likely_educational(pil_image, min_size=150, max_aspect=6.0, variance_threshold=100):
    """
    Returns False for images that are almost certainly decorative or irrelevant:
      - Too small (banners, icons, bullets)
      - Extreme aspect ratio (horizontal rule, thin dividers)
      - Near-uniform color (blank, solid-fill decorations)
    """
    w, h = pil_image.size

    # Too small on either dimension
    if w < min_size or h < min_size:
        return False

    # Extreme aspect ratio — likely a divider or banner strip
    aspect = max(w, h) / min(w, h)
    if aspect > max_aspect:
        return False

    # Near-uniform image — blank or solid color decoration
    gray = np.array(pil_image.convert("L"), dtype=float)
    if gray.var() < variance_threshold:
        return False

    return True


def describe_image_locally(image_bytes):
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ── Pre-filter before any heavy processing ────────────────────────────
        if not _is_likely_educational(pil_image):
            return None     # None = caller skips this image entirely

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

        return " | ".join(results) if results else None

    except Exception as e:
        return f"[Image analysis failed: {e}]"


def extract_page_images(page, page_number, max_images_per_page=3):
    """
    max_images_per_page — hard cap so a page with 20 bullet icons
    doesn't trigger 20 BLIP calls.
    """
    image_descriptions = []
    image_list = page.get_images(full=True)
    if not image_list:
        return ""

    analyzed = 0

    for img_index, img in enumerate(image_list):
        if analyzed >= max_images_per_page:
            print(f"  → Page {page_number + 1}: reached cap of {max_images_per_page} images, skipping rest.")
            break

        xref, width, height = img[0], img[2], img[3]

        # Fast dimension check before even extracting bytes
        if width < 150 or height < 150:
            # print(f"  → Skipping image {img_index + 1} on page {page_number + 1} — too small ({width}x{height}px)")
            continue

        try:
            base_image  = doc.extract_image(xref)
            image_bytes = base_image["image"]

            print(f"  → Analyzing image {img_index + 1} on page {page_number + 1} ({width}x{height}px)...")
            description = describe_image_locally(image_bytes)

            if description is None:
                print(f"     ↳ Skipped — likely decorative.")
                continue

            image_descriptions.append(f"[IMAGE {img_index + 1}: {description}]")
            analyzed += 1

        except Exception as e:
            image_descriptions.append(f"[IMAGE {img_index + 1}: Could not extract — {e}]")

    return "\n" + "\n".join(image_descriptions) + "\n" if image_descriptions else ""