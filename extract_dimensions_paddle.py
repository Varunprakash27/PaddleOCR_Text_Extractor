import os
import re
import cv2
import gc
import traceback
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

def draw_ocr(image, boxes, txts, scores, drop_score=0.5):
    if isinstance(image, Image.Image):
        image = np.array(image)
    img = image.copy()

    for box, txt, score in zip(boxes, txts, scores):
        if score < drop_score or not txt.strip():
            continue

        box = np.array(box).astype(np.int32).reshape(-1, 2)
        center = box.mean(axis=0)
        shrink_factor = 0.96
        box = ((box - center) * shrink_factor + center).astype(np.int32)

        cv2.polylines(img, [box], isClosed=True, color=(0, 255, 0), thickness=1)

        x, y = box[0]
        y = max(y - 12, 5)
        x = max(x, 5)

        ((text_w, text_h), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x, y - text_h - 4), (x + text_w, y + 4), (0, 255, 0), -1)
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return img

INPUT_FOLDER = input("Enter path to input image folder: ").strip()
OUTPUT_FOLDER = input("Enter path to output folder: ").strip()
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ocr = PaddleOCR(use_angle_cls=True, lang='en')

dimension_regex = re.compile(r"(Ø\s*\d+(\.\d+)?|ø\s*\d+(\.\d+)?|R\s*\d+(\.\d+)?|±\s*\d+(\.\d+)?|\d+(\.\d+)?)")

def extract_dimensions_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    result = ocr.ocr(image)

    dims_detected = []
    boxes, txts, scores = [], [], []

    if isinstance(result, list):
        if isinstance(result[0], dict):
            rec_texts = result[0].get("rec_texts", [])
            rec_scores = result[0].get("rec_scores", [])
            rec_polys = result[0].get("rec_polys", [])
            for txt, score, box in zip(rec_texts, rec_scores, rec_polys):
                txts.append(txt)
                scores.append(score)
                boxes.append(box)
                if dimension_regex.search(txt):
                    dims_detected.append(txt)
        else:
            for line in result:
                for res in line:
                    box, (txt, score) = res
                    boxes.append(box)
                    txts.append(txt)
                    scores.append(score)
                    if dimension_regex.search(txt):
                        dims_detected.append(txt)
    else:
        print("Unexpected OCR result format")

    return image, boxes, txts, scores, dims_detected

if __name__ == "__main__":
    for filename in sorted(os.listdir(INPUT_FOLDER)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                print(f"\nProcessing: {filename}")
                path = os.path.join(INPUT_FOLDER, filename)

                img, boxes, txts, scores, dims = extract_dimensions_from_image(path)

                if dims:
                    print(f"Found {len(dims)} dimension(s): {sorted(dims)}")
                else:
                    print("No dimensions found.")

                annotated = draw_ocr(img, boxes, txts, scores)

                save_img = os.path.join(OUTPUT_FOLDER, f"annotated_{filename}")
                save_txt = os.path.join(OUTPUT_FOLDER, f"text_{filename}.txt")

                cv2.imwrite(save_img, annotated)

                with open(save_txt, "w", encoding="utf-8") as f:
                    for dim in dims:
                        f.write(dim + "\n")

                del img, boxes, txts, scores, dims, annotated
                gc.collect()

            except Exception as e:
                print(f"Skipped {filename} due to error:\n{traceback.format_exc()}")

    print("\nDONE. Check the output_dimensions folder.")
