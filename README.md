# PaddleOCR Text Extractor

This script extracts dimensional text (like Ø, R, ±) from images using PaddleOCR, highlights the results, and saves both annotated images and detected text files.

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Varunprakash27/PaddleOCR_Text_Extractor.git
cd PaddleOCR_Text_Extractor
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
python extract_dimensions_paddle.py
```

You will be prompted to enter:
- The **input folder path** (where your images are stored)
- The **output folder path** (where results will be saved)

---

## 📂 Output

- Annotated image with OCR boxes: `annotated_<filename>.jpg`
- Extracted text file: `text_<filename>.txt`

---
