# Pretrained PaddleOCR inference lab

This lab runs pretrained Korean text detection and recognition without training.

```powershell
.\ocr\Scripts\python.exe .\make_sample.py
.\ocr\Scripts\python.exe .\ocr_infer.py .\sample_document.png
```

To use your own image:

```powershell
.\ocr\Scripts\python.exe .\ocr_infer.py C:\path\to\document.jpg
```

The first OCR run downloads pretrained PP-OCR weights. Output includes recognized
text and confidence in the terminal, polygons in `outputs/result.json`, and an
annotated result image in `outputs/`.
