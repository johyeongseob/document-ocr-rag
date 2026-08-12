# OCR

This module runs pretrained Korean PP-OCRv5 text detection and recognition and
provides diagnostic evaluation for word-level polygon annotations.

![PaddleOCR results](../../assets/ocr_demo.gif)

## Evaluation summary

The AI Hub lightweight sample contains 180 financial document images.

| Metric | Result |
|---|---:|
| Word detection coverage recall | 99.74% |
| Whitespace-normalized line exact match | 73.47% |
| **Whitespace-normalized CER** | **1.99%** |

**Whitespace-normalized CER** is the proportion of character insertions, deletions,
and substitutions required after spaces are removed from both the reference and
OCR output. Lower values indicate better text recognition.

## Model configuration

The shared configuration in `src/ocr/utils.py` initializes PaddleOCR with
`lang="korean"` and `ocr_version="PP-OCRv5"`.

| Stage | Pretrained model | Purpose |
|---|---|---|
| Detection | `PP-OCRv5_server_det` | Locate text regions |
| Recognition | `korean_PP-OCRv5_mobile_rec` | Recognize Korean, English, and numeric text |

No manual model download is required. PaddleOCR downloads missing weights on
the first inference and reuses the cached files afterward. On Windows, the
default cache is typically:

```text
C:\Users\<username>\.paddlex\official_models\
├── PP-OCRv5_server_det\
└── korean_PP-OCRv5_mobile_rec\
```

The project uses CPU inference with oneDNN disabled for Windows compatibility.
Document orientation classification, unwarping, and text-line orientation are
also disabled.

## Dataset

The evaluation uses the lightweight sample of AI Hub's
[Financial Industry-Specific Document OCR Data](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=632).
Download the sample from AI Hub and arrange it as follows:

```text
data/
└── financial_document_OCR_dataset_sample/
    ├── annotations/
    │   ├── bank_00001.json
    │   └── ...
    └── images/
        ├── bank_00001.jpg
        └── ...
```

The images and annotations are excluded from this repository and are not
redistributed. Follow the AI Hub terms of use.

## Single-image inference

Run raw OCR inference without comparing the result with an annotation:

```powershell
python -m src.ocr.infer `
  .\data\financial_document_OCR_dataset_sample\images\bank_00001.jpg `
  --output .\outputs\bank_00001
```

Each prediction contains recognized text, confidence, and a four-point polygon.
The command writes:

```text
outputs/bank_00001/
├── result.json
└── bank_00001_ocr_res_img.jpg
```

| Option | Default | Description |
|---|---:|---|
| `--threshold` | `0.5` | Minimum recognition confidence |
| `--det-limit-side-len` | `1280` | Maximum long side used by text detection |
| `--output` | `outputs` | Output directory |

Reducing `--det-limit-side-len` speeds up CPU inference but may miss small text.

## Dataset evaluation

Evaluate one pair, five pairs, or the full lightweight sample:

```powershell
python -m src.ocr.evaluate --limit 1
python -m src.ocr.evaluate --limit 5 --det-limit-side-len 960
python -m src.ocr.evaluate
```

Evaluate a specific document:

```powershell
python -m src.ocr.evaluate --image-name bank_00095.jpg
```

High-resolution, text-dense documents are slow on CPU. Start with one or a few
images before running the entire sample. The detailed report is saved under
`outputs/`.

| Option | Default | Description |
|---|---:|---|
| `--coverage-threshold` | `0.5` | Minimum fraction of a GT word covered by a prediction |
| `--iou-threshold` | `0.5` | Legacy one-to-one polygon IoU threshold |
| `--score-threshold` | `0.5` | Minimum recognition confidence |
| `--det-limit-side-len` | `1280` | Maximum long side used by text detection |

## Evaluation method

For each document, the evaluator:

1. Applies JPEG EXIF orientation so pixels and annotation coordinates align.
2. Reads the annotation ID, source index, `type`, `text`, and four-point
   `points` polygon. Empty text and `type=0` ignore regions are excluded;
   printed (`type=1`) and handwritten (`type=2`) regions are retained.
3. Runs PaddleOCR and extracts text, confidence, and `rec_polys`.
4. Calculates polygon IoU for every ground-truth/prediction pair and retains
   one-to-one IoU matching as a legacy diagnostic.
5. Measures how much of every word polygon is covered by a prediction and assigns
   each word to its best prediction at coverage 0.5.
6. Allows a prediction line to contain multiple ground-truth words, sorts them
   left to right, and constructs a reference line.
7. Reports overall, printed, and handwritten word-coverage recall.
8. Calculates raw and whitespace-normalized CER and exact match.

Axis-aligned bounding boxes are not used. AI Hub annotations are generally
word-level, while PaddleOCR commonly returns one polygon for a text line.
Coverage-based one-to-many assignment better reflects this mismatch than
one-to-one IoU, but remains a project-specific diagnostic rather than an
official benchmark protocol.

`src/ocr/evaluate.py` explicitly applies EXIF orientation because some sample
JPEG files store landscape pixels while annotations use rotated portrait
coordinates.
