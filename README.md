# Smart Invoice HCR

AI-powered invoice OCR system with handwritten character recognition, human-in-the-loop corrections, and batch processing.

## Features

- **Azure Document Intelligence Integration** - Custom trained model for invoice processing
- **Line Items Extraction** - Automatic table parsing with proper column ordering
- **Batch Processing** - Upload multiple invoices with next/prev navigation
- **Human Correction Loop** - Edit fields and line items with visual feedback
- **Draft & Export** - Save progress and export to CSV
- **Correction Tracking** - SQLite storage with snippet generation

## Setup

1. **Install dependencies:**
   ```bash
   cd ocr-service
   pip install -r requirements.txt
   ```

2. **Configure Azure credentials:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Azure credentials
   ```

3. **Run the server:**
   ```bash
   python -m uvicorn app:app --host 127.0.0.1 --port 8000
   ```

4. **Open UI:**
   ```
   http://127.0.0.1:8000/ui
   ```

## Environment Variables

Create a `.env` file in `ocr-service/` directory:

```env
AZURE_FORM_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_FORM_KEY=your_azure_key_here
AZURE_MODEL_ID=your_custom_model_id
```

## API Endpoints

- `POST /analyze` - Analyze invoice image
- `GET /annotated/{name}` - Serve annotated images
- `GET /snippets/{name}` - Serve field snippets
- `POST /corrections` - Save field corrections
- `GET /corrections` - List corrections
- `POST /draft` - Save batch draft
- `POST /export-csv` - Export batch to CSV

## Architecture

```
ocr-service/
├── app.py              # FastAPI backend
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── ui/
│   └── index.html      # Web UI
└── storage/
    ├── originals/      # Uploaded images
    ├── annotated/      # Images with bounding boxes
    └── snippets/       # Cropped field regions
```

## Workflow

1. Upload invoices (single or batch)
2. Azure analyzes with custom model
3. Review extracted fields and line items
4. Correct errors inline
5. Save corrections (tracks changes)
6. Export to CSV or save as draft

## Tech Stack

- **Backend**: FastAPI, Python 3.13
- **OCR**: Azure Document Intelligence (Custom Model)
- **Storage**: SQLite
- **Frontend**: Vanilla HTML/JS
- **Image Processing**: OpenCV, Pillow

## License

MIT
