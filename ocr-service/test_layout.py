# test_layout.py
import os, io
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

# read endpoint and key from environment variables
ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")
KEY = os.getenv("AZURE_FORM_KEY")
if not ENDPOINT or not KEY:
    raise SystemExit("Set AZURE_FORM_ENDPOINT and AZURE_FORM_KEY in env")

client = DocumentAnalysisClient(ENDPOINT, AzureKeyCredential(KEY))

# <- update this path if your image is in another location
path = "/Users/iamdheeraj/Downloads/WhatsApp Image 2025-12-12 at 15.54.52.jpeg"
with open(path, "rb") as f:
    data = f.read()

print("Calling prebuilt layout (document) model...")
buf = io.BytesIO(data)
poller = client.begin_analyze_document("prebuilt-document", document=buf)
res = poller.result()

for p in res.pages:
    print("PAGE", p.page_number)
    if getattr(p, "words", None):
        for w in p.words[:40]:  # show first 40 words
            bb = getattr(w, "bounding_box", None)
            if bb:
                print("WORD:", w.content, "bbox:", [bb[0].x, bb[0].y, bb[2].x, bb[2].y])
            else:
                print("WORD:", w.content, "bbox: None")
    else:
        print("No words/words missing on this page")
