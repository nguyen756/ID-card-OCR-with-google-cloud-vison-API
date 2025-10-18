from pathlib import Path
from dotenv import load_dotenv
import os

# Load the .env that sits next to this file (robust even if CWD changes)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# Optional: print to verify the variables are available
print("DOC_AI_PROJECT_ID =", os.getenv("DOC_AI_PROJECT_ID"))
print("DOC_AI_LOCATION   =", os.getenv("DOC_AI_LOCATION"))
print("DOC_AI_PROCESSOR_ID =", os.getenv("DOC_AI_PROCESSOR_ID"))
print("GOOGLE_APPLICATION_CREDENTIALS =", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

# Now it's safe to read (use os.getenv to avoid KeyError)
project_id  = os.getenv("DOC_AI_PROJECT_ID")
location    = os.getenv("DOC_AI_LOCATION", "us")
processor_id= os.getenv("DOC_AI_PROCESSOR_ID")
creds       = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

missing = [k for k,v in {
    "DOC_AI_PROJECT_ID": project_id,
    "DOC_AI_LOCATION": location,
    "DOC_AI_PROCESSOR_ID": processor_id,
    "GOOGLE_APPLICATION_CREDENTIALS": creds
}.items() if not v]
if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

from google.api_core.client_options import ClientOptions
from google.cloud import documentai as docai
from PIL import Image
import io

client = docai.DocumentProcessorServiceClient(
    client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
)
name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

# tiny sanity call
img = Image.new("RGB", (300, 150), "white")
buf = io.BytesIO(); img.save(buf, "PNG")
req = docai.ProcessRequest(
    name=name,
    raw_document=docai.RawDocument(content=buf.getvalue(), mime_type="image/png")
)
resp = client.process_document(request=req)
print("Success. Text length:", len(resp.document.text or ""))
