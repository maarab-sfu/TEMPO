from huggingface_hub import hf_hub_url, get_token
import requests

REPO_ID = "chengyenhsieh/TAO-Amodal"
FILENAME = "frames/val/Charades.zip"

url = hf_hub_url(
    repo_id=REPO_ID,
    filename=FILENAME,
    repo_type="dataset",
)

headers = {"Range": "bytes=0-1023"}

token = get_token()
if token:
    headers["Authorization"] = f"Bearer {token}"

response = requests.get(
    url,
    headers=headers,
    stream=True,
    timeout=60,
    allow_redirects=True,
)

try:
    print("HTTP status:", response.status_code)
    print("Accept-Ranges:", response.headers.get("Accept-Ranges"))
    print("Content-Range:", response.headers.get("Content-Range"))
    print("Content-Length:", response.headers.get("Content-Length"))
finally:
    response.close()