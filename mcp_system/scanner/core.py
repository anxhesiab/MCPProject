"""mcp_system.scanner.core
===================================

Functions to scan objects in S3 / Azure and extract previews for LLM-based side-car generation.
"""
from __future__ import annotations
import os
import logging
from typing import Tuple
import boto3
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)
AZ_CONTAINER = os.getenv("AZURE_CONTAINER")
AZ_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_svc = BlobServiceClient.from_connection_string(AZ_CONN)


def get_blob_client(path: str):
    """Return a BlobClient for Azure Blob Storage path"""
    if "/" in path:
        container, blob = path.split("/", 1)
    else:
        container, blob = AZ_CONTAINER, path
    return blob_svc.get_blob_client(container, blob)

def get_blob_properties(path: str):
    """Get Azure blob properties including size and content type"""
    if "/" in path:
        container, blob = path.split("/", 1)
    else:
        container, blob = AZ_CONTAINER, path
    client = blob_svc.get_blob_client(container, blob)
    return client.get_blob_properties()

# Constants for head download
_DEFAULT_HEAD_SIZE = int(os.getenv("SCAN_HEAD_SIZE", 65536))
def scan_object(path: str, cloud: str) -> None:
    """
    Download the first chunk of the object, ask the LLM to describe it,
    and persist the description into side-car metadata.
    """
    raw_head, sample_bytes = _download_head(path, cloud)
    from anthropic import Anthropic
    client = Anthropic()
    prompt = f"""
You are helping index a dataset for search and discovery.
Below is a preview sample from a file stored in cloud storage:
File path: {path}
Cloud: {cloud.upper()}
------------------------
{raw_head.strip()}
------------------------
Give a clear one-line factual description of the file contents. Describe what kind of data is inside and how it could be used. Format your response like this:
<description>
ONLY return the description text — no filename, no prefix, no extra commentary.
""".strip()
    try:
        msg = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        desc = "".join(chunk.text for chunk in msg.content).strip()
        if len(desc.split()) > 2:
            from mcp_system.scanner.metadata_store import FileMeta, SidecarStore
            from datetime import datetime
            import os

            # Get file properties
            size = 0
            mime_type = "application/octet-stream"

            try:
                if cloud == "s3":
                    # Split path into bucket and key
                    if "/" in path:
                        bucket, key = path.split("/", 1)
                    else:
                        bucket = os.getenv("S3_BUCKET")
                        key = path

                    # Get S3 object properties
                    s3 = boto3.client("s3")
                    response = s3.head_object(Bucket=bucket, Key=key)
                    size = response["ContentLength"]
                    mime_type = response.get("ContentType", "application/octet-stream")

                else:  # Azure
                    client = get_blob_client(path)
                    props = client.get_blob_properties()
                    size = props.size
                    mime_type = props.content_settings.content_type or "application/octet-stream"

            except Exception as e:
                print(f"⚠️ Failed to get properties for {path}: {e}")

            # Create and save metadata
            meta = FileMeta(
                path=path,
                cloud=cloud,
                title=os.path.basename(path),
                description=desc,
                mime_type=mime_type,
                size=size,
                last_scanned=datetime.now()
            )

            store = SidecarStore()
            store.write_meta(meta, cloud=cloud)
            print(f"✅ Saved cloud meta for {path}")
    except Exception as e:
        print(f"❌ Failed to scan {path}: {e}")


def _download_head(path: str, cloud: str) -> Tuple[str, bytes]:
    """
    Download and return (utf-8 decoded head text, raw bytes).
    """
    if cloud == "s3":
        # If path contains "/", treat it as "bucket/key"
        if "/" in path:
            bucket, key = path.split("/", 1)
        else:  # called with bare key → use env bucket
            bucket = os.getenv("S3_BUCKET")
            key = path
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{_DEFAULT_HEAD_SIZE-1}")
        data = obj["Body"].read()
        text = data.decode("utf-8", errors="ignore")
        return text, data
    else:
        # For Azure, delegate path handling to get_blob_client (handles container logic)
        client = get_blob_client(path)  # works with both "blob.csv" and "container/blob.csv"
        stream = client.download_blob(offset=0, length=_DEFAULT_HEAD_SIZE)
        data = stream.readall()
        text = data.decode("utf-8", errors="ignore")
        return text, data
