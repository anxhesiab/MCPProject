"""mcp_system.scanner.metadata_store
===================================

Write/read side-car `.meta.json` files that live next to data blobs in
S3 or Azure Blob Storage.

Design goals:
- Keep the public `FileMeta` dataclass unchanged so callers stay compatible.
- Route to S3 vs Azure based on the first segment of the path (bucket/container).
- Support `TEST_MODE=1` for local `.sidecar_cache` usage in offline or pytest runs.
"""
from __future__ import annotations
import json
from typing import Optional
import datetime as _dt
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

logger = logging.getLogger(__name__)
ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
TEST_MODE = os.getenv("TEST_MODE", "0") == "1"
@dataclass(slots=True)
class FileMeta:
    path: str
    cloud: str
    title: str
    description: str
    mime_type: str
    size: int
    last_scanned: _dt.datetime

    def to_json(self) -> str:
        d = asdict(self)
        d["last_scanned"] = self.last_scanned.strftime(ISO_FMT)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "FileMeta":
        d = json.loads(raw)
        d["last_scanned"] = _dt.datetime.strptime(d["last_scanned"], ISO_FMT)
        return cls(**d)


class SidecarStore:
    SUFFIX = ".meta.json"
    _local_root = Path(tempfile.gettempdir()) / ".sidecar_cache"
    _local_root.mkdir(exist_ok=True, parents=True)

    def write_meta(self, meta: FileMeta, *, cloud: str | None = None) -> None:
        """Persist `meta` next to its blob (overwrite-safe)."""
        cloud = cloud or meta.cloud
        key = f"{meta.path}{self.SUFFIX}"
        self._save(key, meta.to_json().encode("utf-8"), cloud=cloud)
        logger.debug("Sidecar written → %s", key)

    def get_meta(self, path: str, cloud: str) -> Optional[FileMeta]:
        key = f"{path}{self.SUFFIX}"
        try:
            raw = self._load(key)  # bytes
            return FileMeta.from_json(raw.decode("utf-8"))
        except Exception as e:
            logger.debug(f"Metadata not found for {path}: {e}")
            return None

    def read_all(self, prefix: str = "") -> List[FileMeta]:
        """Read all sidecar metas under `prefix`."""
        metas: List[FileMeta] = []
        for key in self._iter_keys(prefix):
            try:
                raw = self._load(key)
                metas.append(FileMeta.from_json(raw.decode("utf-8")))
            except Exception as exc:
                logger.warning("Could not parse sidecar %s: %s", key, exc)
        return metas

    # inside class SidecarStore

    def _save(self, key: str, data: bytes, *, cloud: str) -> None:
        # Split bucket/container and blob
        parts = key.split("/", 1)
        if len(parts) < 2:
            raise ValueError(f"Invalid key: {key}")

        container_or_bucket, blob = parts

        # Determine cloud type
        if self._is_s3_bucket(container_or_bucket):
            from mcp_system.aws_s3.client_s3 import get_s3_client
            client = get_s3_client()
            client.put_object(Bucket=container_or_bucket, Key=blob, Body=data)
        else:
            from mcp_system.azure_blob.client_azure import get_blob_client
            client = get_blob_client(f"{container_or_bucket}/{blob}")
            client.upload_blob(data, overwrite=True)

    def _load(self, key: str) -> bytes:
        if TEST_MODE:
            local_file = self._local_root / key.replace("/", "__")
            return local_file.read_bytes()

        # Split bucket/container and blob
        parts = key.split("/", 1)
        if len(parts) < 2:
            raise ValueError(f"Invalid key format: {key}")

        container_or_bucket, blob = parts

        # Determine cloud type
        if self._is_s3_bucket(container_or_bucket):
            from mcp_system.aws_s3.client_s3 import get_s3_client
            client = get_s3_client()
            return client.get_object(Bucket=container_or_bucket, Key=blob)["Body"].read()
        else:
            from mcp_system.azure_blob.client_azure import get_blob_client
            client = get_blob_client(f"{container_or_bucket}/{blob}")
            return client.download_blob().readall()

    def _iter_keys(self, prefix: str) -> Iterable[str]:
        if TEST_MODE:
            for p in self._local_root.glob(f"*__*{self.SUFFIX}"):
                path = p.name.replace("__", "/")
                if path.startswith(prefix):
                    yield path
            return

        # Handle S3 bucket
        s3_bucket = os.getenv("S3_BUCKET")
        if s3_bucket:
            yield from self._iter_s3(s3_bucket, prefix)

        # Handle Azure containers
        az_containers = [os.getenv("AZURE_CONTAINER"), "feedback"]
        for container in az_containers:
            if container:
                yield from self._iter_azure(container, prefix)

    @staticmethod
    def _iter_s3(bucket: str, prefix: str) -> Iterable[str]:
        from mcp_system.aws_s3.client_s3 import get_s3_client
        client = get_s3_client()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for itm in page.get("Contents", []):
                key = itm.get("Key")
                if key and key.endswith(SidecarStore.SUFFIX):
                    yield f"{bucket}/{key}"

    @staticmethod
    def _iter_azure(container: str, prefix: str) -> Iterable[str]:
        # Use BlobServiceClient to list sidecars in the container
        from azure.storage.blob import BlobServiceClient
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        svc = BlobServiceClient.from_connection_string(conn_str)
        container_client = svc.get_container_client(container)
        for blob in container_client.list_blobs(name_starts_with=prefix):
            if blob.name.endswith(SidecarStore.SUFFIX):
                yield f"{container}/{blob.name}"

    @staticmethod
    def _is_s3_bucket(name: str) -> bool:
        """Determine if a path belongs to S3 or Azure"""
        # Explicit list of known S3 buckets
        s3_buckets = [os.getenv("S3_BUCKET"), "my-api-bucket-100757077"]

        # Explicit list of known Azure containers
        az_containers = [os.getenv("AZURE_CONTAINER"), "feedback"]

        # First check if we recognize the name
        if name in s3_buckets:
            return True
        if name in az_containers:
            return False

        # Fallback to naming pattern detection
        if "." in name:  # Azure containers often look like DNS names
            return False
        return True  # Default to S3 for simple names


__all__ = ["FileMeta", "SidecarStore"]
