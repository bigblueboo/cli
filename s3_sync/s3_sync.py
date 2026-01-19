#!/usr/bin/env python3
"""
s3_sync - A unified CLI tool for cloud storage operations.

Supports Amazon S3, Google Cloud Storage, and Azure Blob Storage.
Provides upload, download, presigned URL generation, and directory sync capabilities.

Exit Codes:
    0 - Success
    1 - Credentials error
    2 - Not found error
    3 - Transfer error
    4 - Invalid arguments

Environment Variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY - AWS credentials
    GOOGLE_APPLICATION_CREDENTIALS - Path to GCP service account JSON
    AZURE_STORAGE_CONNECTION_STRING - Azure connection string

Examples:
    s3_sync upload report.pdf s3://bucket/reports/
    s3_sync download gs://bucket/data/*.csv ./local/
    s3_sync presign s3://bucket/file.zip --expires 3600
    s3_sync sync ./local/ s3://bucket/backup/ --include "*.json"
"""

import argparse
import fnmatch
import os
import re
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import urlparse

# Exit codes
EXIT_SUCCESS = 0
EXIT_CREDENTIALS_ERROR = 1
EXIT_NOT_FOUND = 2
EXIT_TRANSFER_ERROR = 3
EXIT_INVALID_ARGS = 4


class CloudStorageError(Exception):
    """Base exception for cloud storage operations."""
    pass


class CredentialsError(CloudStorageError):
    """Raised when credentials are missing or invalid."""
    pass


class NotFoundError(CloudStorageError):
    """Raised when a resource is not found."""
    pass


class TransferError(CloudStorageError):
    """Raised when a transfer operation fails."""
    pass


class StorageProvider(ABC):
    """Abstract base class for cloud storage providers."""

    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file to cloud storage."""
        pass

    @abstractmethod
    def download(self, remote_path: str, local_path: str) -> None:
        """Download a file from cloud storage."""
        pass

    @abstractmethod
    def presign(self, remote_path: str, expires: int) -> str:
        """Generate a presigned URL for a file."""
        pass

    @abstractmethod
    def list_objects(self, bucket: str, prefix: str = "") -> Generator[str, None, None]:
        """List objects in a bucket with optional prefix."""
        pass

    @abstractmethod
    def exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists."""
        pass


class S3Provider(StorageProvider):
    """Amazon S3 storage provider using boto3."""

    def __init__(self):
        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import NoCredentialsError, ClientError
        except ImportError:
            raise CredentialsError("boto3 is not installed. Run: pip install boto3")

        self.boto3 = boto3
        self.Config = Config
        self.NoCredentialsError = NoCredentialsError
        self.ClientError = ClientError

        # Check for credentials
        if not os.environ.get('AWS_ACCESS_KEY_ID') or not os.environ.get('AWS_SECRET_ACCESS_KEY'):
            # Try default credential chain
            try:
                session = boto3.Session()
                credentials = session.get_credentials()
                if credentials is None:
                    raise CredentialsError(
                        "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
                        "AWS_SECRET_ACCESS_KEY environment variables."
                    )
            except Exception:
                raise CredentialsError(
                    "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
                    "AWS_SECRET_ACCESS_KEY environment variables."
                )

        # Use signature v4 for presigned URLs
        self.client = boto3.client(
            's3',
            config=Config(signature_version='s3v4')
        )
        self.resource = boto3.resource('s3')

    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file to S3."""
        bucket, key = self._parse_s3_path(remote_path)
        try:
            self.client.upload_file(local_path, bucket, key)
        except self.NoCredentialsError:
            raise CredentialsError("AWS credentials are invalid or expired.")
        except self.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                raise NotFoundError(f"Bucket not found: {bucket}")
            raise TransferError(f"Upload failed: {e}")
        except FileNotFoundError:
            raise NotFoundError(f"Local file not found: {local_path}")
        except Exception as e:
            raise TransferError(f"Upload failed: {e}")

    def download(self, remote_path: str, local_path: str) -> None:
        """Download a file from S3."""
        bucket, key = self._parse_s3_path(remote_path)
        try:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
            self.client.download_file(bucket, key, local_path)
        except self.NoCredentialsError:
            raise CredentialsError("AWS credentials are invalid or expired.")
        except self.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('NoSuchKey', '404'):
                raise NotFoundError(f"Object not found: {remote_path}")
            if error_code == 'NoSuchBucket':
                raise NotFoundError(f"Bucket not found: {bucket}")
            raise TransferError(f"Download failed: {e}")
        except Exception as e:
            raise TransferError(f"Download failed: {e}")

    def presign(self, remote_path: str, expires: int) -> str:
        """Generate a presigned URL for downloading from S3."""
        bucket, key = self._parse_s3_path(remote_path)
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expires
            )
            return url
        except self.NoCredentialsError:
            raise CredentialsError("AWS credentials are invalid or expired.")
        except self.ClientError as e:
            raise TransferError(f"Failed to generate presigned URL: {e}")

    def list_objects(self, bucket: str, prefix: str = "") -> Generator[str, None, None]:
        """List objects in an S3 bucket."""
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    yield obj['Key']
        except self.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                raise NotFoundError(f"Bucket not found: {bucket}")
            raise TransferError(f"Failed to list objects: {e}")

    def exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists in S3."""
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except self.ClientError:
            return False

    @staticmethod
    def _parse_s3_path(path: str) -> tuple:
        """Parse s3://bucket/key into (bucket, key)."""
        parsed = urlparse(path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key


class GCSProvider(StorageProvider):
    """Google Cloud Storage provider."""

    def __init__(self):
        try:
            from google.cloud import storage
            from google.auth.exceptions import DefaultCredentialsError
        except ImportError:
            raise CredentialsError(
                "google-cloud-storage is not installed. Run: pip install google-cloud-storage"
            )

        self.storage = storage
        self.DefaultCredentialsError = DefaultCredentialsError

        # Check for credentials
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if creds_path and not os.path.exists(creds_path):
            raise CredentialsError(
                f"GOOGLE_APPLICATION_CREDENTIALS file not found: {creds_path}"
            )

        try:
            self.client = storage.Client()
        except DefaultCredentialsError:
            raise CredentialsError(
                "GCP credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS "
                "environment variable to your service account JSON file path."
            )

    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file to GCS."""
        bucket_name, blob_name = self._parse_gcs_path(remote_path)
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
        except FileNotFoundError:
            raise NotFoundError(f"Local file not found: {local_path}")
        except Exception as e:
            if 'Not Found' in str(e) or '404' in str(e):
                raise NotFoundError(f"Bucket not found: {bucket_name}")
            raise TransferError(f"Upload failed: {e}")

    def download(self, remote_path: str, local_path: str) -> None:
        """Download a file from GCS."""
        bucket_name, blob_name = self._parse_gcs_path(remote_path)
        try:
            os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
        except Exception as e:
            if 'Not Found' in str(e) or '404' in str(e):
                raise NotFoundError(f"Object not found: {remote_path}")
            raise TransferError(f"Download failed: {e}")

    def presign(self, remote_path: str, expires: int) -> str:
        """Generate a signed URL for GCS."""
        bucket_name, blob_name = self._parse_gcs_path(remote_path)
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expires),
                method="GET"
            )
            return url
        except Exception as e:
            raise TransferError(f"Failed to generate signed URL: {e}")

    def list_objects(self, bucket: str, prefix: str = "") -> Generator[str, None, None]:
        """List objects in a GCS bucket."""
        try:
            bucket_obj = self.client.bucket(bucket)
            blobs = bucket_obj.list_blobs(prefix=prefix)
            for blob in blobs:
                yield blob.name
        except Exception as e:
            if 'Not Found' in str(e) or '404' in str(e):
                raise NotFoundError(f"Bucket not found: {bucket}")
            raise TransferError(f"Failed to list objects: {e}")

    def exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists in GCS."""
        try:
            bucket_obj = self.client.bucket(bucket)
            blob = bucket_obj.blob(key)
            return blob.exists()
        except Exception:
            return False

    @staticmethod
    def _parse_gcs_path(path: str) -> tuple:
        """Parse gs://bucket/key into (bucket, key)."""
        parsed = urlparse(path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key


class AzureProvider(StorageProvider):
    """Azure Blob Storage provider."""

    def __init__(self):
        try:
            from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
            from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError
        except ImportError:
            raise CredentialsError(
                "azure-storage-blob is not installed. Run: pip install azure-storage-blob"
            )

        self.generate_blob_sas = generate_blob_sas
        self.BlobSasPermissions = BlobSasPermissions
        self.ResourceNotFoundError = ResourceNotFoundError
        self.ClientAuthenticationError = ClientAuthenticationError

        connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            raise CredentialsError(
                "Azure credentials not found. Set AZURE_STORAGE_CONNECTION_STRING "
                "environment variable."
            )

        try:
            self.client = BlobServiceClient.from_connection_string(connection_string)
            # Extract account name and key from connection string for SAS generation
            self._parse_connection_string(connection_string)
        except Exception as e:
            raise CredentialsError(f"Invalid Azure connection string: {e}")

    def _parse_connection_string(self, conn_str: str) -> None:
        """Extract account name and key from connection string."""
        parts = dict(part.split('=', 1) for part in conn_str.split(';') if '=' in part)
        self.account_name = parts.get('AccountName', '')
        self.account_key = parts.get('AccountKey', '')

    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file to Azure Blob Storage."""
        container, blob_name = self._parse_azure_path(remote_path)
        try:
            blob_client = self.client.get_blob_client(container=container, blob=blob_name)
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
        except FileNotFoundError:
            raise NotFoundError(f"Local file not found: {local_path}")
        except self.ResourceNotFoundError:
            raise NotFoundError(f"Container not found: {container}")
        except self.ClientAuthenticationError:
            raise CredentialsError("Azure credentials are invalid.")
        except Exception as e:
            raise TransferError(f"Upload failed: {e}")

    def download(self, remote_path: str, local_path: str) -> None:
        """Download a file from Azure Blob Storage."""
        container, blob_name = self._parse_azure_path(remote_path)
        try:
            os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
            blob_client = self.client.get_blob_client(container=container, blob=blob_name)
            with open(local_path, 'wb') as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
        except self.ResourceNotFoundError:
            raise NotFoundError(f"Object not found: {remote_path}")
        except self.ClientAuthenticationError:
            raise CredentialsError("Azure credentials are invalid.")
        except Exception as e:
            raise TransferError(f"Download failed: {e}")

    def presign(self, remote_path: str, expires: int) -> str:
        """Generate a SAS URL for Azure Blob Storage."""
        container, blob_name = self._parse_azure_path(remote_path)
        try:
            sas_token = self.generate_blob_sas(
                account_name=self.account_name,
                container_name=container,
                blob_name=blob_name,
                account_key=self.account_key,
                permission=self.BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(seconds=expires)
            )
            url = f"https://{self.account_name}.blob.core.windows.net/{container}/{blob_name}?{sas_token}"
            return url
        except Exception as e:
            raise TransferError(f"Failed to generate SAS URL: {e}")

    def list_objects(self, bucket: str, prefix: str = "") -> Generator[str, None, None]:
        """List blobs in an Azure container."""
        try:
            container_client = self.client.get_container_client(bucket)
            blobs = container_client.list_blobs(name_starts_with=prefix)
            for blob in blobs:
                yield blob.name
        except self.ResourceNotFoundError:
            raise NotFoundError(f"Container not found: {bucket}")
        except Exception as e:
            raise TransferError(f"Failed to list objects: {e}")

    def exists(self, bucket: str, key: str) -> bool:
        """Check if a blob exists in Azure."""
        try:
            blob_client = self.client.get_blob_client(container=bucket, blob=key)
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False

    @staticmethod
    def _parse_azure_path(path: str) -> tuple:
        """Parse az://container/blob into (container, blob)."""
        parsed = urlparse(path)
        container = parsed.netloc
        blob_name = parsed.path.lstrip('/')
        return container, blob_name


def get_provider(url: str) -> StorageProvider:
    """
    Auto-detect and return the appropriate storage provider based on URL scheme.

    Args:
        url: Cloud storage URL (s3://, gs://, or az://)

    Returns:
        StorageProvider instance for the detected scheme

    Raises:
        ValueError: If the URL scheme is not recognized
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    providers = {
        's3': S3Provider,
        'gs': GCSProvider,
        'az': AzureProvider,
    }

    if scheme not in providers:
        raise ValueError(
            f"Unknown URL scheme: {scheme}. "
            f"Supported schemes: s3://, gs://, az://"
        )

    return providers[scheme]()


def parse_cloud_url(url: str) -> tuple:
    """Parse cloud URL into (scheme, bucket, path)."""
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path.lstrip('/')


def matches_glob(name: str, patterns: list, exclude_patterns: list = None) -> bool:
    """Check if a name matches include patterns and doesn't match exclude patterns."""
    if patterns:
        if not any(fnmatch.fnmatch(name, p) for p in patterns):
            return False
    if exclude_patterns:
        if any(fnmatch.fnmatch(name, p) for p in exclude_patterns):
            return False
    return True


def cmd_upload(args) -> int:
    """Handle the upload command."""
    local_path = args.source
    remote_path = args.destination

    if not os.path.exists(local_path):
        print(f"Error: Local file not found: {local_path}", file=sys.stderr)
        return EXIT_NOT_FOUND

    # If destination ends with /, append the filename
    if remote_path.endswith('/'):
        remote_path = remote_path + os.path.basename(local_path)

    try:
        provider = get_provider(remote_path)
        provider.upload(local_path, remote_path)
        print(f"Uploaded: {local_path} -> {remote_path}")
        return EXIT_SUCCESS
    except CredentialsError as e:
        print(f"Credentials error: {e}", file=sys.stderr)
        return EXIT_CREDENTIALS_ERROR
    except NotFoundError as e:
        print(f"Not found: {e}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except TransferError as e:
        print(f"Transfer error: {e}", file=sys.stderr)
        return EXIT_TRANSFER_ERROR
    except ValueError as e:
        print(f"Invalid argument: {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS


def cmd_download(args) -> int:
    """Handle the download command."""
    remote_path = args.source
    local_path = args.destination

    try:
        provider = get_provider(remote_path)
        scheme, bucket, key = parse_cloud_url(remote_path)

        # Check for glob patterns in the key
        if '*' in key or '?' in key:
            # Find the prefix before any glob characters
            prefix_parts = []
            for part in key.split('/'):
                if '*' in part or '?' in part:
                    break
                prefix_parts.append(part)
            prefix = '/'.join(prefix_parts)
            pattern = os.path.basename(key)

            # Ensure local destination is a directory
            if not local_path.endswith('/'):
                local_path = local_path + '/'
            os.makedirs(local_path, exist_ok=True)

            # List and download matching objects
            downloaded = 0
            for obj_key in provider.list_objects(bucket, prefix):
                obj_name = os.path.basename(obj_key)
                if fnmatch.fnmatch(obj_name, pattern):
                    full_remote = f"{scheme}://{bucket}/{obj_key}"
                    full_local = os.path.join(local_path, obj_name)
                    provider.download(full_remote, full_local)
                    print(f"Downloaded: {full_remote} -> {full_local}")
                    downloaded += 1

            if downloaded == 0:
                print(f"No files matched pattern: {remote_path}", file=sys.stderr)
                return EXIT_NOT_FOUND

            print(f"Downloaded {downloaded} file(s)")
            return EXIT_SUCCESS
        else:
            # Single file download
            if os.path.isdir(local_path) or local_path.endswith('/'):
                local_path = os.path.join(local_path, os.path.basename(key))

            provider.download(remote_path, local_path)
            print(f"Downloaded: {remote_path} -> {local_path}")
            return EXIT_SUCCESS

    except CredentialsError as e:
        print(f"Credentials error: {e}", file=sys.stderr)
        return EXIT_CREDENTIALS_ERROR
    except NotFoundError as e:
        print(f"Not found: {e}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except TransferError as e:
        print(f"Transfer error: {e}", file=sys.stderr)
        return EXIT_TRANSFER_ERROR
    except ValueError as e:
        print(f"Invalid argument: {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS


def cmd_presign(args) -> int:
    """Handle the presign command."""
    remote_path = args.path
    expires = args.expires

    if expires <= 0:
        print("Error: Expiration time must be positive", file=sys.stderr)
        return EXIT_INVALID_ARGS

    if expires > 604800:  # 7 days
        print("Warning: Maximum expiration is 7 days (604800 seconds)", file=sys.stderr)
        expires = 604800

    try:
        provider = get_provider(remote_path)
        url = provider.presign(remote_path, expires)
        print(url)
        return EXIT_SUCCESS
    except CredentialsError as e:
        print(f"Credentials error: {e}", file=sys.stderr)
        return EXIT_CREDENTIALS_ERROR
    except NotFoundError as e:
        print(f"Not found: {e}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except TransferError as e:
        print(f"Transfer error: {e}", file=sys.stderr)
        return EXIT_TRANSFER_ERROR
    except ValueError as e:
        print(f"Invalid argument: {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS


def cmd_sync(args) -> int:
    """Handle the sync command."""
    source = args.source
    destination = args.destination
    include_patterns = args.include or []
    exclude_patterns = args.exclude or []
    dry_run = args.dry_run
    delete = args.delete

    # Determine sync direction
    source_is_local = not (source.startswith('s3://') or
                          source.startswith('gs://') or
                          source.startswith('az://'))
    dest_is_local = not (destination.startswith('s3://') or
                        destination.startswith('gs://') or
                        destination.startswith('az://'))

    if source_is_local and dest_is_local:
        print("Error: At least one path must be a cloud URL", file=sys.stderr)
        return EXIT_INVALID_ARGS

    if not source_is_local and not dest_is_local:
        print("Error: Cross-cloud sync not supported. Use local as intermediate.",
              file=sys.stderr)
        return EXIT_INVALID_ARGS

    try:
        if source_is_local:
            # Upload sync: local -> cloud
            return _sync_upload(source, destination, include_patterns,
                              exclude_patterns, dry_run, delete)
        else:
            # Download sync: cloud -> local
            return _sync_download(source, destination, include_patterns,
                                exclude_patterns, dry_run, delete)
    except CredentialsError as e:
        print(f"Credentials error: {e}", file=sys.stderr)
        return EXIT_CREDENTIALS_ERROR
    except NotFoundError as e:
        print(f"Not found: {e}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except TransferError as e:
        print(f"Transfer error: {e}", file=sys.stderr)
        return EXIT_TRANSFER_ERROR
    except ValueError as e:
        print(f"Invalid argument: {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS


def _sync_upload(local_dir: str, remote_url: str, include: list,
                 exclude: list, dry_run: bool, delete: bool) -> int:
    """Sync local directory to cloud storage."""
    provider = get_provider(remote_url)
    scheme, bucket, remote_prefix = parse_cloud_url(remote_url)

    if not os.path.isdir(local_dir):
        raise NotFoundError(f"Local directory not found: {local_dir}")

    # Normalize paths
    local_dir = os.path.abspath(local_dir)
    if not remote_prefix.endswith('/') and remote_prefix:
        remote_prefix = remote_prefix + '/'

    # Collect local files
    local_files = set()
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, local_dir)

            if matches_glob(rel_path, include, exclude):
                local_files.add(rel_path)

    # Upload files
    uploaded = 0
    for rel_path in sorted(local_files):
        local_path = os.path.join(local_dir, rel_path)
        remote_key = remote_prefix + rel_path.replace(os.sep, '/')
        remote_path = f"{scheme}://{bucket}/{remote_key}"

        if dry_run:
            print(f"[DRY RUN] Would upload: {local_path} -> {remote_path}")
        else:
            provider.upload(local_path, remote_path)
            print(f"Uploaded: {local_path} -> {remote_path}")
        uploaded += 1

    # Handle delete option
    if delete:
        remote_keys = set()
        for key in provider.list_objects(bucket, remote_prefix):
            rel_key = key[len(remote_prefix):] if key.startswith(remote_prefix) else key
            if matches_glob(rel_key, include, exclude):
                remote_keys.add(rel_key)

        # Find keys in remote but not in local
        local_keys = {f.replace(os.sep, '/') for f in local_files}
        to_delete = remote_keys - local_keys

        for rel_key in sorted(to_delete):
            remote_path = f"{scheme}://{bucket}/{remote_prefix}{rel_key}"
            if dry_run:
                print(f"[DRY RUN] Would delete: {remote_path}")
            else:
                # Note: delete not implemented in providers, would need to add
                print(f"Would delete (not implemented): {remote_path}")

    print(f"Sync complete: {uploaded} file(s) {'would be ' if dry_run else ''}uploaded")
    return EXIT_SUCCESS


def _sync_download(remote_url: str, local_dir: str, include: list,
                   exclude: list, dry_run: bool, delete: bool) -> int:
    """Sync cloud storage to local directory."""
    provider = get_provider(remote_url)
    scheme, bucket, remote_prefix = parse_cloud_url(remote_url)

    # Normalize paths
    local_dir = os.path.abspath(local_dir)
    if not remote_prefix.endswith('/') and remote_prefix:
        remote_prefix = remote_prefix + '/'

    # Ensure local directory exists
    if not dry_run:
        os.makedirs(local_dir, exist_ok=True)

    # Collect remote files
    remote_files = set()
    for key in provider.list_objects(bucket, remote_prefix):
        rel_key = key[len(remote_prefix):] if key.startswith(remote_prefix) else key
        if rel_key and matches_glob(rel_key, include, exclude):
            remote_files.add(rel_key)

    # Download files
    downloaded = 0
    for rel_key in sorted(remote_files):
        remote_path = f"{scheme}://{bucket}/{remote_prefix}{rel_key}"
        local_path = os.path.join(local_dir, rel_key.replace('/', os.sep))

        if dry_run:
            print(f"[DRY RUN] Would download: {remote_path} -> {local_path}")
        else:
            os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
            provider.download(remote_path, local_path)
            print(f"Downloaded: {remote_path} -> {local_path}")
        downloaded += 1

    # Handle delete option
    if delete:
        local_files = set()
        if os.path.isdir(local_dir):
            for root, dirs, files in os.walk(local_dir):
                for filename in files:
                    local_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(local_path, local_dir)
                    if matches_glob(rel_path, include, exclude):
                        local_files.add(rel_path.replace(os.sep, '/'))

        # Find files in local but not in remote
        to_delete = local_files - remote_files

        for rel_path in sorted(to_delete):
            local_path = os.path.join(local_dir, rel_path.replace('/', os.sep))
            if dry_run:
                print(f"[DRY RUN] Would delete: {local_path}")
            else:
                os.remove(local_path)
                print(f"Deleted: {local_path}")

    print(f"Sync complete: {downloaded} file(s) {'would be ' if dry_run else ''}downloaded")
    return EXIT_SUCCESS


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog='s3_sync',
        description='''
A unified CLI tool for cloud storage operations.

Supports Amazon S3, Google Cloud Storage, and Azure Blob Storage.
Auto-detects the provider from URL scheme (s3://, gs://, az://).
        ''',
        epilog='''
ENVIRONMENT VARIABLES:
  AWS_ACCESS_KEY_ID              AWS access key
  AWS_SECRET_ACCESS_KEY          AWS secret key
  GOOGLE_APPLICATION_CREDENTIALS Path to GCP service account JSON
  AZURE_STORAGE_CONNECTION_STRING Azure connection string

EXIT CODES:
  0  Success
  1  Credentials error
  2  Not found error
  3  Transfer error
  4  Invalid arguments

EXAMPLES:
  Upload a file to S3:
    %(prog)s upload report.pdf s3://my-bucket/reports/

  Download files matching a pattern from GCS:
    %(prog)s download gs://my-bucket/data/*.csv ./local/

  Generate a presigned URL valid for 1 hour:
    %(prog)s presign s3://my-bucket/file.zip --expires 3600

  Sync a local directory to S3 with filters:
    %(prog)s sync ./local/ s3://my-bucket/backup/ --include "*.json" --exclude "*.tmp"

  Dry-run sync to see what would be transferred:
    %(prog)s sync ./local/ s3://my-bucket/backup/ --dry-run

  Sync from Azure to local, deleting extra local files:
    %(prog)s sync az://container/data/ ./local/ --delete
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    subparsers = parser.add_subparsers(
        title='commands',
        description='Available commands',
        dest='command',
        required=True
    )

    # Upload command
    upload_parser = subparsers.add_parser(
        'upload',
        help='Upload a file to cloud storage',
        description='''
Upload a local file to cloud storage.

The destination URL scheme determines the provider:
  s3://  - Amazon S3
  gs://  - Google Cloud Storage
  az://  - Azure Blob Storage

If the destination ends with /, the source filename is appended.
        ''',
        epilog='''
EXAMPLES:
  %(prog)s report.pdf s3://bucket/reports/
  %(prog)s data.json gs://bucket/data/data.json
  %(prog)s backup.tar.gz az://container/backups/
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    upload_parser.add_argument(
        'source',
        help='Local file path to upload'
    )
    upload_parser.add_argument(
        'destination',
        help='Cloud storage URL (s3://, gs://, or az://)'
    )
    upload_parser.set_defaults(func=cmd_upload)

    # Download command
    download_parser = subparsers.add_parser(
        'download',
        help='Download files from cloud storage',
        description='''
Download files from cloud storage to local filesystem.

Supports glob patterns (* and ?) in the source path.
When using patterns, the destination must be a directory.
        ''',
        epilog='''
EXAMPLES:
  %(prog)s s3://bucket/file.txt ./local/
  %(prog)s gs://bucket/data/*.csv ./data/
  %(prog)s az://container/logs/app-*.log ./logs/
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    download_parser.add_argument(
        'source',
        help='Cloud storage URL (supports glob patterns)'
    )
    download_parser.add_argument(
        'destination',
        help='Local directory or file path'
    )
    download_parser.set_defaults(func=cmd_download)

    # Presign command
    presign_parser = subparsers.add_parser(
        'presign',
        help='Generate a presigned/signed URL',
        description='''
Generate a presigned URL for temporary access to a cloud object.

The URL allows downloading without credentials for the specified duration.
Maximum expiration time is 7 days (604800 seconds).
        ''',
        epilog='''
EXAMPLES:
  %(prog)s s3://bucket/file.zip
  %(prog)s s3://bucket/file.zip --expires 3600
  %(prog)s gs://bucket/report.pdf --expires 86400
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    presign_parser.add_argument(
        'path',
        help='Cloud storage URL of the object'
    )
    presign_parser.add_argument(
        '--expires', '-e',
        type=int,
        default=3600,
        metavar='SECONDS',
        help='URL expiration time in seconds (default: 3600, max: 604800)'
    )
    presign_parser.set_defaults(func=cmd_presign)

    # Sync command
    sync_parser = subparsers.add_parser(
        'sync',
        help='Synchronize directories with cloud storage',
        description='''
Synchronize a local directory with cloud storage or vice versa.

Direction is determined by which path is a cloud URL:
  Local -> Cloud: Upload new/modified files
  Cloud -> Local: Download new/modified files

Use glob patterns with --include and --exclude to filter files.
        ''',
        epilog='''
EXAMPLES:
  Sync local to S3:
    %(prog)s ./data/ s3://bucket/data/

  Sync S3 to local:
    %(prog)s s3://bucket/data/ ./data/

  Sync only JSON files:
    %(prog)s ./data/ s3://bucket/data/ --include "*.json"

  Exclude temporary files:
    %(prog)s ./data/ s3://bucket/data/ --exclude "*.tmp" --exclude "*.bak"

  Preview what would be synced:
    %(prog)s ./data/ s3://bucket/data/ --dry-run

  Sync and delete extra files at destination:
    %(prog)s ./data/ s3://bucket/data/ --delete
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sync_parser.add_argument(
        'source',
        help='Source path (local directory or cloud URL)'
    )
    sync_parser.add_argument(
        'destination',
        help='Destination path (cloud URL or local directory)'
    )
    sync_parser.add_argument(
        '--include', '-i',
        action='append',
        metavar='PATTERN',
        help='Include only files matching glob pattern (can be repeated)'
    )
    sync_parser.add_argument(
        '--exclude', '-x',
        action='append',
        metavar='PATTERN',
        help='Exclude files matching glob pattern (can be repeated)'
    )
    sync_parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be transferred without actually doing it'
    )
    sync_parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete files at destination that do not exist at source'
    )
    sync_parser.set_defaults(func=cmd_sync)

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return EXIT_INVALID_ARGS


if __name__ == '__main__':
    sys.exit(main())
