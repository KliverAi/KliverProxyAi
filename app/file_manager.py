"""
File management for Gemini File API
Handles file uploads to Google's servers for context caching
"""
import logging
import tempfile
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile
import google.generativeai as genai

logger = logging.getLogger("kliver.ai")

# File type to MIME type mapping
SUPPORTED_FILE_TYPES = {
    # Text files
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.json': 'application/json',
    '.csv': 'text/csv',
    '.html': 'text/html',
    '.xml': 'application/xml',

    # Code files
    '.py': 'text/x-python',
    '.js': 'application/javascript',
    '.ts': 'application/typescript',
    '.java': 'text/x-java',
    '.cpp': 'text/x-c++src',
    '.c': 'text/x-csrc',

    # Documents
    '.pdf': 'application/pdf',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',

    # Images (Gemini supports vision)
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',

    # Audio
    '.mp3': 'audio/mp3',
    '.wav': 'audio/wav',
    '.aac': 'audio/aac',

    # Video
    '.mp4': 'video/mp4',
    '.avi': 'video/x-msvideo',
    '.mov': 'video/quicktime',
}

# Size limits (as per Gemini API)
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB per file (updated limit)
MAX_TOTAL_STORAGE = 50 * 1024 * 1024 * 1024  # 50 GB total storage


class GeminiFileManager:
    """Manages file uploads to Gemini File API"""

    def __init__(self, api_key: str):
        """Initialize with API key"""
        genai.configure(api_key=api_key)
        # Note: genai.Client doesn't exist in the current version of google-generativeai
        # The client is implicitly configured via genai.configure()

    def get_mime_type(self, filename: str) -> Optional[str]:
        """Get MIME type from filename extension"""
        ext = Path(filename).suffix.lower()
        return SUPPORTED_FILE_TYPES.get(ext)

    async def upload_file(
        self,
        file: UploadFile,
        display_name: Optional[str] = None
    ) -> genai.types.File:
        """
        Upload a file to Gemini File API

        Args:
            file: FastAPI UploadFile object
            display_name: Optional display name for the file

        Returns:
            genai.types.File: Uploaded file object with URI

        Raises:
            ValueError: If file type not supported or size exceeds limit
        """
        # Validate file type
        mime_type = self.get_mime_type(file.filename)
        if not mime_type:
            raise ValueError(
                f"Unsupported file type: {Path(file.filename).suffix}. "
                f"Supported types: {', '.join(SUPPORTED_FILE_TYPES.keys())}"
            )

        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        # Validate size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File size {file_size / 1024 / 1024:.2f}MB exceeds "
                f"maximum {MAX_FILE_SIZE / 1024 / 1024 / 1024}GB"
            )

        logger.info(f"Uploading file: {file.filename} ({file_size / 1024:.2f}KB, {mime_type})")

        # Save to temporary file (Gemini File API needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            # Upload to Gemini
            uploaded_file = genai.upload_file(
                path=tmp_path,
                display_name=display_name or file.filename,
                mime_type=mime_type
            )

            logger.info(f"✅ File uploaded successfully: {uploaded_file.name} (URI: {uploaded_file.uri})")
            return uploaded_file

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    async def upload_multiple_files(
        self,
        files: List[UploadFile],
        display_name_prefix: Optional[str] = None
    ) -> List[genai.types.File]:
        """
        Upload multiple files to Gemini File API

        Args:
            files: List of FastAPI UploadFile objects
            display_name_prefix: Optional prefix for display names

        Returns:
            List[genai.types.File]: List of uploaded file objects
        """
        uploaded_files = []

        for idx, file in enumerate(files):
            display_name = None
            if display_name_prefix:
                display_name = f"{display_name_prefix}_{idx+1}_{file.filename}"

            uploaded_file = await self.upload_file(file, display_name)
            uploaded_files.append(uploaded_file)

        logger.info(f"✅ Successfully uploaded {len(uploaded_files)} files")
        return uploaded_files

    def list_files(self) -> List[genai.types.File]:
        """List all uploaded files"""
        return list(genai.list_files())

    def get_file(self, file_name: str) -> genai.types.File:
        """Get file metadata by name"""
        return genai.get_file(file_name)

    def delete_file(self, file_name: str) -> None:
        """Delete a file from Gemini File API"""
        genai.delete_file(file_name)
        logger.info(f"🗑️ Deleted file: {file_name}")

    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Delete files older than max_age_hours

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            int: Number of files deleted
        """
        from datetime import datetime, timedelta

        files = self.list_files()
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0

        for file in files:
            # Parse file creation time
            file_time = datetime.fromisoformat(file.create_time.replace('Z', '+00:00'))

            if file_time < cutoff_time:
                try:
                    self.delete_file(file.name)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete file {file.name}: {e}")

        logger.info(f"🧹 Cleaned up {deleted_count} old files")
        return deleted_count
