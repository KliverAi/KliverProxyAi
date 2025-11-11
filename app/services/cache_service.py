"""Cache service for managing Gemini Context Caching"""
import json
import time
from datetime import timedelta
from typing import List, Optional
from fastapi import HTTPException, status, UploadFile

from app.models import TokenAiServiceUsageInfo, CacheCreationResponse
from app.config import logger


async def create_gemini_context_cache(
    api_key: str,
    model: str,
    system_instruction: Optional[str],
    display_name: Optional[str],
    contents_to_cache: Optional[str],
    files: Optional[List[UploadFile]],
    use_file_api: bool
) -> CacheCreationResponse:
    """
    Create a Gemini context cache with the provided content.

    NOTE: Google Gemini API supports TWO methods for caching:
    1. File API: Upload files to Google's servers (better for large files, supports more formats)
    2. Text extraction: Extract text and send directly (faster but limited)
    Reference: https://ai.google.dev/gemini-api/docs/caching
    """
    if model is None or not model.lower().startswith('gemini'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Context Caching is only supported for Gemini models."
        )

    logger.info(f"Attempting to create Context Cache: {display_name} using {model}...")

    try:
        import google.generativeai as genai

        # Configure the API key
        genai.configure(api_key=api_key)

        # Prepare content for caching
        content_list = []

        # Process text contents if provided
        if contents_to_cache:
            try:
                # Try to parse as JSON array
                parsed_contents = json.loads(contents_to_cache)
                if isinstance(parsed_contents, list):
                    content_list.extend(parsed_contents)
                else:
                    content_list.append(str(parsed_contents))
            except json.JSONDecodeError:
                # If not JSON, treat as single string
                content_list.append(contents_to_cache)

        # Process uploaded files
        if files and use_file_api:
            # METHOD 1: Use Gemini File API (RECOMMENDED)
            logger.info(f"Using Gemini File API to upload {len(files)} files")

            from app.file_manager import GeminiFileManager

            file_manager = GeminiFileManager(api_key)
            uploaded_files = await file_manager.upload_multiple_files(
                files,
                display_name_prefix=display_name
            )

            # Add uploaded file URIs to content
            for uploaded_file in uploaded_files:
                content_list.append(uploaded_file)  # File objects can be passed directly
                logger.info(f"Added file to cache: {uploaded_file.display_name} (URI: {uploaded_file.uri})")

        elif files:
            # METHOD 2: Extract text from files directly
            for file in files:
                try:
                    # Read file content
                    file_content = await file.read()

                    # Decode based on file type
                    if file.filename.endswith(('.txt', '.md', '.json', '.csv', '.py', '.js', '.html', '.xml')):
                        # Text files
                        text_content = file_content.decode('utf-8')
                        content_list.append(f"File: {file.filename}\n\n{text_content}")
                    elif file.filename.endswith('.pdf'):
                        # Extract text from PDF using PyPDF2
                        try:
                            from PyPDF2 import PdfReader
                            import io

                            logger.info(f"Processing PDF file: {file.filename}")

                            # Create a file-like object from bytes
                            pdf_file = io.BytesIO(file_content)

                            # Read the PDF
                            pdf_reader = PdfReader(pdf_file)

                            # Extract text from all pages
                            pdf_text = []
                            for page_num, page in enumerate(pdf_reader.pages, 1):
                                page_text = page.extract_text()
                                if page_text.strip():
                                    pdf_text.append(f"--- Page {page_num} ---\n{page_text}")

                            if pdf_text:
                                combined_text = "\n\n".join(pdf_text)
                                content_list.append(f"File: {file.filename}\n\n{combined_text}")
                                logger.info(f"Successfully extracted text from PDF: {file.filename} ({len(pdf_reader.pages)} pages)")
                            else:
                                logger.warning(f"No text extracted from PDF: {file.filename}")
                                raise HTTPException(
                                    status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"Could not extract text from PDF {file.filename}. The PDF might be image-based or encrypted."
                                )
                        except ImportError:
                            logger.error("PyPDF2 not installed")
                            raise HTTPException(
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="PDF support requires PyPDF2 package. Please install it."
                            )
                        except Exception as pdf_error:
                            logger.error(f"Error parsing PDF {file.filename}: {pdf_error}")
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Error parsing PDF {file.filename}: {str(pdf_error)}"
                            )
                    else:
                        # Try to decode as UTF-8 for other files
                        try:
                            text_content = file_content.decode('utf-8')
                            content_list.append(f"File: {file.filename}\n\n{text_content}")
                        except UnicodeDecodeError:
                            logger.warning(f"Cannot decode file {file.filename} as text")
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"File {file.filename} is not a supported text format"
                            )
                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Error processing file {file.filename}: {str(e)}"
                    )

        if not content_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content provided. Please provide either 'contents_to_cache' or upload files."
            )

        logger.info(f"Total content items to cache: {len(content_list)}")

        # Create the cache using the Google GenAI SDK
        cache = genai.caching.CachedContent.create(
            model=model,
            display_name=display_name or f"cache_{int(time.time())}",
            system_instruction=system_instruction,
            contents=content_list,
            ttl=timedelta(hours=1)  # Cache expires in 1 hour
        )

        logger.info(f"✅ Context Cache successfully created: {cache.name}")

        # Extract token usage from cache metadata
        token_usage = None
        if hasattr(cache, 'usage_metadata') and cache.usage_metadata:
            total_tokens = getattr(cache.usage_metadata, 'total_token_count', 0)
            token_usage = TokenAiServiceUsageInfo(
                input_tokens=total_tokens,  # All cached content counts as input tokens
                output_tokens=0,  # Cache creation doesn't generate output tokens
                total_tokens=total_tokens
            )
            logger.info(f"Token usage - Input: {total_tokens}, Total: {total_tokens}")

        return CacheCreationResponse(
            cache_name=cache.name,
            token_usage=token_usage
        )

    except ImportError:
        logger.error("google-generativeai package not installed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Missing required package: google-generativeai. Please install it."
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Gemini Context Cache creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during Gemini Context Cache creation: {str(e)}"
        )
