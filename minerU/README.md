# MinerU Parser

## Overview

MinerU is a PDF parsing integration for RagFlow that leverages the MinerU tool to extract high-quality content from PDF documents. This integration focuses specifically on using MinerU's capabilities to parse PDF files and extract structured data for use in RAG workflows.

## Features

- High-quality PDF to Markdown conversion
- Extraction of text with preserved layout
- Support for formula extraction
- Table recognition and conversion
- Support for multi-language documents
- Hierarchical text structure preservation (headings, paragraphs, lists)

## Integration

MinerU connects to the MinerU API service (via HTTP) to process PDF documents. The integration handles:

- PDF document submission
- Processing status monitoring
- Result retrieval and transformation for RagFlow

## Configuration

The MinerU parser can be configured via environment variables or through the RagFlow configuration interface:

```yaml
MINERU_API_URL: "http://192.168.130.24:8889"  # URL of MinerU API service
MINERU_TIMEOUT: 300  # Timeout for API requests in seconds
```

## Usage

MinerU parser is automatically available as a parsing option for PDF documents in the RagFlow interface. When uploading PDF documents, select "MinerU" as the parsing method to leverage its extraction capabilities.

## Dependencies

- `requests`: For HTTP communication with MinerU API
- `tenacity`: For retry logic with backoff

## Limitations

- Requires network access to MinerU API service
- Processing large documents may take significant time 