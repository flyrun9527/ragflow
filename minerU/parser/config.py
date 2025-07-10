#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os
from pydantic import BaseModel, Field


class MinerUParserConfig(BaseModel):
    """Configuration for the MinerU parser."""
    
    # API URL for the MinerU service
    api_url: str = Field(
        default="http://192.168.130.24:8889",
        description="URL of the MinerU API service"
    )
    
    # Timeout for API requests in seconds
    timeout: int = Field(
        default=300,
        description="Timeout for API requests in seconds"
    )
    
    # Whether to use hierarchical chunking method
    use_hierarchical: bool = Field(
        default=True,
        description="Whether to use hierarchical chunking method"
    )
    
    # Backend to use for parsing
    backend: str = Field(
        default="vlm-sglang-client",
        description="Backend to use for parsing (pipeline, vlm-transformers, vlm-sglang-engine, vlm-sglang-client)"
    )
    
    # Language for OCR
    language: str = Field(
        default="ch",
        description="Language for OCR recognition (auto, ch, en, etc.)"
    )
    
    @classmethod
    def from_env(cls) -> "MinerUParserConfig":
        """Create a config from environment variables."""
        return cls(
            api_url=os.environ.get("MINERU_API_URL", "http://192.168.130.24:8889"),
            timeout=int(os.environ.get("MINERU_TIMEOUT", "300")),
            use_hierarchical=os.environ.get("MINERU_USE_HIERARCHICAL", "true").lower() == "true",
            backend=os.environ.get("MINERU_BACKEND", "pipeline"),
            language=os.environ.get("MINERU_LANGUAGE", "auto"),
        ) 