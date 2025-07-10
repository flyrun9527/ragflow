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
import json
import logging
import mimetypes
import tempfile
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, HTTPException, Body
from pydantic import BaseModel

from .mineru_parser import MinerUParser
from .config import MinerUParserConfig
from rag.schema.document import Document
from rag.utils.chunk_utils import chunk_by_hierarchy

logger = logging.getLogger(__name__)

# Define Gotenberg service URL
GOTENBERG_URL = os.environ.get('GOTENBERG_URL', 'http://192.168.130.24:23000')

# Create router
router = APIRouter(
    prefix="/api/v1/parser/mineru",
    tags=["mineru"],
)


class ParseRequest(BaseModel):
    """解析文件的请求模型。"""
    file_path: str
    use_hierarchical: bool = True


class ParseResponse(BaseModel):
    """解析请求的响应模型。"""
    status: str
    message: str
    documents: List[Dict[str, Any]] = []


def convert_to_pdf(file_path: str) -> str:
    """
    使用 Gotenberg 服务将文件转换为 PDF。
    
    参数:
        file_path: 要转换的文件路径
        
    返回:
        转换后的 PDF 文件路径
    """
    logger.info(f"将文件转换为 PDF: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")
    
    # 确定文件类型
    file_ext = Path(file_path).suffix.lower()
    
    # 创建临时文件存储 PDF
    pdf_path = tempfile.mktemp(suffix=".pdf")
    
    try:
        # 根据文件类型选择正确的 Gotenberg 端点
        if file_ext in ['.docx', '.doc', '.odt', '.rtf']:
            endpoint = f"{GOTENBERG_URL}/forms/libreoffice/convert"
            files = {
                'files': (os.path.basename(file_path), open(file_path, 'rb')),
            }
        elif file_ext in ['.html', '.htm']:
            endpoint = f"{GOTENBERG_URL}/forms/chromium/convert/html"
            files = {
                'files': (os.path.basename(file_path), open(file_path, 'rb')),
            }
        elif file_ext in ['.md']:
            endpoint = f"{GOTENBERG_URL}/forms/markdown/convert"
            files = {
                'files': (os.path.basename(file_path), open(file_path, 'rb')),
            }
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.gif', '.bmp']:
            endpoint = f"{GOTENBERG_URL}/forms/chromium/convert/image"
            files = {
                'files': (os.path.basename(file_path), open(file_path, 'rb')),
            }
        else:
            raise ValueError(f"不支持转换的文件类型: {file_ext}")
        
        # 向 Gotenberg 发送请求
        response = requests.post(endpoint, files=files)
        
        if response.status_code != 200:
            raise RuntimeError(f"Gotenberg 转换失败，状态码 {response.status_code}: {response.text}")
        
        # 将 PDF 保存到临时文件
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"文件成功转换为 PDF: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        logger.exception(f"转换文件为 PDF 失败: {str(e)}")
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        raise


def apply_hierarchical_chunking(document: Document) -> List[Document]:
    """
    对文档应用层次化切片。
    
    参数:
        document: 要切片的文档
        
    返回:
        切片后的文档列表
    """
    logger.info(f"对文档应用层次化切片")
    
    # 使用 rag.utils.chunk_utils 中的函数进行层次化切片
    try:
        chunked_documents = chunk_by_hierarchy(
            document.page_content,
            metadata=document.metadata
        )
        logger.info(f"文档成功切片为 {len(chunked_documents)} 个块")
        return chunked_documents
    except Exception as e:
        logger.exception(f"应用层次化切片失败: {str(e)}")
        # 如果切片失败，返回原始文档
        return [document]


@router.post("/parse", response_model=ParseResponse)
async def parse_document(
    request: ParseRequest = Body(...),
    background_tasks: BackgroundTasks = None,
) -> ParseResponse:
    """
    使用 MinerU 解析器解析文档。
    
    参数:
        request: ParseRequest 对象
        background_tasks: 后台任务
        
    返回:
        ParseResponse 对象
    """
    temp_files = []  # 跟踪临时文件以便清理
    
    try:
        # 检查文件是否存在
        file_path = request.file_path
        path = Path(file_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"文件未找到: {file_path}")
        
        # 获取文件扩展名并确定是否需要转换
        file_ext = path.suffix.lower()
        is_pdf = file_ext == '.pdf'
        
        # 如果不是 PDF，则转换为 PDF
        if not is_pdf:
            logger.info(f"文件不是 PDF，正在转换: {file_path}")
            try:
                pdf_path = convert_to_pdf(str(path.absolute()))
                temp_files.append(pdf_path)
                file_path = pdf_path
            except Exception as e:
                logger.exception(f"转换文件为 PDF 失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"转换文件为 PDF 失败: {str(e)}")
        
        # 解析 PDF 文件
        config = MinerUParserConfig.from_env()
        parser = MinerUParser(api_url=config.api_url, timeout=config.timeout)
        documents = parser.parse(file_path)
        
        # 如果请求，应用层次化切片
        if request.use_hierarchical:
            all_chunked_docs = []
            for doc in documents:
                chunked_docs = apply_hierarchical_chunking(doc)
                all_chunked_docs.extend(chunked_docs)
            documents = all_chunked_docs
        
        # 将文档转换为字典以用于 API 响应
        doc_dicts = []
        for doc in documents:
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            doc_dicts.append(doc_dict)
        
        return ParseResponse(
            status="success",
            message=f"成功解析文档: {path.name}",
            documents=doc_dicts,
        )
        
    except Exception as e:
        logger.exception(f"解析文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"解析文档失败: {str(e)}")
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"删除临时文件 {temp_file} 失败: {str(e)}")


class HealthCheckResponse(BaseModel):
    """健康检查的响应模型。"""
    status: str
    version: str


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    检查 MinerU 解析器是否健康。
    
    返回:
        HealthCheckResponse 对象
    """
    try:
        # 尝试初始化解析器以检查配置
        parser = MinerUParser()
        
        return HealthCheckResponse(
            status="healthy",
            version="1.0.0"
        )
    except Exception as e:
        logger.exception(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}") 


class VersionResponse(BaseModel):
    """版本信息的响应模型。"""
    version: str
    backend: str
    features: List[str]


@router.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    """
    获取 MinerU 解析器的版本信息。
    
    返回:
        VersionResponse 对象
    """
    return VersionResponse(
        version="1.0.0",
        backend="pipeline",
        features=["pdf", "hierarchical", "tables", "formulas"]
    )


class ConfigResponse(BaseModel):
    """配置信息的响应模型。"""
    api_url: str
    timeout: int
    use_hierarchical: bool
    backend: str
    language: str


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """
    获取 MinerU 解析器的当前配置。
    
    返回:
        ConfigResponse 对象
    """
    # 从环境获取配置
    config = MinerUParserConfig.from_env()
    
    return ConfigResponse(
        api_url=config.api_url,
        timeout=config.timeout,
        use_hierarchical=config.use_hierarchical,
        backend=config.backend,
        language=config.language
    )


class UpdateConfigRequest(BaseModel):
    """更新配置的请求模型。"""
    api_url: Optional[str] = None
    timeout: Optional[int] = None
    use_hierarchical: Optional[bool] = None
    backend: Optional[str] = None
    language: Optional[str] = None


@router.post("/config", response_model=ConfigResponse)
async def update_config(request: UpdateConfigRequest) -> ConfigResponse:
    """
    更新 MinerU 解析器的配置。
    
    参数:
        request: UpdateConfigRequest 对象
        
    返回:
        更新配置后的 ConfigResponse 对象
    """
    import os
    
    # 获取当前配置
    config = MinerUParserConfig.from_env()
    
    # 如果提供了新值，则更新环境变量
    if request.api_url is not None:
        os.environ["MINERU_API_URL"] = request.api_url
    
    if request.timeout is not None:
        os.environ["MINERU_TIMEOUT"] = str(request.timeout)
    
    if request.use_hierarchical is not None:
        os.environ["MINERU_USE_HIERARCHICAL"] = str(request.use_hierarchical).lower()
    
    if request.backend is not None:
        os.environ["MINERU_BACKEND"] = request.backend
    
    if request.language is not None:
        os.environ["MINERU_LANGUAGE"] = request.language
    
    # 获取更新后的配置
    updated_config = MinerUParserConfig.from_env()
    
    return ConfigResponse(
        api_url=updated_config.api_url,
        timeout=updated_config.timeout,
        use_hierarchical=updated_config.use_hierarchical,
        backend=updated_config.backend,
        language=updated_config.language
    )


@router.post("/upload", response_model=ParseResponse)
async def upload_and_parse(
    file: UploadFile = File(...),
    use_hierarchical: bool = Form(True),
    background_tasks: BackgroundTasks = None,
) -> ParseResponse:
    """
    上传文件并使用 MinerU 解析器解析。
    
    参数:
        file: 上传的文件
        use_hierarchical: 是否使用层次化切片
        background_tasks: 后台任务
        
    返回:
        ParseResponse 对象
    """
    temp_files = []  # 跟踪临时文件以便清理
    
    try:
        # 获取文件扩展名
        file_ext = Path(file.filename).suffix.lower() if file.filename else ""
        is_pdf = file.content_type == "application/pdf" or file_ext == '.pdf'
        
        # 创建临时文件存储上传的内容
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
            temp_files.append(temp_file_path)
            content = await file.read()
            temp_file.write(content)
        
        # 如果不是 PDF，则转换为 PDF
        pdf_path = temp_file_path
        if not is_pdf:
            logger.info(f"文件不是 PDF，正在转换: {file.filename}")
            try:
                pdf_path = convert_to_pdf(temp_file_path)
                temp_files.append(pdf_path)
            except Exception as e:
                logger.exception(f"转换文件为 PDF 失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"转换文件为 PDF 失败: {str(e)}")
        
        # 解析 PDF 文件
        config = MinerUParserConfig.from_env()
        parser = MinerUParser(api_url=config.api_url, timeout=config.timeout)
        documents = parser.parse(pdf_path)
        
        # 如果请求，应用层次化切片
        if use_hierarchical:
            all_chunked_docs = []
            for doc in documents:
                chunked_docs = apply_hierarchical_chunking(doc)
                all_chunked_docs.extend(chunked_docs)
            documents = all_chunked_docs
        
        # 将文档转换为字典以用于 API 响应
        doc_dicts = []
        for doc in documents:
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            doc_dicts.append(doc_dict)
        
        return ParseResponse(
            status="success",
            message=f"成功解析文档: {file.filename}",
            documents=doc_dicts,
        )
                
    except Exception as e:
        logger.exception(f"解析上传的文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"解析上传的文档失败: {str(e)}")
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"删除临时文件 {temp_file} 失败: {str(e)}")


class BatchParseRequest(BaseModel):
    """批量解析的请求模型。"""
    file_paths: List[str]
    use_hierarchical: bool = True


class BatchParseResponse(BaseModel):
    """批量解析请求的响应模型。"""
    status: str
    message: str
    results: Dict[str, Any] = {}
    failed: List[Dict[str, str]] = []


@router.post("/batch", response_model=BatchParseResponse)
async def batch_parse(
    request: BatchParseRequest,
    background_tasks: BackgroundTasks = None,
) -> BatchParseResponse:
    """
    批量解析多个文档。
    
    参数:
        request: 包含文件路径的 BatchParseRequest
        background_tasks: 后台任务
        
    返回:
        BatchParseResponse 对象
    """
    results = {}
    failed = []
    temp_files = []  # 跟踪临时文件以便清理
    
    try:
        for file_path in request.file_paths:
            try:
                # 检查文件是否存在
                path = Path(file_path)
                if not path.exists():
                    failed.append({
                        "file_path": file_path,
                        "error": "文件未找到"
                    })
                    continue
                
                # 获取文件扩展名并确定是否需要转换
                file_ext = path.suffix.lower()
                is_pdf = file_ext == '.pdf'
                
                # 如果不是 PDF，则转换为 PDF
                pdf_path = file_path
                if not is_pdf:
                    logger.info(f"文件不是 PDF，正在转换: {file_path}")
                    try:
                        pdf_path = convert_to_pdf(str(path.absolute()))
                        temp_files.append(pdf_path)
                    except Exception as e:
                        logger.exception(f"转换文件为 PDF 失败: {str(e)}")
                        failed.append({
                            "file_path": file_path,
                            "error": f"转换文件为 PDF 失败: {str(e)}"
                        })
                        continue
                
                # 解析 PDF 文件
                config = MinerUParserConfig.from_env()
                parser = MinerUParser(api_url=config.api_url, timeout=config.timeout)
                documents = parser.parse(pdf_path)
                
                # 如果请求，应用层次化切片
                if request.use_hierarchical:
                    all_chunked_docs = []
                    for doc in documents:
                        chunked_docs = apply_hierarchical_chunking(doc)
                        all_chunked_docs.extend(chunked_docs)
                    documents = all_chunked_docs
                
                # 将文档转换为字典以用于 API 响应
                doc_dicts = []
                for doc in documents:
                    doc_dict = {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    doc_dicts.append(doc_dict)
                
                # 添加到结果中
                results[file_path] = {
                    "status": "success",
                    "documents": doc_dicts
                }
                
            except Exception as e:
                logger.exception(f"解析文档 {file_path} 失败: {str(e)}")
                failed.append({
                    "file_path": file_path,
                    "error": str(e)
                })
        
        return BatchParseResponse(
            status="completed",
            message=f"处理了 {len(request.file_paths)} 个文件: {len(results)} 个成功, {len(failed)} 个失败",
            results=results,
            failed=failed
        )
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"删除临时文件 {temp_file} 失败: {str(e)}") 