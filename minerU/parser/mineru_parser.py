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
import logging
import time
from pathlib import Path
import tempfile
import json
import base64
from typing import Dict, List, Optional, Tuple, Any, Union

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from minerU.utils.file_converter import ensure_pdf

from .config import MinerUParserConfig

# 移除对 rag.schema 的导入
# from rag.schema.document import Document
# from rag.schema.multi_modal import (
#     AudioSegment,
#     Formula,
#     Image,
#     Link,
#     MarkdownFragment,
#     Table,
# )


logger = logging.getLogger(__name__)


class MinerUParserError(Exception):
    """MinerU 解析器错误的异常类。"""
    pass


class MinerUAPIClient:
    """与 MinerU API 交互的客户端。"""

    def __init__(self, api_url: str, timeout: int = 300):
        """初始化 MinerU API 客户端。
        
        参数:
            api_url: MinerU API 的 URL
            timeout: API 请求的超时时间（秒）
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def parse_pdf(self, file_path: str, backend: str = "vlm-sglang-client", language: str = "ch") -> Dict[str, Any]:
        """
        提交 PDF 文件到 MinerU 进行解析。
        
        参数:
            file_path: PDF 文件路径
            backend: 使用的后端引擎
            language: OCR 识别的语言
            
        返回:
            包含解析内容的字典
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        
        # 准备请求数据
        data = {
                'output_dir': './output',
                'lang_list': 'ch',
                'backend': 'vlm-sglang-client',
                'parse_method': 'auto',
                'formula_enable': True,  
                'table_enable': True,    
                'server_url': 'http://192.168.130.24:30000',
                'return_md': True,       
                'return_middle_json': True,  
                'return_model_output': False,  
                'return_content_list': False,  
                'return_images': False,   
                'start_page_id': 0,       
                'end_page_id': 99999     
            }
            
        # 准备表单数据
        with open(file_path, 'rb') as f:
            files = {'files': f}
            
            # 提交请求到 MinerU API
            try:
                # 添加详细的请求信息
                session = requests.Session()
                session.trust_env = False  # 禁用代理环境变量
                
                # 发送请求
                response = session.post(
                    f"{self.api_url}/file_parse", 
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
                
                # 打印响应信息
                logger.debug(f"响应状态码: {response.status_code}")
                logger.debug(f"响应头: {response.headers}")
                
                # 检查响应状态
                if response.status_code != 200:
                    logger.info(f"向 MinerU API 提交文件: {file_path}")
                    logger.debug(f"API URL: {self.api_url}/file_parse")
                    logger.debug(f"请求参数: {data}")
                    logger.error(f"API返回错误状态码: {response.status_code}")
                    logger.error(f"响应内容: {response.text[:500]}")
                    response.raise_for_status()
                
                # 解析JSON响应
                try:
                    result = response.json()
                    logger.info(f"MinerU API 返回结果: {result}")
                    return result
                except ValueError as e:
                    logger.error(f"无法解析API响应为JSON: {response.text[:500]}")
                    raise MinerUParserError(f"无法解析API响应为JSON: {str(e)}")
                
            except requests.RequestException as e:
                logger.error(f"与 MinerU API 通信错误: {str(e)}")
                raise MinerUParserError(f"与 MinerU API 通信错误: {str(e)}")


class MinerUParser:
    """MinerU PDF 解析器实现。"""

    def __init__(self, api_url: Optional[str] = None, timeout: int = 300, config: Optional[MinerUParserConfig] = None, s3_config: Optional[Dict[str, Any]] = None):
        """初始化 MinerU 解析器。
        
        参数:
            api_url: MinerU API 的 URL
            timeout: API 请求的超时时间（秒）
            config: MinerU 解析器配置
            s3_config: S3配置，包含endpoint_url、access_key、secret_key等
        """
        # 优先使用传入的配置
        if config is None:
            config = MinerUParserConfig.from_env()
        
        # 使用提供的 API URL 或从配置获取
        self.api_url = api_url or config.api_url
        self.timeout = timeout or config.timeout
        self.backend = config.backend
        self.language = config.language
        
        # S3配置
        self.s3_config = s3_config or {}
        
        # 初始化 API 客户端
        self.client = MinerUAPIClient(self.api_url, self.timeout)
        
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        使用 MinerU 解析 PDF 文件并转换为文档对象。
        
        参数:
            file_path: PDF 文件路径
            
        返回:
            文档对象列表
        """
        logger.info(f"使用 MinerU 解析 PDF: {file_path}")
        
        # 从 MinerU 获取原始解析结果
        try:
            result = self.client.parse_pdf(file_path, self.backend, self.language)
        except MinerUParserError as e:
            logger.error(f"使用 MinerU 解析 PDF 失败: {str(e)}")
            raise  # 直接重新抛出 MinerUParserError
        except Exception as e:
            logger.error(f"使用 MinerU 解析 PDF 失败: {str(e)}")
            raise
        
        # 将 MinerU 结果转换为文档对象
        documents = self._convert_to_documents(result, file_path)
        
        logger.info(f"使用 MinerU 成功解析 PDF: {file_path}")
        return documents
    
    def _convert_to_documents(self, result: Dict[str, Any], file_path: str) -> List[Dict[str, Any]]:
        """
        将 MinerU 解析结果转换为文档对象。
        
        参数:
            result: MinerU 解析结果
            file_path: 原始文件路径
            
        返回:
            文档对象列表
        """
        documents = []
        
        # 检查返回结果格式
        if not isinstance(result, dict) or 'results' not in result:
            logger.warning(f"{file_path} 返回结果格式不正确")
            return []
        
        # 获取结果中的第一个文档（通常只有一个）
        results = result.get('results', {})
        if not results:
            logger.warning(f"{file_path} 没有解析结果")
            return []
            
        # 获取第一个文档的键（通常是文件名）
        doc_key = next(iter(results.keys()), None)
        if not doc_key:
            logger.warning(f"{file_path} 解析结果中没有文档")
            return []
            
        # 获取文档内容
        doc_content = results[doc_key]
        
        # 提取 markdown 内容
        markdown_content = doc_content.get('md_content', '')
        
        if not markdown_content:
            logger.warning(f"{file_path} 未返回 markdown 内容")
            return []
        
        # 创建文档对象
        doc = {
            "page_content": markdown_content,
            "metadata": {
                "source": file_path,
                "parser": "mineru",
                "title": Path(file_path).stem,
            }
        }
        
        # 将文档添加到列表中
        documents.append(doc)
        
        return documents
    
    def __call__(self, filename_or_binary, binary=None, from_page=None, to_page=None, callback=None, kb_id=None, doc_id=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        解析 PDF 文件并返回文档块和表格。
        
        参数:
            filename_or_binary: 文件名或二进制内容
            binary: 二进制内容（如果 filename_or_binary 是文件名）
            from_page: 起始页码
            to_page: 结束页码
            callback: 回调函数，用于报告进度
            kb_id: 知识库ID，用于从minio对象存储中读取文件
            doc_id: 文档ID，用于从minio对象存储中读取文件
            
        返回:
            (文档块列表, 表格列表)
        """
        # 创建临时目录用于处理文件
        temp_dir = tempfile.mkdtemp(prefix="mineru_parser_")
        temp_file_path = None
        pdf_to_process = None
        temp_pdf_to_delete = None
        
        try:
            # 如果提供了kb_id和doc_id，从minio对象存储中读取文件
            if kb_id and doc_id:
                if callback:
                    callback(prog=0.1, msg="从minio对象存储中读取文件")
                
                try:
                    # 从Minio读取文件
                    from rag.utils.storage_factory import STORAGE_IMPL
                    from api.db.services.file2document_service import File2DocumentService
                    from api.db.services.document_service import DocumentService
                    
                    # 获取文档信息
                    e, doc = DocumentService.get_by_id(doc_id)
                    if not e:
                        raise Exception(f"找不到文档: {doc_id}")
                    
                    # 获取文件的存储位置
                    bucket, location = File2DocumentService.get_storage_address(doc_id=doc_id)
                    
                    # 读取文件
                    logger.info(f"从minio读取文件: {bucket}/{location}")
                    file_bytes = STORAGE_IMPL.get(bucket, location)
                    
                    # 确定文件扩展名
                    ext = os.path.splitext(doc.name)[1]
                    if not ext:
                        ext = '.pdf'  # 默认使用PDF扩展名
                    
                    # 保存到临时文件
                    temp_file_path = os.path.join(temp_dir, f"temp_file{ext}")
                    with open(temp_file_path, 'wb') as f:
                        f.write(file_bytes)
                    
                    logger.info(f"成功从minio读取文件并保存到临时文件: {temp_file_path}")
                    
                    # 检查是否需要转换为PDF
                    if not ext.lower() == '.pdf':
                        if callback:
                            callback(prog=0.15, msg="转换文件为PDF格式")
                        pdf_to_process, temp_pdf_to_delete = ensure_pdf(temp_file_path, temp_dir)
                    else:
                        pdf_to_process = temp_file_path
                        temp_pdf_to_delete = temp_file_path
                
                except Exception as e:
                    logger.error(f"从minio读取或处理文件失败: {str(e)}")
                    raise MinerUParserError(f"从minio读取或处理文件失败: {str(e)}")
            # 处理二进制内容
            elif binary is not None:
                # 保存二进制内容到临时文件
                temp_file_path = os.path.join(temp_dir, "temp_file")
                
                # 检查内容类型并添加适当的扩展名
                if binary.startswith(b'%PDF'):
                    temp_file_path += '.pdf'
                else:
                    # 默认保存为二进制文件，稍后会尝试转换
                    temp_file_path += '.bin'
                
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(binary)
                
                if callback:
                    callback(prog=0.1, msg="二进制内容已保存到临时文件")
                
                # 尝试转换为PDF（如果需要）
                pdf_to_process, temp_pdf_to_delete = ensure_pdf(temp_file_path, temp_dir, callback, self.s3_config)
                
                if not pdf_to_process:
                    raise MinerUParserError("无法处理文件，转换为PDF失败")
            # 处理文件路径
            else:
                file_path = filename_or_binary
                
                # 检查本地文件是否存在
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"文件未找到: {file_path}")
                
                if callback:
                    callback(prog=0.1, msg="文件路径检查完成")
                
                # 尝试转换为PDF（如果需要）
                pdf_to_process, temp_pdf_to_delete = ensure_pdf(file_path, temp_dir, callback, self.s3_config)
                
                if not pdf_to_process:
                    raise MinerUParserError(f"无法处理文件: {file_path}，转换为PDF失败")
            
            # 解析 PDF 文件
            if callback:
                callback(prog=0.2, msg="开始解析PDF文件")
                
            documents = self.parse(pdf_to_process)
            
            if callback:
                callback(prog=0.8, msg="PDF解析完成")
            
            # 提取文档块和表格
            sections = []
            tables = []
            
            for doc in documents:
                # 添加文档块
                sections.append({
                    "text": doc["page_content"],
                    "metadata": doc["metadata"]
                })
            
            if callback:
                callback(prog=0.9, msg="文档处理完成")
                
            logger.info(f"MinerUParser.__call__ 返回 {len(sections)} 个文档块和 {len(tables)} 个表格")
            return sections, tables
            
        except FileNotFoundError as e:
            logger.error(f"MinerUParser.__call__ 失败: {str(e)}")
            raise  # 直接重新抛出 FileNotFoundError
        except Exception as e:
            logger.error(f"MinerUParser.__call__ 失败: {str(e)}")
            raise
        finally:
            # 清理临时文件和目录
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            # 删除临时目录
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass 