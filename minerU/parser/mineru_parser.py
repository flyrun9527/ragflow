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
import re
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
                'return_images': True,   
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
        
    def _save_images_from_result(self, result, images_dir):
        """从 MinerU API 结果中保存图片到临时目录
        
        参数:
            result: MinerU API 返回的结果
            images_dir: 保存图片的临时目录路径
            
        返回:
            保存的图片数量
        """
        saved_count = 0
        
        if not result or not isinstance(result, dict) or 'images' not in result or not result['images']:
            logger.warning(f"API结果中没有images字段或为空")
            return 0
            
        os.makedirs(images_dir, exist_ok=True)
        logger.info(f"创建/确认临时图片目录: {images_dir}")
        
        for image_name, image_data in result['images'].items():
            try:
                if not image_data:
                    logger.warning(f"图片 {image_name} 的数据为空")
                    continue
                    
                # 提取 base64 数据（去掉 data:image/jpeg;base64, 前缀）
                base64_data = image_data
                if isinstance(image_data, str):
                    if image_data.startswith('data:image/'):
                        # 找到逗号后的base64数据
                        comma_index = image_data.find(',')
                        if comma_index != -1:
                            base64_data = image_data[comma_index + 1:]
                        else:
                            logger.warning(f"图片 {image_name} 的data URL格式不正确，缺少逗号分隔符")
                            continue
                    else:
                        base64_data = image_data
                    
                    # 清理base64字符串中的空白字符
                    base64_data = base64_data.strip()
                    
                # 解码并保存图片
                try:
                    image_bytes = base64.b64decode(base64_data)
                    if not image_bytes:
                        logger.warning(f"图片 {image_name} 解码后数据为空")
                        continue
                        
                    # 验证图片数据长度
                    if len(image_bytes) < 100:  # 100字节是一个合理的最小图片大小
                        logger.warning(f"图片 {image_name} 数据太小，可能不是有效图片: {len(image_bytes)} 字节")
                        continue
                        
                    image_path = os.path.join(images_dir, image_name)
                    
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                        
                    saved_count += 1
                    logger.info(f"保存图片: {image_path}")
                except Exception as decode_err:
                    logger.error(f"解码或保存图片 {image_name} 数据失败: {decode_err}")
                    continue
                
            except Exception as e:
                logger.error(f"处理图片 {image_name} 失败: {e}")
        
        logger.info(f"总共保存了 {saved_count} 张图片到 {images_dir}")
        return saved_count
    
    def _upload_images_to_minio(self, kb_id, images_dir):
        """将图片上传到 MinIO 对象存储
        
        参数:
            kb_id: 知识库ID，用作 MinIO bucket 名称
            images_dir: 图片所在的本地临时目录
            
        返回:
            上传成功的图片数量
        """
        try:
            from rag.utils.storage_factory import STORAGE_IMPL
            
            if not os.path.exists(images_dir) or not os.path.isdir(images_dir):
                logger.error(f"图片目录不存在或不是目录: {images_dir}")
                return 0
                
            # 检查目录中是否有图片文件
            image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)) 
                           and os.path.splitext(f.lower())[1] in ('.png', '.jpg', '.jpeg', '.gif', '.webp')]
            
            if not image_files:
                logger.warning(f"图片目录 {images_dir} 中没有图片文件")
                return 0
            
            # 不再检查桶是否存在，直接上传 - STORAGE_IMPL会在put时创建桶
            success_count = 0
            total_count = len(image_files)
            
            # 上传目录中的所有图片
            for img_file in image_files:
                img_path = os.path.join(images_dir, img_file)
                
                try:
                    # 读取图片文件并完整加载到内存中
                    with open(img_path, 'rb') as f:
                        img_data = f.read()
                    
                    # 确保数据已读取
                    if img_data is None or len(img_data) == 0:
                        logger.error(f"无法读取图片数据: {img_file}")
                        continue
                    
                    # 确定内容类型
                    ext = os.path.splitext(img_file)[1].lower()
                    content_type = f"image/{ext[1:]}"
                    if content_type == "image/jpg":
                        content_type = "image/jpeg"
                    
                    # 上传到 MinIO - 使用项目的API
                    # 注意：项目中RAGFlowMinio.put会自己创建BytesIO，所以这里直接传递字节数据
                    try:
                        STORAGE_IMPL.put(kb_id, img_file, img_data)
                        success_count += 1
                        logger.info(f"成功上传图片到 MinIO: {img_file}")
                    except Exception as put_err:
                        logger.error(f"上传图片到MinIO失败: {put_err}")
                        
                except Exception as e:
                    logger.error(f"读取图片 {img_file} 失败: {e}")
                    
            logger.info(f"上传到 MinIO 完成: 成功 {success_count}/{total_count}")
            return success_count
            
        except ImportError as ie:
            logger.error(f"无法导入STORAGE_IMPL: {ie}")
            return 0
        except Exception as e:
            logger.error(f"上传图片到MinIO时出错: {e}")
            return 0
    
    def _get_image_url(self, kb_id, image_name):
        """生成MinIO图片访问URL
        
        参数:
            kb_id: 知识库ID
            image_name: 图片名称
            
        返回:
            图片访问URL
        """
        try:
            from rag.utils.storage_factory import STORAGE_IMPL
            
            # 优先尝试从配置文件构建永久URL（最稳定的方法）
            try:
                from rag import settings
                minio_host = settings.MINIO.get("host", "localhost:9000")
                secure = settings.MINIO.get("secure", False)
                protocol = "https" if secure else "http"
                url = f"{protocol}://{minio_host}/{kb_id}/{image_name}"
                logger.debug(f"从配置构建永久URL: {url}")
                return url
            except Exception as e:
                logger.warning(f"从配置构建URL失败，尝试其他方法: {e}")
            
            # 备选方案1: 尝试get_url（如果存在的话，通常是永久URL）
            # try:
            #     if hasattr(STORAGE_IMPL, 'get_url'):
            #         url = STORAGE_IMPL.get_url(kb_id, image_name)
            #         logger.debug(f"获取URL成功(get_url): {url}")
            #         return url
            # except Exception as e:
            #     logger.warning(f"get_url方法失败: {e}")
            
            # 最后返回原始图片名（可能在某些环境下仍然有效）
            logger.warning(f"所有URL获取方法都失败，使用原始图片名: {image_name}")
            return image_name
            
        except Exception as e:
            logger.error(f"获取图片URL时发生异常: {e}")
            return image_name
    
    def _update_markdown_image_urls(self, markdown_content, kb_id):
        """更新Markdown内容中的图片URL
        
        参数:
            markdown_content: 原始Markdown内容
            kb_id: 知识库ID
            
        返回:
            更新后的Markdown内容
        """
        def _replace_img(match):
            img_path = match.group(1)  # 获取图片路径
            img_name = os.path.basename(img_path)
            
            # 只处理本地图片路径，不处理已经是URL的图片
            if not img_path.startswith(('http://', 'https://')):
                # img_url = self._get_image_url(kb_id, img_name) # 别删除 该地址获取minio地址 但是需要web访问所以采用前端代理 /minio方式
                img_url = f"/minio/{kb_id}/{img_name}"
                logger.info(f"需要替换img_path：{img_path}，img_url: {img_url}")
                return f'<img src="{img_url}" style="max-width: 300px;" alt="图片">'
            else:
                # 已经是URL的图片也转换为HTML标签
                return f'<img src="{img_path}" style="max-width: 300px;" alt="图片">'
        
        try:
            # 匹配MinerU生成的两种格式：![](文件名) 或 ![图片](文件名)
            updated_content = re.sub(r'!\[(?:图片)?\]\((.*?)\)', _replace_img, markdown_content)
            
            if updated_content != markdown_content:
                logger.info(f"已更新Markdown中的图片URL，kb_id: {kb_id}")
            else:
                logger.warning(f"没有图片链接被替换，kb_id: {kb_id}")
                
            return updated_content
            
        except Exception as e:
            logger.error(f"更新Markdown图片URL失败: {e}")
            return markdown_content
    
    def parse(self, file_path: str, kb_id: str = None, doc_id: str = None) -> List[Dict[str, Any]]:
        """
        使用 MinerU 解析 PDF 文件并转换为文档对象。
        
        参数:
            file_path: PDF 文件路径
            kb_id: 知识库ID，用于图片上传到MinIO
            doc_id: 文档ID，用于创建唯一的临时目录
            
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
        documents = self._convert_to_documents(result, file_path, kb_id, doc_id)
        
        logger.info(f"使用 MinerU 成功解析 PDF: {file_path}，kb_id: {kb_id}")
        return documents
    
    def _convert_to_documents(self, result: Dict[str, Any], file_path: str, kb_id: str = None, doc_id: str = None) -> List[Dict[str, Any]]:
        """
        将 MinerU 解析结果转换为文档对象。
        
        参数:
            result: MinerU 解析结果
            file_path: 原始文件路径
            kb_id: 知识库ID，用于图片上传到MinIO
            doc_id: 文档ID，用于创建唯一的临时目录
            
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
        
        # 处理图片 - 优先检查顶层result，然后检查doc_content
        if kb_id:
            images_data = None
            if 'images' in result and result['images']:
                images_data = result['images']
                logger.info(f"从顶层result中找到images字段")
            elif 'images' in doc_content and doc_content['images']:
                images_data = doc_content['images']
                logger.info(f"从doc_content中找到images字段")
            
            if images_data:
                try:
                    # 处理images并更新markdown_content
                    processed_content = self._process_images(images_data, markdown_content, kb_id, doc_id)
                    if processed_content != markdown_content:
                        markdown_content = processed_content
                        logger.info("已更新markdown内容中的图片")
                except Exception as e:
                    logger.error(f"处理图片时出错: {e}")
            else:
                logger.warning(f"提供了kb_id({kb_id})，但未找到图片数据")
        
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

    def _process_images(self, images, markdown_content, kb_id, doc_id=None):
        """处理图片的辅助方法
        
        参数:
            images: 图片数据字典，键为图片名称，值为base64编码的图片数据
            markdown_content: 原始markdown内容
            kb_id: 知识库ID
            doc_id: 文档ID，用于创建唯一的临时目录
            
        返回:
            更新后的markdown内容
        """
        if not images:
            logger.warning("images参数为空，无法处理图片")
            return markdown_content
            
        # 创建基于doc_id的临时目录，防止高并发情况下的冲突
        temp_dir_id = doc_id if doc_id else f"mineru_{int(time.time())}_{os.getpid()}"
        temp_base_dir = os.path.join(tempfile.gettempdir(), f"ragflow_{temp_dir_id}")
        temp_images_dir = os.path.join(temp_base_dir, "images")
        
        # 创建临时目录结构
        os.makedirs(temp_images_dir, exist_ok=True)
        logger.info(f"创建临时目录结构: {temp_base_dir}")
        
        try:
            # 记录图片名称，用于后续检查
            image_names = list(images.keys())
            logger.info(f"需要处理的图片: {len(image_names)} 个")
            
            # 检查markdown中的图片引用 - 使用与_update_markdown_image_urls一致的正则
            image_refs = re.findall(r'!\[(?:图片)?\]\((.*?)\)', markdown_content)
            ref_names = [os.path.basename(path) for path in image_refs]
            logger.info(f"Markdown中引用的图片: {len(ref_names)} 个")
            
            # 检查哪些图片没有被引用
            not_referenced = set(image_names) - set(ref_names)
            if not_referenced:
                logger.warning(f"有 {len(not_referenced)} 张图片未在markdown中被引用")
            
            # 保存图片到临时目录
            saved_count = self._save_images_from_result({'images': images}, temp_images_dir)
            logger.info(f"保存了 {saved_count} 张图片到临时目录")
            
            # 上传图片到MinIO并更新markdown
            updated_content = markdown_content
            if saved_count > 0:
                # 上传图片到MinIO
                uploaded_count = self._upload_images_to_minio(kb_id, temp_images_dir)
                logger.info(f"上传了 {uploaded_count} 张图片到MinIO")
                
                # 只有在成功上传图片后才更新markdown中的图片链接
                if uploaded_count > 0:
                    # 更新markdown中的图片链接
                    updated_content = self._update_markdown_image_urls(markdown_content, kb_id)
                    
                    # 将未引用的图片添加到markdown末尾（使用HTML标签格式） 暂时不需要添加附加图片（没有图片的大概率是表格图片）
                    # if not_referenced:
                    #     additional_content = "\n\n## 附加图片\n\n"
                    #     for img_name in not_referenced:
                    #         img_url = self._get_image_url(kb_id, img_name)
                    #         additional_content += f'<img src="{img_url}" style="max-width: 300px;" alt="{img_name}">\n\n'
                    #     updated_content += additional_content
                    #     logger.info(f"已将 {len(not_referenced)} 张未引用的图片添加到markdown末尾")
            
        except Exception as e:
            logger.error(f"处理图片过程中出错: {str(e)}")
            # 在出错时返回原始内容
            return markdown_content
        finally:
            # 清理临时目录
            try:
                import shutil
                shutil.rmtree(temp_base_dir, ignore_errors=True)
                logger.debug(f"已清理临时目录结构: {temp_base_dir}")
            except Exception as e:
                logger.warning(f"清理临时目录失败: {e}")
        
        return updated_content
    
    def __call__(self, filename_or_binary, binary=None, from_page=None, to_page=None, callback=None, kb_id=None, doc_id=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        解析 PDF 文件并返回文档块和表格。
        
        参数:
            filename_or_binary: 文件名或二进制内容
            binary: 二进制内容（如果 filename_or_binary 是文件名）
            from_page: 起始页码
            to_page: 结束页码
            callback: 回调函数，用于报告进度
            kb_id: 知识库ID，用于从minio对象存储中读取文件和存储图片
            doc_id: 文档ID，用于从minio对象存储中读取文件和创建唯一的临时目录
            
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
                
            # 确保kb_id和doc_id被传递给parse方法
            documents = self.parse(pdf_to_process, kb_id, doc_id)
            
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
            
            # 清理转换后的PDF文件
            if temp_pdf_to_delete and temp_pdf_to_delete != temp_file_path and os.path.exists(temp_pdf_to_delete):
                try:
                    os.unlink(temp_pdf_to_delete)
                except:
                    pass
            
            # 删除临时目录
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass 