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
import base64
from typing import Dict, List, Optional, Tuple, Any, Union

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from api.constants import RAG_FLOW_SERVICE_NAME
from api.utils import get_base_config
from minerU.utils.file_converter import ensure_pdf

from .config import MinerUParserConfig

logger = logging.getLogger(__name__)


class MinerUParserError(Exception):
    """MinerU 解析器错误的异常类。"""
    pass


class MinerUParser:
    """MinerU PDF 解析器实现。
    
    使用远程API服务解析PDF文件，处理返回的Markdown内容和图片。
    """

    def __init__(self, api_url: Optional[str] = None, timeout: int = 300, 
                 config: Optional[MinerUParserConfig] = None, s3_config: Optional[Dict[str, Any]] = None):
        """初始化 MinerU 解析器。
        
        参数:
            api_url: MinerU API 的 URL
            timeout: API 请求的超时时间（秒）
            config: MinerU 解析器配置
            s3_config: S3配置，包含endpoint_url、access_key、secret_key等
        """
        # 配置初始化
        if config is None:
            config = MinerUParserConfig.from_env()
        
        self.api_url = api_url or config.api_url
        self.server_url= config.server_url
        self.timeout = timeout or config.timeout
        self.backend = config.backend
        self.language = config.language
        self.s3_config = s3_config or {}

    def __call__(self, filename_or_binary, binary=None, from_page=None, to_page=None, 
                 callback=None, kb_id=None, doc_id=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """解析PDF文件并返回文档块和表格。
        
        参数:
            filename_or_binary: 文件名或二进制内容
            binary: 二进制内容（如果filename_or_binary是文件名）
            from_page: 起始页码
            to_page: 结束页码
            callback: 进度回调函数
            kb_id: 知识库ID
            doc_id: 文档ID
            
        返回:
            (文档块列表, 表格列表)
        """
        # 创建临时目录用于文档处理
        temp_dir = tempfile.mkdtemp(prefix="mineru_parser_")
        source_file_path = None
        pdf_file_path = None
        temp_pdf_to_delete = None
        is_temp_source = False
        
        try:
            # 1. 准备源文件 - 根据输入类型获取文件
            if callback:
                callback(prog=0.05, msg="准备源文件")
                
            # 处理二进制内容
            if binary is not None:
                # 传递原始文件名，便于提取扩展名
                original_filename = filename_or_binary if isinstance(filename_or_binary, str) else None
                source_file_path = self._save_binary_to_temp(binary, temp_dir, original_filename)
                is_temp_source = True
                
            # 处理MinIO存储的文件    
            elif kb_id and doc_id:
                source_file_path = self._get_file_from_minio(kb_id, doc_id, temp_dir)
                is_temp_source = True
                
            # 处理本地文件    
            else:
                # 对于本地文件，直接使用原始路径
                if not os.path.exists(filename_or_binary):
                    raise FileNotFoundError(f"文件未找到: {filename_or_binary}")
                source_file_path = filename_or_binary
                is_temp_source = False
                
            if callback:
                callback(prog=0.1, msg="文件准备完成")
                
            # 2. 确保文件格式为PDF - 必要时进行转换
            if callback:
                callback(prog=0.15, msg="检查文档格式并转换")
                
            logger.info(f"检查文档格式并转换: {source_file_path}")
            pdf_file_path, temp_pdf_to_delete = ensure_pdf(source_file_path, temp_dir)
            
            if not pdf_file_path:
                raise Exception(f"无法处理文件: {source_file_path}，转换为PDF失败")
            
            if temp_pdf_to_delete:
                logger.info(f"文档已转换为PDF: {pdf_file_path}")
                if callback:
                    callback(prog=0.25, msg="文档转换完成")
            else:
                logger.info(f"文档已是PDF格式: {pdf_file_path}")
                if callback:
                    callback(prog=0.25, msg="PDF文件检查完成")
            
            # 3. 调用解析API处理PDF
            if callback:
                callback(prog=0.3, msg="开始解析PDF文件")
                
            documents = self._parse_pdf(
                pdf_path=pdf_file_path, 
                kb_id=kb_id, 
                doc_id=doc_id
            )
            
            if callback:
                callback(prog=0.8, msg="PDF解析完成")
            
            # 4. 提取文档块
            sections = []
            tables = []
            
            for doc in documents:
                sections.append({
                    "text": doc["page_content"],
                    "metadata": doc["metadata"]
                })
                
            if callback:
                callback(prog=1.0, msg="文档处理完成")
                
            return sections, tables
            
        except Exception as e:
            logger.error(f"PDF解析失败: {str(e)}")
            raise
        finally:
            # 清理临时PDF文件
            if temp_pdf_to_delete and os.path.exists(temp_pdf_to_delete):
                try:
                    os.remove(temp_pdf_to_delete)
                    logger.info(f"已清理临时PDF文件: {temp_pdf_to_delete}")
                except OSError as e:
                    logger.warning(f"清理临时PDF文件失败: {temp_pdf_to_delete}, 错误: {e}")
            
            # 清理临时源文件（如果是临时创建的）
            if is_temp_source and source_file_path and os.path.exists(source_file_path):
                try:
                    os.remove(source_file_path)
                    logger.info(f"已清理临时源文件: {source_file_path}")
                except OSError as e:
                    logger.warning(f"清理临时源文件失败: {source_file_path}, 错误: {e}")
            
            # 清理临时目录
            try:
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug(f"已清理临时目录: {temp_dir}")
            except OSError as e:
                logger.warning(f"清理临时目录失败: {temp_dir}, 错误: {e}")
                
            # 检查临时文件是否都已清理
            if temp_pdf_to_delete and os.path.exists(temp_pdf_to_delete):
                logger.warning(f"临时PDF文件未被删除: {temp_pdf_to_delete}")
                
            if is_temp_source and source_file_path and os.path.exists(source_file_path):
                logger.warning(f"临时源文件未被删除: {source_file_path}")
                
            if os.path.exists(temp_dir):
                logger.warning(f"临时目录未被删除: {temp_dir}")

    def _get_file_from_minio(self, kb_id, doc_id, temp_dir):
        """从MinIO读取文件。
        
        参数:
            kb_id: 知识库ID
            doc_id: 文档ID
            temp_dir: 临时目录
            
        返回:
            str: 临时文件路径
            
        异常:
            MinerUParserError: 读取失败时抛出
        """
        try:
            from rag.utils.storage_factory import STORAGE_IMPL
            from api.db.services.file2document_service import File2DocumentService
            from api.db.services.document_service import DocumentService
            
            # 获取文档信息
            _, doc = DocumentService.get_by_id(doc_id)
            if not doc:
                raise MinerUParserError(f"找不到文档: {doc_id}")
            
            # 获取文件存储位置
            bucket, location = File2DocumentService.get_storage_address(doc_id=doc_id)
            
            # 读取文件
            file_bytes = STORAGE_IMPL.get(bucket, location)
            
            # 确定文件扩展名
            ext = os.path.splitext(doc.name)[1] or '.pdf'
            
            # 保存到临时文件
            temp_file_path = os.path.join(temp_dir, f"temp_file{ext}")
            with open(temp_file_path, 'wb') as f:
                f.write(file_bytes)
            
            logger.info(f"已从MinIO读取文件到: {temp_file_path}")
            return temp_file_path
            
        except Exception as e:
            logger.error(f"从MinIO读取文件失败: {str(e)}")
            raise MinerUParserError(f"从MinIO读取文件失败: {str(e)}")

    def _parse_pdf(self, pdf_path, kb_id, doc_id):
        """解析PDF文件，提取内容并转换为文档对象。
        
        参数:
            pdf_path: PDF文件的路径
            kb_id: 知识库ID（用于图片处理）
            doc_id: 文档ID（用于图片处理）
            
        返回:
            List[Dict]: 包含解析结果的文档对象列表
            
        异常:
            MinerUParserError: 解析失败时抛出
        """
        # 1. 调用API解析PDF
        api_result = self._api_parse_pdf(file_path=pdf_path)
        
        # 2. 将API结果转换为文档对象
        return self._convert_to_documents(
            result=api_result, 
            file_path=pdf_path, 
            kb_id=kb_id, 
            doc_id=doc_id
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _api_parse_pdf(self, file_path):
        """调用MinerU API解析PDF文件，支持自动重试。
        
        参数:
            file_path: PDF文件的路径
            
        返回:
            Dict: API返回的解析结果
            
        异常:
            MinerUParserError: API调用失败或解析响应失败时抛出
            requests.exceptions.RequestException: 网络请求失败时抛出
        """
        # 1. 准备请求参数
        request_params = {
            'output_dir': './output',
            'lang_list': self.language,
            'backend': self.backend,
            'parse_method': 'auto',
            'server_url': self.server_url,
            'formula_enable': True,  
            'table_enable': True,
            'return_md': True,
            'return_middle_json': True,
            'return_model_output': False,
            'return_content_list': False,
            'return_images': True,
            'start_page_id': 0,
            'end_page_id': 99999
        }
        
        # 2. 发送API请求
        with open(file_path, 'rb') as file_object:
            files = {'files': file_object}
            
            # 创建会话并配置
            session = requests.Session()
            session.trust_env = False
            
            # 发送请求
            logger.info(f"发送请求到MinerU API: {self.api_url}")
            logger.info(f"请求参数: {request_params}")
            response = session.post(
                url=f"{self.api_url.rstrip('/')}/file_parse", 
                files=files,
                data=request_params,
                timeout=self.timeout
            )
            
            # 3. 检查响应状态
            if response.status_code != 200:
                error_msg = f"API返回错误状态码: {response.status_code}"
                logger.error(error_msg)
                response.raise_for_status()
            
            # 4. 解析JSON响应
            try:
                result = response.json()
                return result
            except ValueError as e:
                error_msg = f"无法解析API响应为JSON: {str(e)}"
                logger.error(error_msg)
                raise MinerUParserError(error_msg)

    def _convert_to_documents(self, result, file_path, kb_id, doc_id):
        """将API结果转换为文档对象，处理Markdown内容和图片。
        
        参数:
            result: API返回的解析结果
            file_path: 原始文件路径
            kb_id: 知识库ID（用于图片处理）
            doc_id: 文档ID（用于图片处理）
            
        返回:
            List[Dict]: 包含处理后内容的文档对象列表
        """
        documents = []
        
        # 1. 验证结果格式
        if not isinstance(result, dict) or 'results' not in result:
            logger.warning(f"文件 {file_path} 的返回结果格式不正确")
            return []
        
        # 2. 获取文档内容
        results_data = result.get('results', {})
        if not results_data:
            logger.warning(f"文件 {file_path} 的返回结果为空")
            return []
            
        # 3. 获取第一个文档的键
        doc_key = next(iter(results_data.keys()), None)
        if not doc_key:
            logger.warning(f"文件 {file_path} 的返回结果中未找到文档键")
            return []
            
        # 4. 提取Markdown内容
        doc_content = results_data[doc_key]
        markdown_content = doc_content.get('md_content', '')
        
        if not markdown_content:
            logger.warning(f"文件 {file_path} 未返回Markdown内容")
            return []
        
        # 5. 处理图片（如果有）
        if kb_id:
            images_data = result.get('images') or doc_content.get('images')
            if images_data:
                # 处理图片并更新Markdown中的图片链接
                markdown_content = self._process_images(
                    images=images_data,
                    markdown_content=markdown_content,
                    kb_id=kb_id,
                    doc_id=doc_id
                )
        
        # 6. 创建文档对象
        document = {
            "page_content": markdown_content,
            "metadata": {
                "source": file_path,
                "parser": "mineru",
                "title": Path(file_path).stem,
            }
        }
        
        documents.append(document)
        return documents

    def _process_images(self, images, markdown_content, kb_id, doc_id=None):
        """处理图片并更新Markdown内容中的图片引用。
        
        参数:
            images: 图片数据字典，键为图片名称，值为base64编码的图片内容
            markdown_content: 包含图片引用的Markdown内容
            kb_id: 知识库ID（用于存储和构建URL）
            doc_id: 文档ID（可选，用于构建唯一临时目录）
            
        返回:
            str: 更新了图片链接的Markdown内容
        """
        # 如果没有图片，直接返回原始内容
        if not images:
            return markdown_content
            
        # 1. 创建临时目录
        temp_dir_id = doc_id or f"mineru_{int(time.time())}_{os.getpid()}"
        temp_base_dir = os.path.join(tempfile.gettempdir(), f"ragflow_{temp_dir_id}")
        temp_images_dir = os.path.join(temp_base_dir, "images")
        os.makedirs(temp_images_dir, exist_ok=True)
        
        try:
            # 2. 保存图片到临时目录
            saved_count = self._save_images(
                images=images, 
                images_dir=temp_images_dir
            )
            
            # 3. 上传图片到存储服务并更新Markdown
            if saved_count > 0:
                uploaded_count = self._upload_to_minio(
                    kb_id=kb_id, 
                    images_dir=temp_images_dir
                )
                
                # 如果成功上传，更新Markdown中的图片链接
                if uploaded_count > 0:
                    return self._update_image_links(
                        markdown_content=markdown_content, 
                        kb_id=kb_id
                    )
            
            # 如果没有保存或上传成功，返回原始内容
            return markdown_content
            
        finally:
            # 4. 清理临时目录
            try:
                import shutil
                if os.path.exists(temp_base_dir):
                    shutil.rmtree(temp_base_dir, ignore_errors=True)
                    logger.debug(f"已清理临时图片目录: {temp_base_dir}")
            except OSError as e:
                logger.warning(f"清理临时图片目录失败: {temp_base_dir}, 错误: {e}")

    def _save_images(self, images, images_dir):
        """保存图片到临时目录。"""
        saved_count = 0
        
        for image_name, image_data in images.items():
            try:
                if not image_data:
                    continue
                
                # 提取base64数据
                base64_data = image_data
                if isinstance(image_data, str) and image_data.startswith('data:image/'):
                    comma_index = image_data.find(',')
                    if comma_index != -1:
                        base64_data = image_data[comma_index + 1:]
                    else:
                        continue
                
                # 解码并保存
                try:
                    image_bytes = base64.b64decode(base64_data.strip())
                    if len(image_bytes) < 100:
                        continue
                    
                    image_path = os.path.join(images_dir, image_name)
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                        
                    saved_count += 1
                    
                except Exception as e:
                    logger.error(f"解码图片失败: {str(e)}")
                    
            except Exception as e:
                logger.error(f"处理图片失败: {str(e)}")
        
        return saved_count

    def _upload_to_minio(self, kb_id, images_dir):
        """上传图片到MinIO。"""
        try:
            from rag.utils.storage_factory import STORAGE_IMPL
            
            # 检查图片文件
            image_files = [f for f in os.listdir(images_dir) 
                          if os.path.isfile(os.path.join(images_dir, f)) 
                          and os.path.splitext(f.lower())[1] in ('.png', '.jpg', '.jpeg', '.gif', '.webp')]
            
            if not image_files:
                return 0
            
            # 上传图片
            success_count = 0
            
            for img_file in image_files:
                try:
                    img_path = os.path.join(images_dir, img_file)
                    with open(img_path, 'rb') as f:
                        img_data = f.read()
                    
                    STORAGE_IMPL.put(kb_id, img_file, img_data)
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"上传图片失败: {str(e)}")
            
            return success_count
            
        except ImportError:
            logger.error("无法导入存储模块")
            return 0

    def _update_image_links(self, markdown_content, kb_id):
        """更新Markdown中的图片链接。"""
        def _replace_img(match):
            img_path = match.group(1)
            img_name = os.path.basename(img_path)
            
            if not img_path.startswith(('http://', 'https://')):
                from rag import settings
                image_host = settings.MINIO.get("externalHost", "localhost:9380")
                img_url = f"{image_host}/v1/document/image/{kb_id}-{img_name}"
                return f'<img src="{img_url}" style="max-width: 300px;" alt="图片">'
            else:
                return f'<img src="{img_path}" style="max-width: 300px;" alt="图片">'
        
        try:
            return re.sub(r'!\[(?:图片)?\]\((.*?)\)', _replace_img, markdown_content)
        except Exception as e:
            logger.error(f"更新图片链接失败: {str(e)}")
            return markdown_content

    def _save_binary_to_temp(self, binary, temp_dir, original_filename=None):
        """将二进制内容保存到临时文件。
        
        参数:
            binary: 二进制内容
            temp_dir: 临时目录
            original_filename: 原始文件名（可选），用于提取扩展名
            
        返回:
            str: 临时文件路径
        """
        # 默认扩展名
        ext = '.bin'
        
        # 1. 首先尝试从原始文件名中获取扩展名
        if original_filename:
            file_ext = os.path.splitext(original_filename)[1].lower()
            if file_ext:
                ext = file_ext
                logger.info(f"使用原始文件名的扩展名: {ext}")
        
        # 2. 如果没有获取到扩展名，则通过二进制内容特征检测
        if ext == '.bin':
            logger.info("无法从文件名获取扩展名，尝试通过二进制内容特征检测")
            # 检测常见文件类型的魔术字节
            if binary.startswith(b'%PDF'):
                ext = '.pdf'
            elif binary.startswith(b'\xD0\xCF\x11\xE0'):  # MS Office 97-2003
                ext = '.doc'  # 可能是doc/xls/ppt等
            elif binary.startswith(b'PK\x03\x04'):  # ZIP文件，包括DOCX/XLSX/PPTX
                # 检查更具体的Office Open XML标识
                if b'word/' in binary[:4000]:
                    ext = '.docx'
                elif b'xl/' in binary[:4000]:
                    ext = '.xlsx'
                elif b'ppt/' in binary[:4000]:
                    ext = '.pptx'
                else:
                    ext = '.zip'
            logger.info(f"根据二进制内容特征检测文件类型: {ext}")
        
        # 生成临时文件名，如果有原始文件名，优先使用其基本名称
        if original_filename:
            base_name = os.path.splitext(os.path.basename(original_filename))[0]
            # 替换特殊字符，防止文件名无效
            base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
            # 限制长度
            if len(base_name) > 50:
                base_name = base_name[:50]
            temp_file_name = f"{base_name}{ext}"
        else:
            temp_file_name = f"temp_file{ext}"
            
        temp_file_path = os.path.join(temp_dir, temp_file_name)
        
        # 保存二进制内容到临时文件
        with open(temp_file_path, 'wb') as f:
            f.write(binary)
            
        logger.info(f"已保存二进制内容到临时文件: {temp_file_path}")
        return temp_file_path


# 测试代码
if __name__ == "__main__":
    import os
    import time
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 简单的回调函数，只打印消息
    def print_msg(prog=0, msg="No message"):
        print(f"[{prog*100:.1f}%] {msg}")

    # 初始化解析器并调用
    parser = MinerUParser()
    print(f"初始化解析器完成，API URL: {parser.api_url}")
    
    # 确定测试文件路径 - 使用正确的路径格式和绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, "demo.doc")
    test_file = os.path.abspath(test_file)  # 确保是绝对路径
    
    # 检查文件是否存在
    if not os.path.exists(test_file):
        print(f"错误: 测试文件不存在 - {test_file}")
        print(f"当前工作目录: {os.getcwd()}")
        # 列出当前目录内容帮助调试
        print("当前目录内容:")
        try:
            for item in os.listdir(current_dir):
                print(f"  - {item}")
        except Exception as e:
            print(f"无法列出目录内容: {str(e)}")
        import sys
        sys.exit(1)
    
    # 显示文件信息
    file_size = os.path.getsize(test_file) / 1024
    print(f"开始解析文件: {test_file} (大小: {file_size:.2f} KB)")
    start_time = time.time()
    
    # 解析文件
    try:
        sections, tables = parser(
            filename_or_binary=test_file, 
            callback=print_msg
        )
        
        # 打印解析结果摘要
        elapsed = time.time() - start_time
        print(f"\n解析完成，耗时: {elapsed:.2f} 秒，得到 {len(sections)} 个文本块")
        
        # 验证文件是否仍然存在
        if os.path.exists(test_file):
            print(f"文件完好: {test_file} 仍然存在")
        else:
            print(f"警告: 文件 {test_file} 已被删除!")
            
        if sections:
            preview_text = sections[0]['text'][:200] + "..." if len(sections[0]['text']) > 200 else sections[0]['text']
            print(f"第一个文本块内容预览:\n{preview_text}")
    except Exception as e:
        print(f"解析失败: {str(e)}")
        import traceback
        traceback.print_exc()