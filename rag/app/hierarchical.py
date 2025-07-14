#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于标题层级的文档切片方法
支持处理PDF解析器提取的文本内容以及Markdown文件
基于标题层级（H1, H2, H3）进行智能切片
优化版本：采用更先进的分块策略和AST解析
"""

import re
import os
import tempfile
from typing import List, Dict, Any, Tuple
from rag.app.utils import split_markdown_to_chunks_configured
import tiktoken
# 使用临时目录作为tiktoken缓存
tiktoken_cache_dir = tempfile.gettempdir()
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
encoder = tiktoken.get_encoding("cl100k_base")

# 标题层级正则表达式
HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

# 代码块正则表达式
CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)

# 表格正则表达式
TABLE_PATTERN = re.compile(r'\|[^\n]*\|[\s\S]*?(?=\n\n|\n#|$)', re.MULTILINE)

# 公式正则表达式
MATH_PATTERN = re.compile(r'\$\$[\s\S]*?\$\$|\$[^\$\n]+\$', re.MULTILINE)
def chunk(filename: str = None, binary=None, **kwargs) -> List[Dict[str, Any]]:
    """
    基于标题层级的文档切片入口函数
    支持处理PDF解析器提取的文本内容以及Markdown文件
    
    Args:
        filename: 文件路径（可选）
        binary: 二进制内容或文本内容（可选）
        **kwargs: 其他参数
        
    Returns:
        切片结果列表，每个元素包含 content_with_weight 等字段
    """
    
    try:
        from rag.nlp import tokenize_chunks
        import logging
        import re
        from deepdoc.parser import PdfParser as Pdf
        from deepdoc.parser import PlainParser
        
        content = None
        
        # 检查是否使用 MinerU 解析器
        layout_recognize = kwargs.get("parser_config", {}).get("layout_recognize", "DeepDOC")
        logging.info(f"hierarchical.chunk: layout_recognize={layout_recognize}")
        
        if filename and re.search(r"\.pdf$", filename, re.IGNORECASE):
            logging.info("处理 PDF 文件")
            
            if layout_recognize == "Plain Text":
                logging.info("使用 Plain Text 解析器")
                pdf_parser = PlainParser()
                # PlainParser 接受 **kwargs，但不处理 binary 参数
                if binary is not None:
                    # 如果有二进制内容，先保存到临时文件
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                        temp_file.write(binary)
                        temp_path = temp_file.name
                    try:
                        sections, tbls = pdf_parser(temp_path, from_page=kwargs.get('from_page', 0), to_page=kwargs.get('to_page', 1000), callback=kwargs.get('callback', None))
                    finally:
                        import os
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                else:
                    sections, tbls = pdf_parser(filename, from_page=kwargs.get('from_page', 0), to_page=kwargs.get('to_page', 1000), callback=kwargs.get('callback', None))
                
                # 将 sections 转换为文本内容
                content = "\n\n".join([section[0] for section in sections if section[0]])
                logging.info(f"Plain Text 解析完成，提取文本长度: {len(content)}")
                
            elif layout_recognize == "MinerU":
                # 尝试导入 MinerU 解析器
                try:
                    logging.info("尝试导入 MinerU 解析器")
                    from minerU.parser import MinerUParser
                    pdf_parser = MinerUParser()
                    logging.info("成功导入并初始化 MinerU 解析器")
                    
                    # 调用 MinerU 解析器
                    logging.info(f"调用 MinerU 解析器处理文件: {filename}")
                    try:
                        sections, tbls = pdf_parser(filename if not binary else binary, binary=binary,
                                                  from_page=kwargs.get('from_page', 0), to_page=kwargs.get('to_page', 1000), 
                                                  callback=kwargs.get('callback', None), 
                                                  kb_id=kwargs.get('kb_id'), doc_id=kwargs.get('doc_id'))
                        logging.info(f"MinerU 解析器返回结果: {len(sections)} 个文档块, {len(tbls)} 个表格")
                        
                        # 检查解析结果
                        if sections:
                            sample = sections[0]
                            logging.info(f"MinerU 解析结果示例: {sample}")
                            
                            # 将 sections 转换为文本内容
                            content = "\n\n".join([section.get('text', '') for section in sections if section.get('text')])
                            logging.info(f"MinerU 解析完成，提取文本长度: {len(content)}")
                        else:
                            logging.error("MinerU 解析器返回空结果")
                            raise Exception("MinerU 服务异常：解析器返回空结果")
                    except Exception as e:
                        # 改进错误处理
                        error_msg = str(e)
                        logging.error(f"调用 MinerU 解析器失败: {error_msg}")
                        
                        # 特殊处理RetryError
                        if "RetryError" in error_msg and "ValueError" in error_msg:
                            logging.error("检测到JSON解析错误，MinerU服务可能返回了无效响应")
                            raise Exception("MinerU 服务返回了无效的响应格式，请检查服务是否正常运行")
                        else:
                            raise Exception(f"MinerU 服务异常: {error_msg}")
                except ImportError as e:
                    logging.error(f"导入 MinerU 解析器失败: {str(e)}")
                    raise Exception(f"MinerU 解析器导入失败: {str(e)}")
            else:
                logging.info(f"使用默认解析器: {layout_recognize}")
                pdf_parser = Pdf()
                # RAGFlowPdfParser 不接受 binary 参数
                if binary is not None:
                    # 如果有二进制内容，先保存到临时文件
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                        temp_file.write(binary)
                        temp_path = temp_file.name
                    try:
                        sections = pdf_parser(temp_path, from_page=kwargs.get('from_page', 0), to_page=kwargs.get('to_page', 1000))
                    finally:
                        import os
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                else:
                    sections = pdf_parser(filename, from_page=kwargs.get('from_page', 0), to_page=kwargs.get('to_page', 1000))
                content = "\n\n".join(sections)
                logging.info(f"默认解析完成，提取文本长度: {len(content)}")
        
        # 如果没有通过 PDF 解析器获取内容，尝试其他方式
        if content is None:
            # 处理 Markdown 文件或直接提供的文本内容
            if binary is not None and isinstance(binary, str):
                content = binary
            elif binary is not None:
                content = binary.decode('utf-8', errors='ignore')
            elif filename:
                if re.search(r"\.(md|markdown)$", filename, re.IGNORECASE):
                    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                else:
                    with open(filename, 'rb') as f:
                        content = f.read().decode('utf-8', errors='ignore')
        
        if not content:
            logging.warning("未能获取文档内容")
            return []
        
        if not content or not content.strip():
            return []
        chunk_token_num = kwargs.get("parser_config", {}).get("chunk_token_num", 256)
        min_token_num = kwargs.get("parser_config", {}).get("min_token_num", 10)

        chunks = split_markdown_to_chunks_configured(
                content, 
                chunk_token_num=chunk_token_num,
                min_chunk_tokens=min_token_num,
                chunking_config={
                    "strategy": "advanced", 
                    "min_chunk_tokens": min_token_num,
                    "chunk_token_num": chunk_token_num
                }
            )
        
        # 创建文档基础信息
        doc = {
            "docnm_kwd": filename,
            "title_tks": [],
            "title_sm_tks": []
        }
        
        # 检查是否为英文
        is_english = kwargs.get('lang', 'Chinese').lower() == 'english'
        
        # 检查是否有PDF解析器（用于位置信息提取）
        pdf_parser = kwargs.get('pdf_parser', None)
        
        # 使用统一的tokenize_chunks函数处理位置信息
        if pdf_parser:
            # 如果有PDF解析器，使用它来提取位置信息
            result = tokenize_chunks(chunks, doc, is_english, pdf_parser)
        else:
            # 没有PDF解析器时，使用简单的位置信息
            result = tokenize_chunks(chunks, doc, is_english, None)
        
        return result
        
    except Exception as e:
        logging.error(f"基于标题层级的文档切片失败: {str(e)}")
        raise Exception(f"基于标题层级的文档切片失败: {str(e)}")