#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于正则表达式的文档分块方法
使用用户自定义正则表达式对文档内容进行精确分割
特别适用于法规、条款等具有固定格式的文档
"""

import copy
import os
import re
import logging
import tempfile
from typing import List, Dict, Any, Optional
from rag.app.utils import batch_add_chunk, batch_get_bbox_for_chunk_middle, get_bbox_for_chunk_middle, split_markdown_to_chunks_configured, split_markdown_to_chunks_strict_regex, split_markdown_to_chunks_smart
import tiktoken

from rag.nlp import add_positions, rag_tokenizer, tokenize
# 使用临时目录作为tiktoken缓存
tiktoken_cache_dir = tempfile.gettempdir()
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
encoder = tiktoken.get_encoding("cl100k_base")

def chunk(filename: str = None, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs) -> List[Dict[str, Any]]:
    """
    基于正则表达式的文档分块入口函数
    
    Args:
        filename: 文件路径（可选）
        binary: 二进制内容或文本内容（可选）
        **kwargs: 其他参数
            - parser_config.regex_pattern: 用户自定义正则表达式
            - parser_config.chunk_token_num: 分块大小（token数）
            - parser_config.min_chunk_token_num: 最小分块大小（token数）
            
    Returns:
        切片结果列表，每个元素包含 content_with_weight 等字段
    """
    try:
        from rag.nlp import tokenize_chunks
        import logging
        import re
        
        # 获取分块配置
        parser_config = kwargs.get("parser_config", {})
        regex_pattern = parser_config.get("regex_pattern", "第[零一二三四五六七八九十百千万\\d]+条")
        chunk_token_num = parser_config.get("chunk_token_num", 256)
        min_token_num = parser_config.get("min_chunk_token_num", 10)
        
        content = None
        middle_json = None

        layout_recognize = parser_config.get("layout_recognize", "PlainText")
        logging.info(f"strict_regex.chunk: layout_recognize={layout_recognize}, regex_pattern={regex_pattern}")
        doc_id = kwargs.get('doc_id')
        kb_id = kwargs.get('kb_id')
        name = kwargs.get('name')

        # 检查是否使用 MinerU 解析器
        if layout_recognize == "MinerU":
            logging.info(f"使用 MinerU 解析器处理文件: {filename}")
            try:
                logging.info("尝试导入 MinerU 解析器")
                from minerU.parser import MinerUParser
                pdf_parser = MinerUParser()
                logging.info("成功导入并初始化 MinerU 解析器")
                
                # 调用 MinerU 解析器
                logging.info(f"调用 MinerU 解析器处理文件: {filename}")
                try:
                    sections, tbls = pdf_parser(filename if not binary else binary, binary=binary,
                                              from_page=from_page, to_page=to_page, 
                                              lang=lang,
                                              callback=callback, 
                                              kb_id=kb_id, doc_id=doc_id)
                    logging.info(f"MinerU 解析器返回结果: {len(sections)} 个文档块, {len(tbls)} 个表格")
                    
                    # 检查解析结果
                    if sections:
                        # 将 sections 转换为文本内容
                        content = "\n\n".join([section.get('text', '') for section in sections if section.get('text')])
                        logging.info(f"MinerU 解析完成，提取文本长度: {len(content)}")
                        middle_json = sections[0].get('middle_json', None)
                    else:
                        logging.error("MinerU 解析器返回空结果")
                        raise Exception("MinerU 服务异常：解析器返回空结果")
                except Exception as e:
                    # 改进错误处理
                    error_msg = str(e)
                    logging.error(f"调用 MinerU 解析器失败: {error_msg}")
                    raise Exception(f"MinerU 服务异常: {error_msg}")
            except ImportError as e:
                logging.error(f"导入 MinerU 解析器失败: {str(e)}")
                raise Exception(f"MinerU 解析器导入失败: {str(e)}")
        
        if not content:
            logging.warning("未能获取文档内容")
            return []
        
        if not content or not content.strip():
            return []
            
        # 使用正则表达式分块
        logging.info(f"使用正则表达式分块，模式: {regex_pattern}")
        chunks = split_markdown_to_chunks_configured(
                content, 
                chunk_token_num=chunk_token_num,
                min_chunk_tokens=min_token_num,
                chunking_config={
                    "strategy": "strict_regex", 
                    "min_chunk_tokens": min_token_num,
                    "chunk_token_num": chunk_token_num
                }
            )
        callback(prog=0.85, msg="分块完成，开始提取位置信息")

        batch_chunks = batch_get_bbox_for_chunk_middle(middle_json, chunks)

        callback(prog=0.95, msg="位置信息提取完成")

        # 检查是否为英文
        is_english = lang.lower() == 'english'
        
        result = batch_add_chunk(batch_chunks, doc_id, kb_id, name, is_english)

        return result
        
    except Exception as e:
        logging.error(f"基于正则表达式的文档分块失败: {str(e)}")
        raise Exception(f"基于正则表达式的文档分块失败: {str(e)}") 
