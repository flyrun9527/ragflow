#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于标题层级的文档切片方法
支持处理PDF解析器提取的文本内容以及Markdown文件
基于标题层级（H1, H2, H3）进行智能切片
优化版本：采用更先进的分块策略和AST解析
"""

import copy
import logging
import re
import os
import tempfile
from typing import List, Dict, Any, Tuple
from minerU.parser.mineru_parser import MinerUParser
from rag.app.utils import get_bbox_for_chunk, get_bbox_for_chunk_middle, split_markdown_to_chunks_configured
import tiktoken

from rag.nlp import add_positions, rag_tokenizer
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
def chunk(filename: str = None, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs) -> List[Dict[str, Any]]:
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
        
        content = None
        middle_json = None
        
        # 检查是否使用 MinerU 解析器
        layout_recognize = kwargs.get("parser_config", {}).get("layout_recognize", "MinerU")
        logging.info(f"hierarchical.chunk: layout_recognize={layout_recognize}")
        
        # 优先检查是否使用 MinerU 解析器，不受文件类型限制
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
                                              kb_id=kwargs.get('kb_id'), doc_id=kwargs.get('doc_id'))
                    logging.info(f"MinerU 解析器返回结果: {len(sections)} 个文档块, {len(tbls)} 个表格")
                    
                    # 检查解析结果
                    if sections:
                        sample = sections[0]
                        logging.info(f"MinerU 解析结果示例: {sample}")
                        
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
                    
                    # 特殊处理RetryError
                    if "RetryError" in error_msg and "ValueError" in error_msg:
                        logging.error("检测到JSON解析错误，MinerU服务可能返回了无效响应")
                        raise Exception("MinerU 服务返回了无效的响应格式，请检查服务是否正常运行")
                    else:
                        raise Exception(f"MinerU 服务异常: {error_msg}")
            except ImportError as e:
                logging.error(f"导入 MinerU 解析器失败: {str(e)}")
                raise Exception(f"MinerU 解析器导入失败: {str(e)}")
        
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
        callback(prog=0.85, msg="分块完成，开始提取位置信息")

        # 准备批量数据，包含位置信息
        batch_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk and chunk.strip():
                logging.info(f"chunk: {chunk}")
                chunk_data = {
                    "content": chunk.strip(),
                    "important_keywords": [],  # 可以根据需要添加关键词提取
                    "questions": []  # 可以根据需要添加问题生成
                }
                position_int_temp = get_bbox_for_chunk_middle(middle_json=middle_json, chunk_content=chunk.strip())
                
                # 处理位置信息
                if position_int_temp is not None:
                    # 有完整位置信息，使用positions参数
                    chunk_data["positions"] = position_int_temp
                else:
                    # 没有完整位置信息，使用top_int参数
                    chunk_data["top_int"] = i
                
                # 将处理好的chunk添加到结果列表中
                batch_chunks.append(chunk_data)
        
        callback(prog=0.95, msg="位置信息提取完成")

        # 创建文档基础信息
        doc = {
            "docnm_kwd": filename,
            "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
        }
        doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
        
        # 检查是否为英文
        is_english = lang.lower() == 'english'
        
        # 检查是否有PDF解析器（用于位置信息提取）
        result = []
        # 使用统一的tokenize_chunks函数处理位置信息
        for ii, ck in enumerate(batch_chunks):
            if len(ck["content"].strip()) == 0:
                continue
            logging.debug("-- {}".format(ck["content"]))
            d = copy.deepcopy(doc)
            add_positions(d, ck["positions"])
            tokenize(d, ck["content"], is_english)
            result.append(d)
        return result
        
    except Exception as e:
        logging.error(f"基于标题层级的文档切片失败: {str(e)}")
        raise Exception(f"基于标题层级的文档切片失败: {str(e)}")

def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])

# uv run python -m rag.app.hierarchical
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
    test_file = os.path.join(current_dir, "demo.pdf")
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
    content = None
    middle_json = None
    # 解析文件
    try:
        sections, tables = parser(
            filename_or_binary=test_file, 
            callback=print_msg
        )
        
        # 检查解析结果
        if sections:
            sample = sections[0]
            logging.info(f"MinerU 解析结果示例: {sample}")
            
            # 将 sections 转换为文本内容
            content = "\n\n".join([section.get('text', '') for section in sections if section.get('text')])
            logging.info(f"MinerU 解析完成，提取文本长度: {len(content)}")
            middle_json = sections[0].get('middle_json', None)
            
        chunks = split_markdown_to_chunks_configured(
                content, 
                chunk_token_num=256,
                min_chunk_tokens=10,
                chunking_config={
                    "strategy": "advanced", 
                    "min_chunk_tokens": 10,
                    "chunk_token_num": 256
                }
            )
        batch_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk and chunk.strip():
                chunk_data = {
                    "content": chunk.strip(),
                    "important_keywords": [],  # 可以根据需要添加关键词提取
                    "questions": []  # 可以根据需要添加问题生成
                }
                position_int_temp = get_bbox_for_chunk_middle(middle_json=middle_json, chunk_content=chunk.strip())
                
                # 处理位置信息
                if position_int_temp is not None:
                    # 有完整位置信息，使用positions参数
                    chunk_data["positions"] = position_int_temp
                else:
                    # 没有完整位置信息，使用top_int参数
                    chunk_data["top_int"] = i
                
                # 将处理好的chunk添加到结果列表中
                batch_chunks.append(chunk_data)
    except Exception as e:
        print(f"解析失败: {str(e)}")
        import traceback
        traceback.print_exc()