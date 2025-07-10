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
from typing import List, Dict, Any, Tuple, Optional
from rag.nlp import naive_merge
from rag.nlp import num_tokens_from_string as nlp_num_tokens

try:
    from markdown_it import MarkdownIt
    from markdown_it.tree import SyntaxTreeNode
    MARKDOWN_IT_AVAILABLE = True
except ImportError:
    MARKDOWN_IT_AVAILABLE = False
    print("Warning: markdown-it-py not available. Using fallback implementation.")

try:
    import tiktoken
    # 使用临时目录作为tiktoken缓存
    tiktoken_cache_dir = tempfile.gettempdir()
    os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
    encoder = tiktoken.get_encoding("cl100k_base")
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Using simple token estimation.")

# 标题层级正则表达式
HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

# 代码块正则表达式
CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)

# 表格正则表达式
TABLE_PATTERN = re.compile(r'\|[^\n]*\|[\s\S]*?(?=\n\n|\n#|$)', re.MULTILINE)

# 公式正则表达式
MATH_PATTERN = re.compile(r'\$\$[\s\S]*?\$\$|\$[^\$\n]+\$', re.MULTILINE)

def num_tokens_from_string(string: str) -> int:
    """计算文本的token数量"""
    if TIKTOKEN_AVAILABLE:
        try:
            return len(encoder.encode(string))
        except Exception:
            pass
    
    # 回退到rag.nlp模块的实现
    try:
        return nlp_num_tokens(string)
    except Exception:
        # 最后的简单估算
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', string))
        english_words = len(re.findall(r'\b\w+\b', string))
        return chinese_chars + english_words

def _extract_text_from_node(node):
    """从AST节点提取纯文本内容"""
    if hasattr(node, 'content') and node.content:
        return node.content
    
    if not hasattr(node, 'children') or not node.children:
        return ""
    
    text_parts = []
    for child in node.children:
        if hasattr(child, 'content') and child.content:
            text_parts.append(child.content)
        elif child.type == "text":
            text_parts.append(child.content or "")
        elif child.type == "code_inline":
            text_parts.append(f"`{child.content or ''}`")
        elif child.type == "strong":
            text_parts.append(f"**{_extract_text_from_node(child)}**")
        elif child.type == "em":
            text_parts.append(f"*{_extract_text_from_node(child)}*")
        elif child.type == "link":
            link_text = _extract_text_from_node(child)
            text_parts.append(f"[{link_text}]({child.attrGet('href') or ''})")
        else:
            text_parts.append(_extract_text_from_node(child))
    
    return "".join(text_parts)

def extract_markdown_structure(content: str) -> List[Dict[str, Any]]:
    """
    提取 Markdown 文档的结构信息
    
    Args:
        content: Markdown 文档内容
        
    Returns:
        包含标题信息的列表，每个元素包含 level, title, start_pos, end_pos
    """
    headers = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = sum(len(l) + 1 for l in lines[:i])  # +1 for \n
            headers.append({
                'level': level,
                'title': title,
                'start_pos': start_pos,
                'line_num': i
            })
    
    # 计算每个标题的结束位置
    for i, header in enumerate(headers):
        if i + 1 < len(headers):
            next_start = sum(len(l) + 1 for l in lines[:headers[i + 1]['line_num']])
            header['end_pos'] = next_start
        else:
            header['end_pos'] = len(content)
    
    return headers

def protect_special_content(content: str) -> Tuple[str, Dict[str, str]]:
    """
    保护特殊内容（代码块、表格、公式）不被分割
    
    Args:
        content: 原始内容
        
    Returns:
        (处理后的内容, 占位符映射)
    """
    placeholders = {}
    protected_content = content
    counter = 0
    
    # 保护代码块
    for match in CODE_BLOCK_PATTERN.finditer(content):
        placeholder = f"__CODE_BLOCK_{counter}__"
        placeholders[placeholder] = match.group(0)
        protected_content = protected_content.replace(match.group(0), placeholder, 1)
        counter += 1
    
    # 保护表格
    for match in TABLE_PATTERN.finditer(protected_content):
        placeholder = f"__TABLE_{counter}__"
        placeholders[placeholder] = match.group(0)
        protected_content = protected_content.replace(match.group(0), placeholder, 1)
        counter += 1
    
    # 保护公式
    for match in MATH_PATTERN.finditer(protected_content):
        placeholder = f"__MATH_{counter}__"
        placeholders[placeholder] = match.group(0)
        protected_content = protected_content.replace(match.group(0), placeholder, 1)
        counter += 1
    
    return protected_content, placeholders

def restore_special_content(content: str, placeholders: Dict[str, str]) -> str:
    """
    恢复特殊内容
    
    Args:
        content: 包含占位符的内容
        placeholders: 占位符映射
        
    Returns:
        恢复后的内容
    """
    restored_content = content
    for placeholder, original in placeholders.items():
        restored_content = restored_content.replace(placeholder, original)
    return restored_content

def split_large_chunk(content: str, max_tokens: int = 800) -> List[str]:
    """
    智能分割超大分块
    
    Args:
        content: 要分割的内容
        max_tokens: 最大token数
        
    Returns:
        分割后的内容列表
    """
    # 粗略估算token数（中文按字符数，英文按单词数）
    def estimate_tokens(text: str) -> int:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b\w+\b', text))
        return chinese_chars + english_words
    
    if estimate_tokens(content) <= max_tokens:
        return [content]
    
    # 按段落分割
    paragraphs = re.split(r'\n\s*\n', content)
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
        
        if estimate_tokens(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # 如果单个段落就超过限制，按句子分割
            if estimate_tokens(paragraph) > max_tokens:
                sentences = re.split(r'[。！？.!?]\s*', paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                        
                    test_sentence = temp_chunk + sentence + "。" if temp_chunk else sentence + "。"
                    
                    if estimate_tokens(test_sentence) <= max_tokens:
                        temp_chunk = test_sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence + "。"
                
                if temp_chunk:
                    current_chunk = temp_chunk.strip()
                else:
                    current_chunk = ""
            else:
                current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def merge_small_chunks(chunks: List[str], min_tokens: int = 50, target_tokens: int = 400) -> List[str]:
    """
    智能合并超小分块
    
    Args:
        chunks: 分块列表
        min_tokens: 最小token数
        target_tokens: 目标token数
        
    Returns:
        合并后的分块列表
    """
    def estimate_tokens(text: str) -> int:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b\w+\b', text))
        return chinese_chars + english_words
    
    if not chunks:
        return []
    
    merged_chunks = []
    current_chunk = ""
    
    for chunk in chunks:
        if not chunk.strip():
            continue
            
        test_chunk = current_chunk + "\n\n" + chunk if current_chunk else chunk
        
        if estimate_tokens(test_chunk) <= target_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                merged_chunks.append(current_chunk.strip())
            current_chunk = chunk
    
    if current_chunk:
        merged_chunks.append(current_chunk.strip())
    
    return merged_chunks

def add_context_enhancement(chunks: List[str], headers: List[Dict[str, Any]], 
                          content: str, enable_context: bool = True) -> List[str]:
    """
    为小分块添加上下文增强信息
    
    Args:
        chunks: 分块列表
        headers: 标题信息
        content: 原始内容
        enable_context: 是否启用上下文增强
        
    Returns:
        增强后的分块列表
    """
    if not enable_context or not headers:
        return chunks
    
    def estimate_tokens(text: str) -> int:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b\w+\b', text))
        return chinese_chars + english_words
    
    enhanced_chunks = []
    
    for chunk in chunks:
        if estimate_tokens(chunk) < 100:  # 只为小分块添加上下文
            # 找到分块在原文中的位置
            chunk_pos = content.find(chunk)
            if chunk_pos != -1:
                # 找到相关的标题
                relevant_headers = []
                for header in headers:
                    if header['start_pos'] <= chunk_pos < header['end_pos']:
                        relevant_headers.append(header)
                
                if relevant_headers:
                    # 添加标题上下文
                    context_info = " | ".join([f"H{h['level']}: {h['title']}" for h in relevant_headers[-2:]])
                    enhanced_chunk = f"[上下文: {context_info}]\n\n{chunk}"
                    enhanced_chunks.append(enhanced_chunk)
                else:
                    enhanced_chunks.append(chunk)
            else:
                enhanced_chunks.append(chunk)
        else:
            enhanced_chunks.append(chunk)
    
    return enhanced_chunks

def hierarchical_text_chunk(content: str, chunk_token_num: int = 400, 
                           max_token_num: int = 800, min_token_num: int = 50,
                           enable_context: bool = True) -> List[str]:
    """
    基于标题层级的文档切片（优化版本）
    
    Args:
        content: 文档内容（支持Markdown格式或包含标题结构的文本）
        chunk_token_num: 目标token数
        max_token_num: 最大token数
        min_token_num: 最小token数
        enable_context: 是否启用上下文增强
        
    Returns:
        切片列表
    """
    if not content.strip():
        return []
    
    # 如果有markdown-it可用，使用高级分块
    if MARKDOWN_IT_AVAILABLE:
        return _split_markdown_advanced(
            content, 
            chunk_token_num=chunk_token_num,
            max_token_num=max_token_num, 
            min_token_num=min_token_num,
            enable_context=enable_context
        )
    else:
        # 回退到原始实现
        return _split_markdown_fallback(
            content,
            chunk_token_num=chunk_token_num,
            max_token_num=max_token_num,
            min_token_num=min_token_num,
            enable_context=enable_context
        )

def _split_markdown_advanced(content: str, chunk_token_num: int = 400,
                           max_token_num: int = 800, min_token_num: int = 50,
                           enable_context: bool = True) -> List[str]:
    """
    基于AST的高级Markdown分块实现
    """
    # 动态阈值配置
    target_min_tokens = max(50, min_token_num)
    target_tokens = min(600, chunk_token_num)
    target_max_tokens = min(800, max_token_num)
    
    # 配置要作为分块边界的标题级别
    headers_to_split_on = [1, 2, 3]  # H1, H2, H3 作为分块边界
    
    # 初始化 markdown-it 解析器
    md = MarkdownIt("commonmark", {"breaks": True, "html": True})
    md.enable(['table'])
    
    try:
        # 解析为 AST
        tokens = md.parse(content)
        tree = SyntaxTreeNode(tokens)
        
        # 提取所有节点和标题信息
        nodes_with_headers = _extract_nodes_with_header_info(tree, headers_to_split_on)
        
        # 基于标题层级进行初步分块
        initial_chunks = _split_by_header_levels(nodes_with_headers, headers_to_split_on)
        
        # 应用动态大小控制和优化
        optimized_chunks = _apply_size_control_and_optimization(
            initial_chunks, target_min_tokens, target_tokens, target_max_tokens
        )
        
        # 生成最终分块内容
        final_chunks = []
        for chunk_info in optimized_chunks:
            content_text = _render_header_chunk_advanced(chunk_info)
            if content_text.strip():
                final_chunks.append(content_text)
        
        return final_chunks
    
    except Exception as e:
        print(f"Advanced header-based parsing failed: {e}, falling back to simple chunking")
        return _split_markdown_fallback(content, chunk_token_num, max_token_num, min_token_num, enable_context)

def _split_markdown_fallback(content: str, chunk_token_num: int = 400,
                           max_token_num: int = 800, min_token_num: int = 50,
                           enable_context: bool = True) -> List[str]:
    """
    回退的简单分块实现
    """
    # 保护特殊内容
    protected_content, placeholders = protect_special_content(content)
    
    # 提取标题结构
    headers = extract_markdown_structure(protected_content)
    
    if not headers:
        # 没有标题，使用段落分割
        paragraphs = re.split(r'\n\s*\n', protected_content)
        chunks = [p.strip() for p in paragraphs if p.strip()]
    else:
        # 基于标题层级分割
        chunks = []
        
        # 处理文档开头（第一个标题之前的内容）
        if headers[0]['start_pos'] > 0:
            intro_content = protected_content[:headers[0]['start_pos']].strip()
            if intro_content:
                chunks.append(intro_content)
        
        # 处理每个标题段落
        for i, header in enumerate(headers):
            section_content = protected_content[header['start_pos']:header['end_pos']].strip()
            if section_content:
                chunks.append(section_content)
    
    # 分割超大分块
    split_chunks = []
    for chunk in chunks:
        split_chunks.extend(split_large_chunk(chunk, max_token_num))
    
    # 合并超小分块
    merged_chunks = merge_small_chunks(split_chunks, min_token_num, chunk_token_num)
    
    # 添加上下文增强
    enhanced_chunks = add_context_enhancement(merged_chunks, headers, protected_content, enable_context)
    
    # 恢复特殊内容
    final_chunks = []
    for chunk in enhanced_chunks:
        restored_chunk = restore_special_content(chunk, placeholders)
        if restored_chunk.strip():
            final_chunks.append(restored_chunk.strip())
    
    return final_chunks

def _extract_nodes_with_header_info(tree, headers_to_split_on):
    """提取所有节点及其对应的标题信息"""
    nodes_with_headers = []
    current_headers = {}  # 当前的标题层级路径
    
    for node in tree.children:
        if node.type == "heading":
            level = int(node.tag[1])  # h1 -> 1, h2 -> 2, etc.
            title = _extract_text_from_node(node)
            
            # 更新当前标题路径
            # 移除比当前级别更深的标题
            current_headers = {k: v for k, v in current_headers.items() if k < level}
            # 添加当前标题
            current_headers[level] = title
            
            # 如果是分块边界标题，标记为分块起始点
            is_split_boundary = level in headers_to_split_on
            
            nodes_with_headers.append({
                'node': node,
                'type': 'heading',
                'level': level,
                'title': title,
                'headers': current_headers.copy(),
                'is_split_boundary': is_split_boundary,
                'content': node.markup + " " + title
            })
        else:
            # 非标题节点
            content = _render_node_content(node)
            if content.strip():
                nodes_with_headers.append({
                    'node': node,
                    'type': node.type,
                    'headers': current_headers.copy(),
                    'is_split_boundary': False,
                    'content': content
                })
    
    return nodes_with_headers

def _render_node_content(node):
    """渲染单个节点的内容"""
    if node.type == "table":
        return _render_table_from_ast(node)
    elif node.type == "code_block":
        return f"```{node.info or ''}\n{node.content}```"
    elif node.type == "blockquote":
        return _render_blockquote_from_ast(node)
    elif node.type in ["bullet_list", "ordered_list"]:
        return _render_list_from_ast(node)
    elif node.type == "paragraph":
        return _extract_text_from_node(node)
    elif node.type == "hr":
        return "---"
    else:
        return _extract_text_from_node(node)

def _render_table_from_ast(table_node):
    """从 AST 渲染表格"""
    try:
        # 构建表格的 markdown 表示
        table_md = []
        
        for child in table_node.children:
            if child.type == "thead":
                # 表头处理
                for row in child.children:
                    if row.type == "tr":
                        cells = []
                        for cell in row.children:
                            if cell.type in ["th", "td"]:
                                cells.append(_extract_text_from_node(cell))
                        table_md.append("| " + " | ".join(cells) + " |")
                
                # 添加分隔符
                if table_md:
                    separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                    table_md.append(separator)
                    
            elif child.type == "tbody":
                # 表体处理
                for row in child.children:
                    if row.type == "tr":
                        cells = []
                        for cell in row.children:
                            if cell.type in ["th", "td"]:
                                cells.append(_extract_text_from_node(cell))
                        table_md.append("| " + " | ".join(cells) + " |")
        
        # 返回 markdown 表格
        return "\n".join(table_md)
        
    except Exception as e:
        print(f"Table rendering error: {e}")
        return _extract_text_from_node(table_node)

def _render_list_from_ast(list_node):
    """从 AST 渲染列表"""
    list_items = []
    list_type = list_node.attrGet('type') or 'bullet'
    
    for i, item in enumerate(list_node.children):
        if item.type == "list_item":
            item_content = _extract_text_from_node(item)
            if list_type == 'ordered':
                list_items.append(f"{i+1}. {item_content}")
            else:
                list_items.append(f"- {item_content}")
    
    return "\n".join(list_items)

def _render_blockquote_from_ast(blockquote_node):
    """从 AST 渲染引用块"""
    content = _extract_text_from_node(blockquote_node)
    lines = content.split('\n')
    return '\n'.join(f"> {line}" for line in lines)

def _split_by_header_levels(nodes_with_headers, headers_to_split_on):
    """基于标题层级进行分块，智能处理连续标题"""
    chunks = []
    current_chunk = {
        'headers': {},
        'nodes': []
    }
    
    i = 0
    while i < len(nodes_with_headers):
        node_info = nodes_with_headers[i]
        
        # 检查是否为分块边界标题
        if node_info['is_split_boundary']:
            # 完成当前块（如果有内容）
            if (current_chunk['nodes'] and 
                any(n for n in current_chunk['nodes'] if n['content'].strip())):
                chunks.append(current_chunk)
                current_chunk = {
                    'headers': {},
                    'nodes': []
                }
        
        # 更新当前块的标题信息
        if node_info['headers']:
            current_chunk['headers'] = node_info['headers'].copy()
        
        # 添加节点到当前块
        current_chunk['nodes'].append(node_info)
        i += 1
    
    # 添加最后一个块
    if current_chunk['nodes'] and any(n for n in current_chunk['nodes'] if n['content'].strip()):
        chunks.append(current_chunk)
    
    return chunks

def _apply_size_control_and_optimization(chunks, min_tokens, target_tokens, max_tokens):
    """应用动态大小控制和优化策略"""
    optimized_chunks = []
    
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        chunk_content = _render_header_chunk_advanced(chunk)
        chunk_tokens = num_tokens_from_string(chunk_content)
        
        if chunk_tokens <= max_tokens and chunk_tokens >= min_tokens:
            # 大小合适，直接添加
            chunk['chunk_type'] = 'normal'
            optimized_chunks.append(chunk)
            
        elif chunk_tokens > max_tokens:
            # 超大分块，需要进一步分割
            split_chunks = _split_oversized_chunk(chunk, target_tokens, max_tokens)
            optimized_chunks.extend(split_chunks)
            
        elif chunk_tokens < min_tokens:
            # 超小分块，尝试与下一个分块合并
            merged_chunk = _try_merge_with_next(chunk, chunks, i, target_tokens)
            if merged_chunk:
                optimized_chunks.append(merged_chunk)
                # 跳过被合并的分块
                i += merged_chunk.get('merged_count', 1) - 1
            else:
                # 无法合并，添加上下文增强
                enhanced_chunk = _enhance_small_chunk_with_context(chunk)
                optimized_chunks.append(enhanced_chunk)
        else:
            # 其他情况，直接添加
            optimized_chunks.append(chunk)
        
        i += 1
    
    return optimized_chunks

def _render_header_chunk_advanced(chunk_info):
    """渲染基于标题的分块内容（高级版本）"""
    content_parts = []
    
    # 添加标题上下文（如果分块本身不包含标题）
    chunk_has_header = any(node['type'] == 'heading' for node in chunk_info.get('nodes', []))
    
    if not chunk_has_header and chunk_info.get('headers'):
        # 添加最相关的上下文标题
        context_header = _get_most_relevant_header_advanced(chunk_info['headers'])
        if context_header:
            content_parts.append(context_header)
    
    # 渲染所有节点内容
    for node_info in chunk_info.get('nodes', []):
        if node_info.get('content', '').strip():
            content_parts.append(node_info['content'])
    
    return "\n\n".join(content_parts).strip()

def _get_most_relevant_header_advanced(headers):
    """获取最相关的上下文标题（高级版本）"""
    if not headers:
        return None
    
    # 选择最深层级的标题作为上下文
    max_level = max(headers.keys())
    return f"{'#' * max_level} {headers[max_level]}"

def _split_oversized_chunk(chunk, target_tokens, max_tokens):
    """分割超大分块"""
    # 简单实现：按段落分割
    content = _render_header_chunk_advanced(chunk)
    paragraphs = re.split(r'\n\s*\n', content)
    
    split_chunks = []
    current_content = ""
    
    for paragraph in paragraphs:
        test_content = current_content + "\n\n" + paragraph if current_content else paragraph
        if num_tokens_from_string(test_content) <= target_tokens:
            current_content = test_content
        else:
            if current_content:
                split_chunks.append({
                    'headers': chunk.get('headers', {}),
                    'nodes': [{'content': current_content, 'type': 'text'}],
                    'chunk_type': 'split_oversized'
                })
            current_content = paragraph
    
    if current_content:
        split_chunks.append({
            'headers': chunk.get('headers', {}),
            'nodes': [{'content': current_content, 'type': 'text'}],
            'chunk_type': 'split_oversized'
        })
    
    return split_chunks if split_chunks else [chunk]

def _try_merge_with_next(chunk, chunks, current_index, target_tokens):
    """尝试与下一个分块合并"""
    if current_index + 1 >= len(chunks):
        return None
    
    next_chunk = chunks[current_index + 1]
    current_content = _render_header_chunk_advanced(chunk)
    next_content = _render_header_chunk_advanced(next_chunk)
    
    merged_content = current_content + "\n\n" + next_content
    
    if num_tokens_from_string(merged_content) <= target_tokens:
        # 可以合并
        merged_nodes = chunk.get('nodes', []) + next_chunk.get('nodes', [])
        return {
            'headers': chunk.get('headers', {}),
            'nodes': merged_nodes,
            'chunk_type': 'merged',
            'merged_count': 2
        }
    
    return None

def _enhance_small_chunk_with_context(chunk):
    """为小分块添加上下文增强"""
    chunk['chunk_type'] = 'small_enhanced'
    return chunk

def chunk(filename: str = None, binary=None, chunk_token_num: int = 400, max_token_num: int = 800, 
         min_token_num: int = 50, enable_context: bool = True, **kwargs) -> List[Dict[str, Any]]:
    """
    基于标题层级的文档切片入口函数
    支持处理PDF解析器提取的文本内容以及Markdown文件
    
    Args:
        filename: 文件路径（可选）
        binary: 二进制内容或文本内容（可选）
        chunk_token_num: 目标token数
        max_token_num: 最大token数
        min_token_num: 最小token数
        enable_context: 是否启用上下文增强
        **kwargs: 其他参数
        
    Returns:
        切片结果列表，每个元素包含 content_with_weight 等字段
    """
    from rag.nlp import tokenize_chunks
    import copy
    
    try:
        content = None
        
        # 获取文档内容
        if binary is not None:
            # 处理二进制内容或直接的文本内容
            if isinstance(binary, bytes):
                content = binary.decode('utf-8', errors='ignore')
            else:
                content = str(binary)
        elif filename and os.path.exists(filename):
            # 从文件读取内容
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            raise ValueError("必须提供 filename 或 binary 参数")
        
        if not content or not content.strip():
            return []
        
        # 执行切片
        chunks = hierarchical_text_chunk(
            content, 
            chunk_token_num=chunk_token_num,
            max_token_num=max_token_num, 
            min_token_num=min_token_num,
            enable_context=enable_context
        )
        
        # 创建文档基础信息
        doc = {
            "docnm_kwd": filename or "hierarchical_document",
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
        raise Exception(f"基于标题层级的文档切片失败: {str(e)}")