import { DocumentParserType, LayoutRecognizeType } from '@/constants/knowledge';
import { useTranslate } from '@/hooks/common-hooks';
import { useHandleChunkMethodSelectChange } from '@/hooks/logic-hooks';
import { Form, Select } from 'antd';
import { memo, useEffect, useMemo } from 'react';
import {
  useHasParsedDocument,
  useSelectChunkMethodList,
  useSelectEmbeddingModelOptions,
} from '../hooks';

export const EmbeddingModelItem = memo(function EmbeddingModelItem() {
  const { t } = useTranslate('knowledgeConfiguration');
  const embeddingModelOptions = useSelectEmbeddingModelOptions();
  const disabled = useHasParsedDocument();

  return (
    <Form.Item
      name="embd_id"
      label={t('embeddingModel')}
      rules={[{ required: true }]}
      tooltip={t('embeddingModelTip')}
    >
      <Select
        placeholder={t('embeddingModelPlaceholder')}
        options={embeddingModelOptions}
        disabled={disabled}
      ></Select>
    </Form.Item>
  );
});

export const ChunkMethodItem = memo(function ChunkMethodItem() {
  const { t } = useTranslate('knowledgeConfiguration');
  const form = Form.useFormInstance();
  const handleChunkMethodSelectChange = useHandleChunkMethodSelectChange(form);
  const allParserList = useSelectChunkMethodList();
  const layoutRecognize = Form.useWatch(
    ['parser_config', 'layout_recognize'],
    form,
  );

  // 根据layout_recognize过滤选项
  const parserList = useMemo(() => {
    if (layoutRecognize === LayoutRecognizeType.MinerU) {
      // 只保留hierarchical选项
      return allParserList.filter(
        (option) => option.value === DocumentParserType.Hierarchical,
      );
    } else {
      return allParserList.filter(
        (option) => option.value !== DocumentParserType.Hierarchical,
      );
    }
  }, [allParserList, layoutRecognize]);

  // 监听layoutRecognize变化，自动设置默认切片方法
  useEffect(() => {
    // 只有在layoutRecognize有值时才执行逻辑
    if (!layoutRecognize) return;

    // 获取当前值并记录，防止重复设置
    const currentParserId = form.getFieldValue('parser_id');

    // 非MinerU时执行切换
    if (layoutRecognize !== LayoutRecognizeType.MinerU) {
      // 如果当前是hierarchical，需要切换
      if (currentParserId === DocumentParserType.Hierarchical) {
        // 直接从allParserList筛选，避免使用parserList导致的依赖循环
        const nonHierarchicalOptions = allParserList.filter(
          (option) => option.value !== DocumentParserType.Hierarchical,
        );

        // 如果有可用选项则设置第一个
        if (nonHierarchicalOptions.length > 0) {
          // 使用setTimeout避免React状态更新冲突
          setTimeout(() => {
            form.setFieldValue('parser_id', nonHierarchicalOptions[0].value);
          }, 0);
        }
      }
    } else if (layoutRecognize === LayoutRecognizeType.MinerU) {
      // MinerU模式下，强制设置为hierarchical
      if (currentParserId !== DocumentParserType.Hierarchical) {
        setTimeout(() => {
          form.setFieldValue('parser_id', DocumentParserType.Hierarchical);
        }, 0);
      }
    }
  }, [layoutRecognize, form, allParserList]); // 移除parserList依赖

  return (
    <Form.Item
      name="parser_id"
      label={t('chunkMethod')}
      tooltip={t('chunkMethodTip')}
      rules={[{ required: true }]}
    >
      <Select
        placeholder={t('chunkMethodPlaceholder')}
        onChange={handleChunkMethodSelectChange}
        options={parserList}
      ></Select>
    </Form.Item>
  );
});
