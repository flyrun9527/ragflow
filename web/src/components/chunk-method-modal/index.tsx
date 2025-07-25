import MaxTokenNumber from '@/components/max-token-number';
import { IModalManagerChildrenProps } from '@/components/modal-manager';
import {
  MinusCircleOutlined,
  PlusOutlined,
  QuestionCircleOutlined,
} from '@ant-design/icons';
import {
  Button,
  Divider,
  Form,
  InputNumber,
  Modal,
  Select,
  Space,
  Tooltip,
} from 'antd';
import omit from 'lodash/omit';
import React, { useEffect, useMemo } from 'react';
import { useFetchParserListOnMount, useShowAutoKeywords } from './hooks';

import MaxMinTokenNumber from '@/components/max-min-token-number';
import { DocumentParserType, LayoutRecognizeType } from '@/constants/knowledge';
import { useTranslate } from '@/hooks/common-hooks';
import { useFetchKnowledgeBaseConfiguration } from '@/hooks/knowledge-hooks';
import { IParserConfig } from '@/interfaces/database/document';
import { IChangeParserConfigRequestBody } from '@/interfaces/request/document';
import { get } from 'lodash';
import { AutoKeywordsItem, AutoQuestionsItem } from '../auto-keywords-item';
import { DatasetConfigurationContainer } from '../dataset-configuration-container';
import Delimiter from '../delimiter';
import EntityTypesItem from '../entity-types-item';
import ExcelToHtml from '../excel-to-html';
import LayoutRecognize from '../layout-recognize';
import ParseConfiguration, {
  showRaptorParseConfiguration,
} from '../parse-configuration';
import {
  UseGraphRagItem,
  showGraphRagItems,
} from '../parse-configuration/graph-rag-items';
import styles from './index.less';

interface IProps extends Omit<IModalManagerChildrenProps, 'showModal'> {
  loading: boolean;
  onOk: (
    parserId: DocumentParserType | undefined,
    parserConfig: IChangeParserConfigRequestBody,
  ) => void;
  showModal?(): void;
  parserId: DocumentParserType;
  parserConfig: IParserConfig;
  documentExtension: string;
  documentId: string;
}

const hidePagesChunkMethods = [
  DocumentParserType.Qa,
  DocumentParserType.Table,
  DocumentParserType.Picture,
  DocumentParserType.Resume,
  DocumentParserType.One,
  DocumentParserType.KnowledgeGraph,
];

const ChunkMethodModal: React.FC<IProps> = ({
  documentId,
  parserId,
  onOk,
  hideModal,
  visible,
  documentExtension,
  parserConfig,
  loading,
}) => {
  const [form] = Form.useForm();
  const {
    parserList: allParserList,
    handleChange,
    selectedTag,
  } = useFetchParserListOnMount(documentId, parserId, documentExtension, form);
  const { t } = useTranslate('knowledgeDetails');
  const { data: knowledgeDetails } = useFetchKnowledgeBaseConfiguration();

  // 监听布局识别类型 - 只使用Form.useWatch，不使用state
  const layoutRecognize = Form.useWatch(
    ['parser_config', 'layout_recognize'],
    form,
  );

  // 根据布局识别类型过滤切片方法选项
  const filteredParserList = useMemo(() => {
    // 注意：即使 layoutRecognize 为 null/undefined，当前选择的切片方法是 Hierarchical 或 StrictRegex 时也应该显示
    const isHierarchical = selectedTag === DocumentParserType.Hierarchical;
    const isStrictRegex = selectedTag === DocumentParserType.StrictRegex;

    if (
      layoutRecognize === LayoutRecognizeType.MinerU ||
      isHierarchical ||
      isStrictRegex
    ) {
      // MinerU布局支持Hierarchical和StrictRegex切片方法
      // 查找并返回这两个选项
      const supportedOptions = allParserList.filter(
        (option) =>
          option.value === DocumentParserType.Hierarchical ||
          option.value === DocumentParserType.StrictRegex,
      );

      return supportedOptions.length > 0 ? supportedOptions : [];
    } else {
      // 其他布局不支持Hierarchical
      return allParserList.filter(
        (option) => option.value !== DocumentParserType.Hierarchical,
      );
    }
  }, [allParserList, layoutRecognize, selectedTag]);

  // 监听布局识别类型变化，自动设置对应的切片方法
  useEffect(() => {
    // 只有在对话框可见且布局识别类型有值时才执行逻辑
    if (!visible || !layoutRecognize) return;

    // 简化逻辑：布局和切片方法的一致性检查
    const isSupportedMethod =
      selectedTag === DocumentParserType.Hierarchical ||
      selectedTag === DocumentParserType.StrictRegex;

    if (layoutRecognize === LayoutRecognizeType.MinerU && !isSupportedMethod) {
      // MinerU布局默认使用Hierarchical切片方法
      setTimeout(() => handleChange(DocumentParserType.Hierarchical), 0);
    } else if (
      layoutRecognize !== LayoutRecognizeType.MinerU &&
      selectedTag === DocumentParserType.Hierarchical
    ) {
      // Hierarchical切片方法只能用于MinerU布局
      // 找到第一个非Hierarchical/StrictRegex选项
      const firstNonHierarchical = allParserList.find(
        (option) =>
          option.value !== DocumentParserType.Hierarchical &&
          option.value !== DocumentParserType.StrictRegex,
      );

      if (firstNonHierarchical) {
        setTimeout(() => handleChange(firstNonHierarchical.value), 0);
      }
    }
  }, [layoutRecognize, selectedTag, handleChange, allParserList, visible]);

  // 自定义handleTagChange函数，在切换切片方法时同步设置布局识别类型
  const handleTagChange = (value: DocumentParserType) => {
    // 先更新切片方法
    handleChange(value);

    // MinerU布局支持的切片方法
    const isSupportedMethod =
      value === DocumentParserType.Hierarchical ||
      value === DocumentParserType.StrictRegex;

    // 如果选择的是支持MinerU的方法，设置布局为MinerU
    if (isSupportedMethod) {
      setTimeout(() => {
        form.setFieldValue(
          ['parser_config', 'layout_recognize'],
          LayoutRecognizeType.MinerU,
        );
      }, 0);
    }
  };

  const useGraphRag = useMemo(() => {
    return knowledgeDetails.parser_config?.graphrag?.use_graphrag;
  }, [knowledgeDetails.parser_config?.graphrag?.use_graphrag]);

  const handleOk = async () => {
    const values = await form.validateFields();
    const parser_config = {
      ...values.parser_config,
      pages: values.pages?.map((x: any) => [x.from, x.to]) ?? [],
    };
    onOk(selectedTag, parser_config);
  };

  const isPdf = documentExtension === 'pdf';

  const showPages = useMemo(() => {
    // 简化判断逻辑
    const isPdfOrSupportedMethod =
      isPdf ||
      selectedTag === DocumentParserType.Hierarchical ||
      selectedTag === DocumentParserType.StrictRegex;

    // 不在隐藏列表中的PDF或支持的切片方法需要显示页面范围
    // 确保selectedTag有值且不在隐藏列表中
    return (
      isPdfOrSupportedMethod &&
      selectedTag &&
      !hidePagesChunkMethods.includes(selectedTag)
    );
  }, [selectedTag, isPdf]);

  // 简化为常量，不需要useMemo，始终显示布局识别器选项
  const showOne = true;

  const showMaxTokenNumber =
    selectedTag === DocumentParserType.Naive ||
    selectedTag === DocumentParserType.KnowledgeGraph;

  // 添加MaxMinTokenNumber显示逻辑 - 使用正确的枚举值
  const showMaxMinTokenNumber =
    selectedTag === DocumentParserType.Hierarchical ||
    selectedTag === DocumentParserType.StrictRegex;

  const showEntityTypes = selectedTag === DocumentParserType.KnowledgeGraph;

  const showExcelToHtml =
    selectedTag === DocumentParserType.Naive && documentExtension === 'xlsx';

  const showAutoKeywords = useShowAutoKeywords();

  const afterClose = () => {
    form.resetFields();
  };

  useEffect(() => {
    if (visible) {
      const pages =
        parserConfig?.pages?.map((x) => ({ from: x[0], to: x[1] })) ?? [];
      form.setFieldsValue({
        pages: pages.length > 0 ? pages : [{ from: 1, to: 1024 }],
        parser_config: {
          ...omit(parserConfig, 'pages'),
          graphrag: {
            use_graphrag: get(
              parserConfig,
              'graphrag.use_graphrag',
              useGraphRag,
            ),
          },
        },
      });
    }
  }, [
    form,
    knowledgeDetails.parser_config,
    parserConfig,
    useGraphRag,
    visible,
  ]);

  return (
    <Modal
      title={t('chunkMethod')}
      open={visible}
      onOk={handleOk}
      onCancel={hideModal}
      afterClose={afterClose}
      confirmLoading={loading}
      width={700}
    >
      <Space size={[0, 8]} wrap>
        <Form.Item label={t('chunkMethod')} className={styles.chunkMethod}>
          <Select
            style={{ width: 160 }}
            onChange={handleTagChange}
            value={selectedTag}
            options={filteredParserList}
          />
        </Form.Item>
      </Space>
      <Divider></Divider>
      <Form
        name="dynamic_form_nest_item"
        autoComplete="off"
        form={form}
        className="space-y-4"
      >
        {showPages && (
          <>
            <Space>
              <p>{t('pageRanges')}:</p>
              <Tooltip title={t('pageRangesTip')}>
                <QuestionCircleOutlined
                  className={styles.questionIcon}
                ></QuestionCircleOutlined>
              </Tooltip>
            </Space>
            <Form.List name="pages">
              {(fields, { add, remove }) => (
                <>
                  {fields.map(({ key, name, ...restField }) => (
                    <Space
                      key={key}
                      style={{
                        display: 'flex',
                      }}
                      align="baseline"
                    >
                      <Form.Item
                        {...restField}
                        name={[name, 'from']}
                        dependencies={name > 0 ? [name - 1, 'to'] : []}
                        rules={[
                          {
                            required: true,
                            message: t('fromMessage'),
                          },
                          ({ getFieldValue }) => ({
                            validator(_, value) {
                              if (
                                name === 0 ||
                                !value ||
                                getFieldValue(['pages', name - 1, 'to']) < value
                              ) {
                                return Promise.resolve();
                              }
                              return Promise.reject(
                                new Error(t('greaterThanPrevious')),
                              );
                            },
                          }),
                        ]}
                      >
                        <InputNumber
                          placeholder={t('fromPlaceholder')}
                          min={0}
                          precision={0}
                          className={styles.pageInputNumber}
                        />
                      </Form.Item>
                      <Form.Item
                        {...restField}
                        name={[name, 'to']}
                        dependencies={[name, 'from']}
                        rules={[
                          {
                            required: true,
                            message: t('toMessage'),
                          },
                          ({ getFieldValue }) => ({
                            validator(_, value) {
                              if (
                                !value ||
                                getFieldValue(['pages', name, 'from']) < value
                              ) {
                                return Promise.resolve();
                              }
                              return Promise.reject(
                                new Error(t('greaterThan')),
                              );
                            },
                          }),
                        ]}
                      >
                        <InputNumber
                          placeholder={t('toPlaceholder')}
                          min={0}
                          precision={0}
                          className={styles.pageInputNumber}
                        />
                      </Form.Item>
                      {name > 0 && (
                        <MinusCircleOutlined onClick={() => remove(name)} />
                      )}
                    </Space>
                  ))}
                  <Form.Item>
                    <Button
                      type="dashed"
                      onClick={() => add()}
                      block
                      icon={<PlusOutlined />}
                    >
                      {t('addPage')}
                    </Button>
                  </Form.Item>
                </>
              )}
            </Form.List>
          </>
        )}

        {showPages && (
          <Form.Item
            noStyle
            dependencies={[['parser_config', 'layout_recognize']]}
          >
            {({ getFieldValue }) =>
              getFieldValue(['parser_config', 'layout_recognize']) && (
                <Form.Item
                  name={['parser_config', 'task_page_size']}
                  label={t('taskPageSize')}
                  tooltip={t('taskPageSizeTip')}
                  initialValue={12}
                  rules={[
                    {
                      required: true,
                      message: t('taskPageSizeMessage'),
                    },
                  ]}
                >
                  <InputNumber min={1} max={128} />
                </Form.Item>
              )
            }
          </Form.Item>
        )}
        <DatasetConfigurationContainer
          show={showOne || showMaxTokenNumber || showMaxMinTokenNumber}
        >
          {showOne && <LayoutRecognize></LayoutRecognize>}
          {showMaxTokenNumber && (
            <>
              <MaxTokenNumber
                max={
                  selectedTag === DocumentParserType.KnowledgeGraph
                    ? 8192 * 2
                    : 2048
                }
              ></MaxTokenNumber>
              <Delimiter></Delimiter>
            </>
          )}
          {showMaxMinTokenNumber && (
            <MaxMinTokenNumber selectedTag={selectedTag} />
          )}
        </DatasetConfigurationContainer>
        <DatasetConfigurationContainer
          show={showAutoKeywords(selectedTag) || showExcelToHtml}
        >
          {showAutoKeywords(selectedTag) && (
            <>
              <AutoKeywordsItem></AutoKeywordsItem>
              <AutoQuestionsItem></AutoQuestionsItem>
            </>
          )}
          {showExcelToHtml && <ExcelToHtml></ExcelToHtml>}
        </DatasetConfigurationContainer>
        {showRaptorParseConfiguration(selectedTag) && (
          <DatasetConfigurationContainer>
            <ParseConfiguration></ParseConfiguration>
          </DatasetConfigurationContainer>
        )}
        {showGraphRagItems(selectedTag) && useGraphRag && (
          <UseGraphRagItem></UseGraphRagItem>
        )}
        {showEntityTypes && <EntityTypesItem></EntityTypesItem>}
      </Form>
    </Modal>
  );
};
export default ChunkMethodModal;
