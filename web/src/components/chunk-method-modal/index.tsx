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
import React, { useEffect, useMemo, useRef } from 'react';
import { useFetchParserListOnMount, useShowAutoKeywords } from './hooks';

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
import MaxMinTokenNumber from '../max-min-token-number';
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
  const { parserList, handleChange, selectedTag } = useFetchParserListOnMount(
    documentId,
    parserId,
    documentExtension,
    form,
  );
  const { t } = useTranslate('knowledgeDetails');
  const { data: knowledgeDetails } = useFetchKnowledgeBaseConfiguration();

  const layoutRecognize = Form.useWatch(
    ['parser_config', 'layout_recognize'],
    form,
  );

  // 用于跟踪表单是否已初始化完成
  const initializedRef = useRef(false);

  // 根据布局识别类型过滤切片方法选项
  const filteredParserList = useMemo(() => {
    if (layoutRecognize === LayoutRecognizeType.MinerU) {
      // MinerU布局支持Hierarchical和StrictRegex切片方法
      // 查找并返回这两个选项
      const supportedOptions = parserList.filter(
        (option) =>
          option.value === DocumentParserType.Hierarchical ||
          option.value === DocumentParserType.StrictRegex,
      );

      return supportedOptions.length > 0 ? supportedOptions : [];
    } else {
      const supportedOptions = parserList.filter(
        (option) =>
          option.value !== DocumentParserType.Hierarchical &&
          option.value !== DocumentParserType.StrictRegex,
      );

      return supportedOptions;
    }
  }, [parserList, layoutRecognize]);

  // 监听布局识别类型变化，并在变化时设置合适的切片方法
  useEffect(() => {
    // 如果表单尚未初始化完成，则不执行任何操作
    if (!initializedRef.current || !layoutRecognize) {
      return;
    }

    // 根据新的布局类型选择合适的切片方法
    if (layoutRecognize === LayoutRecognizeType.MinerU) {
      // 如果切换到MinerU，选择Hierarchical作为默认值（如果当前选择的不是支持的切片方法）
      if (
        selectedTag !== DocumentParserType.Hierarchical &&
        selectedTag !== DocumentParserType.StrictRegex
      ) {
        const hierarchicalOption = filteredParserList.find(
          (option) => option.value === DocumentParserType.Hierarchical,
        );

        if (hierarchicalOption) {
          // 使用Hierarchical作为MinerU模式的默认切片方法
          handleChange(hierarchicalOption.value);
        }
      }
    } else {
      // 如果从MinerU切换到其他类型，且当前选择的是Hierarchical或StrictRegex
      if (
        selectedTag === DocumentParserType.Hierarchical ||
        selectedTag === DocumentParserType.StrictRegex
      ) {
        // 选择过滤后列表中的第一个选项
        if (filteredParserList.length > 0) {
          handleChange(filteredParserList[0].value);
        }
      }
    }
  }, [layoutRecognize, filteredParserList, selectedTag, handleChange]);

  // 标记表单初始化完成
  useEffect(() => {
    if (visible) {
      // 延迟一帧将初始化标记设为true
      const timer = setTimeout(() => {
        initializedRef.current = true;
      }, 0);

      return () => clearTimeout(timer);
    } else {
      // 当模态框关闭时，重置初始化状态
      initializedRef.current = false;
    }
  }, [visible]);

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
    if (parserConfig?.layout_recognize === LayoutRecognizeType.MinerU) {
      return true;
    }
    return isPdf && hidePagesChunkMethods.every((x) => x !== selectedTag);
  }, [selectedTag, isPdf]);

  const showOne = useMemo(() => {
    if (parserConfig?.layout_recognize === LayoutRecognizeType.MinerU) {
      return true;
    }
    return (
      isPdf &&
      hidePagesChunkMethods
        .filter((x) => x !== DocumentParserType.One)
        .every((x) => x !== selectedTag)
    );
  }, [selectedTag, isPdf]);

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
            onChange={handleChange}
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
