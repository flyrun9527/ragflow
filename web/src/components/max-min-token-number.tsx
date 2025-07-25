import { DocumentParserType } from '@/constants/knowledge';
import { useTranslate } from '@/hooks/common-hooks';
import { Flex, Form, Input, InputNumber, Slider } from 'antd';

interface IProps {
  initialValue?: number;
  max?: number;
  selectedTag?: DocumentParserType;
}

const MaxMinTokenNumber = ({
  initialValue = 256,
  max = 2048,
  selectedTag,
}: IProps) => {
  const { t } = useTranslate('knowledgeConfiguration');

  // 判断是否显示正则表达式输入
  const showRegexPattern = selectedTag === DocumentParserType.StrictRegex;

  // 正则表达式的默认值
  const defaultRegexPattern = '第[零一二三四五六七八九十百千万\\d]+条';

  return (
    <>
      <Form.Item
        label={t('maxChunkTokenNumber')}
        tooltip={t('maxChunkTokenNumberTip')}
      >
        <Flex gap={20} align="center">
          <Flex flex={1}>
            <Form.Item
              name={['parser_config', 'chunk_token_num']}
              noStyle
              initialValue={initialValue}
              rules={[
                { required: true, message: t('maxChunkTokenNumberMessage') },
              ]}
            >
              <Slider max={max} min={50} style={{ width: '100%' }} />
            </Form.Item>
          </Flex>
          <Form.Item
            name={['parser_config', 'chunk_token_num']}
            noStyle
            rules={[
              { required: true, message: t('maxChunkTokenNumberMessage') },
            ]}
          >
            <InputNumber max={2048} min={50} />
          </Form.Item>
        </Flex>
      </Form.Item>
      <Form.Item
        label={t('minChunkTokenNumber')}
        tooltip={t('minChunkTokenNumberTip')}
      >
        <Flex gap={20} align="center">
          <Flex flex={1}>
            <Form.Item
              name={['parser_config', 'min_chunk_token_num']}
              noStyle
              initialValue={10}
              rules={[
                { required: true, message: t('minChunkTokenNumberMessage') },
              ]}
            >
              <Slider max={500} min={10} style={{ width: '100%' }} />
            </Form.Item>
          </Flex>
          <Form.Item
            name={['parser_config', 'min_chunk_token_num']}
            noStyle
            rules={[
              { required: true, message: t('minChunkTokenNumberMessage') },
            ]}
          >
            <InputNumber max={500} min={10} />
          </Form.Item>
        </Flex>
      </Form.Item>

      {/* 根据条件显示正则表达式输入 */}
      {showRegexPattern && (
        <Form.Item
          initialValue={defaultRegexPattern}
          name={['parser_config', 'regex_pattern']}
          label={t('regexPattern')}
          tooltip={t('regexPatternTip')}
          rules={[
            {
              required: true,
              message: t('regexPatternMessage') || 'Please input regex pattern',
            },
          ]}
          help={t('regexPatternHint')}
        >
          <Input placeholder={t('regexPatternPlaceholder')} allowClear />
        </Form.Item>
      )}
    </>
  );
};

export default MaxMinTokenNumber;
