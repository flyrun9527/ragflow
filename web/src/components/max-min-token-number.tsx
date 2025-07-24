import { useTranslate } from '@/hooks/common-hooks';
import { Flex, Form, InputNumber, Slider } from 'antd';

interface IProps {
  initialValue?: number;
  max?: number;
}

const MaxMinTokenNumber = ({ initialValue = 256, max = 2048 }: IProps) => {
  const { t } = useTranslate('knowledgeConfiguration');

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
    </>
  );
};

export default MaxMinTokenNumber;
