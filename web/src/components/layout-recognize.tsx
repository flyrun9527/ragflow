import { LayoutRecognizeType, LlmModelType } from '@/constants/knowledge';
import { useTranslate } from '@/hooks/common-hooks';
import { useSelectLlmOptionsByModelType } from '@/hooks/llm-hooks';
import { Form, Select } from 'antd';
import { camelCase } from 'lodash';
import { useMemo } from 'react';

const LayoutRecognize = () => {
  const { t } = useTranslate('knowledgeDetails');
  const allOptions = useSelectLlmOptionsByModelType();

  const options = useMemo(() => {
    // 使用 LayoutRecognizeType 枚举值
    const list = Object.values(LayoutRecognizeType).map((x) => ({
      label: x === LayoutRecognizeType.PlainText ? t(camelCase(x)) : x,
      value: x,
    }));

    const image2TextList = allOptions[LlmModelType.Image2text].map((x) => {
      return {
        ...x,
        options: x.options.map((y) => {
          return {
            ...y,
            label: (
              <div className="flex justify-between items-center gap-2">
                {y.label}
                <span className="text-red-500 text-sm">Experimental</span>
              </div>
            ),
          };
        }),
      };
    });

    return [...list, ...image2TextList];
  }, [allOptions, t]);

  return (
    <Form.Item
      name={['parser_config', 'layout_recognize']}
      label={t('layoutRecognize')}
      initialValue={LayoutRecognizeType.DeepDOC}
      tooltip={t('layoutRecognizeTip')}
    >
      <Select options={options} popupMatchSelectWidth={false} />
    </Form.Item>
  );
};

export default LayoutRecognize;
