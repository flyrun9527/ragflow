import { DocumentParserType } from '@/constants/knowledge';
import { useHandleChunkMethodSelectChange } from '@/hooks/logic-hooks';
import { useSelectParserList } from '@/hooks/user-setting-hooks';
import { FormInstance } from 'antd';
import { useCallback, useEffect, useMemo, useState } from 'react';

const ParserListMap = new Map([
  [
    ['pdf'],
    [
      DocumentParserType.Naive,
      DocumentParserType.Resume,
      DocumentParserType.Manual,
      DocumentParserType.Paper,
      DocumentParserType.Book,
      DocumentParserType.Laws,
      DocumentParserType.Presentation,
      DocumentParserType.One,
      DocumentParserType.Qa,
      DocumentParserType.KnowledgeGraph,
      DocumentParserType.Hierarchical,
    ],
  ],
  [
    ['doc', 'docx'],
    [
      DocumentParserType.Naive,
      DocumentParserType.Resume,
      DocumentParserType.Book,
      DocumentParserType.Laws,
      DocumentParserType.One,
      DocumentParserType.Qa,
      DocumentParserType.Manual,
      DocumentParserType.KnowledgeGraph,
      DocumentParserType.Hierarchical,
    ],
  ],
  [
    ['xlsx', 'xls'],
    [
      DocumentParserType.Naive,
      DocumentParserType.Qa,
      DocumentParserType.Table,
      DocumentParserType.One,
      DocumentParserType.KnowledgeGraph,
      DocumentParserType.Hierarchical,
    ],
  ],
  [
    ['ppt', 'pptx'],
    [DocumentParserType.Presentation, DocumentParserType.Hierarchical],
  ],
  [
    ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff', 'webp', 'svg', 'ico'],
    [DocumentParserType.Picture],
  ],
  [
    ['txt'],
    [
      DocumentParserType.Naive,
      DocumentParserType.Resume,
      DocumentParserType.Book,
      DocumentParserType.Laws,
      DocumentParserType.One,
      DocumentParserType.Qa,
      DocumentParserType.Table,
      DocumentParserType.KnowledgeGraph,
      DocumentParserType.Hierarchical,
    ],
  ],
  [
    ['csv'],
    [
      DocumentParserType.Naive,
      DocumentParserType.Resume,
      DocumentParserType.Book,
      DocumentParserType.Laws,
      DocumentParserType.One,
      DocumentParserType.Qa,
      DocumentParserType.Table,
      DocumentParserType.KnowledgeGraph,
      DocumentParserType.Hierarchical,
    ],
  ],
  [
    ['md'],
    [
      DocumentParserType.Naive,
      DocumentParserType.Qa,
      DocumentParserType.KnowledgeGraph,
      DocumentParserType.Hierarchical,
      DocumentParserType.Hierarchical,
    ],
  ],
  [
    ['json'],
    [
      DocumentParserType.Naive,
      DocumentParserType.KnowledgeGraph,
      DocumentParserType.Hierarchical,
    ],
  ],
  [['eml'], [DocumentParserType.Email, DocumentParserType.Hierarchical]],
]);

const getParserList = (
  values: string[],
  parserList: Array<{
    value: string;
    label: string;
  }>,
) => {
  return parserList.filter((x) => values?.some((y) => y === x.value));
};

export const useFetchParserListOnMount = (
  documentId: string,
  parserId: DocumentParserType,
  documentExtension: string,
  form: FormInstance,
) => {
  const [selectedTag, setSelectedTag] = useState<DocumentParserType>();
  const parserList = useSelectParserList();
  const handleChunkMethodSelectChange = useHandleChunkMethodSelectChange(form);

  const nextParserList = useMemo(() => {
    const key = [...ParserListMap.keys()].find((x) =>
      x.some((y) => y === documentExtension),
    );
    if (key) {
      const values = ParserListMap.get(key);
      return getParserList(values ?? [], parserList);
    }

    return getParserList(
      [
        DocumentParserType.Naive,
        DocumentParserType.Resume,
        DocumentParserType.Book,
        DocumentParserType.Laws,
        DocumentParserType.One,
        DocumentParserType.Qa,
        DocumentParserType.Table,
        DocumentParserType.Hierarchical,
      ],
      parserList,
    );
  }, [parserList, documentExtension]);

  useEffect(() => {
    setSelectedTag(parserId);
  }, [parserId, documentId]);

  const handleChange = (tag: string) => {
    handleChunkMethodSelectChange(tag);
    setSelectedTag(tag as DocumentParserType);
  };

  return { parserList: nextParserList, handleChange, selectedTag };
};

const hideAutoKeywords = [
  DocumentParserType.Qa,
  DocumentParserType.Table,
  DocumentParserType.Resume,
  DocumentParserType.KnowledgeGraph,
  DocumentParserType.Tag,
];

export const useShowAutoKeywords = () => {
  const showAutoKeywords = useCallback(
    (selectedTag: DocumentParserType | undefined) => {
      return hideAutoKeywords.every((x) => selectedTag !== x);
    },
    [],
  );

  return showAutoKeywords;
};
