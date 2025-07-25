import { useSelectParserList } from '@/hooks/user-setting-hooks';
import { useCallback, useMemo } from 'react';

const ParserListMap = new Map([
  [
    ['pdf'],
    [
      'naive',
      'resume',
      'manual',
      'paper',
      'book',
      'laws',
      'presentation',
      'one',
      'qa',
      'knowledge_graph',
      'hierarchical',
      'strict_regex',
    ],
  ],
  [
    ['doc', 'docx'],
    [
      'naive',
      'resume',
      'book',
      'laws',
      'one',
      'qa',
      'manual',
      'knowledge_graph',
      'hierarchical',
      'strict_regex',
    ],
  ],
  [
    ['xlsx', 'xls'],
    [
      'naive',
      'qa',
      'table',
      'one',
      'knowledge_graph',
      'hierarchical',
      'strict_regex',
    ],
  ],
  [
    ['ppt', 'pptx'],
    ['presentation', 'hierarchical', 'strict_regex'],
  ],
  [
    ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff', 'webp', 'svg', 'ico'],
    ['picture', 'hierarchical', 'strict_regex'],
  ],
  [
    ['txt'],
    [
      'naive',
      'resume',
      'book',
      'laws',
      'one',
      'qa',
      'table',
      'knowledge_graph',
      'hierarchical',
      'strict_regex',
    ],
  ],
  [
    ['csv'],
    [
      'naive',
      'resume',
      'book',
      'laws',
      'one',
      'qa',
      'table',
      'knowledge_graph',
      'hierarchical',
      'strict_regex',
    ],
  ],
  [
    ['md', 'mdx'],
    ['naive', 'qa', 'knowledge_graph', 'hierarchical', 'strict_regex'],
  ],
  [['json'], ['naive', 'knowledge_graph', 'hierarchical', 'strict_regex']],
  [['eml'], ['email', 'hierarchical', 'strict_regex']],
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

export const useFetchParserListOnMount = (documentExtension: string) => {
  const parserList = useSelectParserList();

  const nextParserList = useMemo(() => {
    const key = [...ParserListMap.keys()].find((x) =>
      x.some((y) => y === documentExtension),
    );
    if (key) {
      const values = ParserListMap.get(key);
      return getParserList(values ?? [], parserList);
    }

    return getParserList(
      ['naive', 'resume', 'book', 'laws', 'one', 'qa', 'table'],
      parserList,
    );
  }, [parserList, documentExtension]);

  return { parserList: nextParserList };
};

const hideAutoKeywords = ['qa', 'table', 'resume', 'knowledge_graph', 'tag'];

export const useShowAutoKeywords = () => {
  const showAutoKeywords = useCallback((selectedTag: string) => {
    return hideAutoKeywords.every((x) => selectedTag !== x);
  }, []);

  return showAutoKeywords;
};
