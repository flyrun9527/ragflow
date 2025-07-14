import {
  AutoKeywordsItem,
  AutoQuestionsItem,
} from '@/components/auto-keywords-item';
import Delimiter from '@/components/delimiter';
import LayoutRecognize from '@/components/layout-recognize';
import MaxTokenNumber from '@/components/max-token-number';
import PageRank from '@/components/page-rank';
import ParseConfiguration from '@/components/parse-configuration';
import GraphRagItems from '@/components/parse-configuration/graph-rag-items';
import { TagItems } from '../tag-item';
import { ChunkMethodItem, EmbeddingModelItem } from './common-item';

export function HierarchicalConfiguration() {
  return (
    <>
      <LayoutRecognize></LayoutRecognize>
      <EmbeddingModelItem></EmbeddingModelItem>
      <ChunkMethodItem></ChunkMethodItem>
      <MaxTokenNumber initialValue={256}></MaxTokenNumber>
      <Delimiter></Delimiter>

      <PageRank></PageRank>

      <>
        <AutoKeywordsItem></AutoKeywordsItem>
        <AutoQuestionsItem></AutoQuestionsItem>
      </>

      <ParseConfiguration></ParseConfiguration>

      <GraphRagItems marginBottom></GraphRagItems>

      <TagItems></TagItems>
    </>
  );
}
