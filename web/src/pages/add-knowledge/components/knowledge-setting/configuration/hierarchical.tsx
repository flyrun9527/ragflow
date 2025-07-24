import {
  AutoKeywordsItem,
  AutoQuestionsItem,
} from '@/components/auto-keywords-item';
import { DatasetConfigurationContainer } from '@/components/dataset-configuration-container';
import LayoutRecognize from '@/components/layout-recognize';
import MaxMinTokenNumber from '@/components/max-min-token-number';
import PageRank from '@/components/page-rank';
import ParseConfiguration from '@/components/parse-configuration';
import GraphRagItems from '@/components/parse-configuration/graph-rag-items';
import { cn } from '@/lib/utils';
import { TagItems } from '../tag-item';
import { ChunkMethodItem, EmbeddingModelItem } from './common-item';

export function HierarchicalConfiguration() {
  return (
    <>
      <DatasetConfigurationContainer className={cn('mb-4')}>
        <LayoutRecognize></LayoutRecognize>
        <EmbeddingModelItem></EmbeddingModelItem>
        <ChunkMethodItem></ChunkMethodItem>
        <MaxMinTokenNumber></MaxMinTokenNumber>
      </DatasetConfigurationContainer>
      <DatasetConfigurationContainer className={cn('mb-4')}>
        <PageRank></PageRank>
        <>
          <AutoKeywordsItem></AutoKeywordsItem>
          <AutoQuestionsItem></AutoQuestionsItem>
        </>
        <TagItems></TagItems>
      </DatasetConfigurationContainer>

      <DatasetConfigurationContainer className={cn('mb-4')}>
        <ParseConfiguration></ParseConfiguration>
      </DatasetConfigurationContainer>

      <GraphRagItems marginBottom></GraphRagItems>
    </>
  );
}
