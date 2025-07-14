import {
  AutoKeywordsFormField,
  AutoQuestionsFormField,
} from '@/components/auto-keywords-form-field';
import Delimiter from '@/components/delimiter';
import { LayoutRecognizeFormField } from '@/components/layout-recognize-form-field';
import MaxTokenNumber from '@/components/max-token-number';
import PageRankFormField from '@/components/page-rank-form-field';
import GraphRagItems from '@/components/parse-configuration/graph-rag-form-fields';
import { ConfigurationFormContainer } from '../configuration-form-container';
import { TagItems } from '../tag-item';
import { ChunkMethodItem, EmbeddingModelItem } from './common-item';

export function HierarchicalConfiguration() {
  return (
    <ConfigurationFormContainer>
      <ChunkMethodItem></ChunkMethodItem>
      <MaxTokenNumber initialValue={256}></MaxTokenNumber>
      <Delimiter></Delimiter>
      <LayoutRecognizeFormField></LayoutRecognizeFormField>
      <EmbeddingModelItem></EmbeddingModelItem>

      <PageRankFormField></PageRankFormField>

      <>
        <AutoKeywordsFormField></AutoKeywordsFormField>
        <AutoQuestionsFormField></AutoQuestionsFormField>
      </>

      <GraphRagItems marginBottom></GraphRagItems>

      <TagItems></TagItems>
    </ConfigurationFormContainer>
  );
}
