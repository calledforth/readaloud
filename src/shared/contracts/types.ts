// Shared TypeScript contracts (placeholders) — Section F will finalize

export interface PrepareDocumentInput {
  kind: 'pdf_base64' | 'raw_text';
  pdf_base64?: string;
  raw_text?: string;
  language?: string; // default 'en'
  max_paragraph_chars?: number; // default 2000
}

export interface PrepareDocumentRequest {
  op: 'prepare_document';
  doc_id: string;
  input: PrepareDocumentInput;
}

export interface ParagraphRef {
  paragraph_id: string;
  text: string;
}

export interface PrepareDocumentResponseOk {
  ok: true;
  doc_id: string;
  paragraphs: ParagraphRef[];
  cleaning_notes?: string[];
  version: string;
}

export interface SynthesizeChunkRequest {
  op: 'synthesize_chunk';
  doc_id: string;
  paragraph_id: string;
  text: string;
  voice: string;
  rate: number; // 1.0 default
  sample_rate: number; // e.g., 24000
}

export interface WordTiming {
  word: string;
  start_ms: number;
  end_ms: number;
  char_start: number;
  char_end: number;
}

export interface SynthesizeChunkResponseOk {
  ok: true;
  doc_id: string;
  paragraph_id: string;
  cleaned_text: string;
  audio_base64: string;
  sample_rate: number;
  timings: WordTiming[];
  inference_ms: { tts: number; align: number; total: number };
  version: string;
}

export interface ErrorResponse {
  ok: false;
  code: 'BadInput' | 'Timeout' | 'ModelLoad' | 'AlignError' | 'AlignWarning' | 'Internal';
  message: string;
}


