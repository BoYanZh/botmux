/**
 * Cursor CoT (thinking process) side-channel reader.
 *
 * Cursor Agent's agent-transcript JSONL carries only user prompts and the
 * final reply — no reasoning, no tool calls/results (tool_use lines are
 * dropped by cursor-transcript.ts). The model-visible timeline DOES persist,
 * in the per-chat SQLite store:
 *   ~/.cursor/chats/<projectHash>/<chatId>/store.db
 * Each row of `blobs` is one provider message: assistant blobs hold
 * reasoning / tool-call / text content blocks, tool blobs hold tool-result
 * blocks. Rows are append-only in rowid order.
 *
 * This module incrementally reads new blobs from a baselined rowid and maps
 * them, in order, to the CoT entries the native thinking bubble renders:
 * reasoning text → thinking; tool-call → tool node (command/path extracted as
 * subject BEFORE truncation); tool-result → result node; assistant text →
 * interim narration (the turn's closing text repeats at the bubble tail, the
 * same trade-off the Claude bridge accepts). Purely cosmetic: every read
 * catches its own errors and never affects turn settlement.
 *
 * Read-only SQLite access via sqlite-compat (bun:sqlite on the compiled
 * binary, node:sqlite under Node).
 */
import { existsSync, readdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import { openDatabaseSyncNow } from './sqlite-compat.js';
import { boundSubjectForTransport, subjectFromArgsString, subjectFromInputObject } from './cot-subject.js';

const COT_TOOL_ARGS_MAX_CHARS = 600;
const COT_TOOL_RESULT_MAX_CHARS = 800;
const BATCH_LIMIT = 500;
const POLL_INTERVAL_MS = 1_000;

export type CursorCotEntry =
  | { kind: 'thinking'; text: string }
  | { kind: 'text'; text: string }
  | {
    kind: 'tool_call'; id: string; name: string; args: string;
    subject?: string;
  }
  | { kind: 'tool_result'; id: string; result: string };

interface ReaderState {
  chatId: string;
  timer: NodeJS.Timeout;
  lastRowid: number;
  chatsRoot?: string;
}

const readers = new Map<string, ReaderState>();

function cursorChatsRoot(rootOverride?: string): string {
  return rootOverride ?? join(homedir(), '.cursor', 'chats');
}

/** Locate the store.db for a chatId: ~/.cursor/chats/<projectHash>/<chatId>/store.db. */
export function findCursorStoreDb(chatId: string, chatsRoot?: string): string | undefined {
  if (!chatId) return undefined;
  const root = cursorChatsRoot(chatsRoot);
  if (!existsSync(root)) return undefined;
  for (const projectHash of readdirSync(root)) {
    const candidate = join(root, projectHash, chatId, 'store.db');
    if (existsSync(candidate)) return candidate;
  }
  return undefined;
}

/** Cursor encodes a tool call id as `<providerCallId>\n<functionCallId>` — a
 *  literal newline that must not reach the AG-UI node id. Normalize to a safe
 *  token; tool-call and tool-result derive from the same raw id, so pairing
 *  stays consistent. */
function sanitizeToolId(rawId: unknown): string {
  return String(rawId ?? '').replace(/[^a-zA-Z0-9_-]/g, '_');
}

function truncateForCot(s: string, max: number): string {
  return s.length > max ? `${s.slice(0, max)}…` : s;
}

function toolCallEntry(block: any): CursorCotEntry | null {
  const id = sanitizeToolId(block?.toolCallId);
  const name = typeof block?.toolName === 'string' ? block.toolName : 'tool';
  if (!id) return null;
  let args = '';
  let rawSubject = '';
  if (typeof block.args === 'string') {
    args = block.args;
    rawSubject = subjectFromArgsString(args);
  } else if (block.args !== undefined) {
    rawSubject = subjectFromInputObject(block.args);
    try { args = JSON.stringify(block.args); } catch { /* unserialisable — show none */ }
  }
  const subject = boundSubjectForTransport(rawSubject);
  return {
    kind: 'tool_call', id, name,
    args: truncateForCot(args, COT_TOOL_ARGS_MAX_CHARS),
    ...(subject ? { subject } : {}),
  };
}

function stringifyToolResult(result: unknown): string {
  if (typeof result === 'string') return result;
  if (result === undefined || result === null) return '';
  try { return JSON.stringify(result); } catch { return ''; }
}

function toolResultEntry(block: any): CursorCotEntry | null {
  const id = sanitizeToolId(block?.toolCallId);
  if (!id) return null;
  const result = truncateForCot(stringifyToolResult(block?.result), COT_TOOL_RESULT_MAX_CHARS);
  return { kind: 'tool_result', id, result };
}

/** Map one parsed blob's content blocks to CoT entries, in block order. */
function entriesFromBlob(blob: any): CursorCotEntry[] {
  if (!blob || typeof blob !== 'object') return [];
  const content = blob.content;
  if (!Array.isArray(content)) return [];
  const entries: CursorCotEntry[] = [];
  for (const block of content) {
    if (!block || typeof block !== 'object') continue;
    switch (block.type) {
      case 'reasoning':
        if (typeof block.text === 'string' && block.text.trim().length > 0) {
          entries.push({ kind: 'thinking', text: block.text });
        }
        break;
      case 'text':
        if (typeof block.text === 'string' && block.text.trim().length > 0) {
          entries.push({ kind: 'text', text: block.text });
        }
        break;
      case 'tool-call': {
        const entry = toolCallEntry(block);
        if (entry) entries.push(entry);
        break;
      }
      case 'tool-result': {
        const entry = toolResultEntry(block);
        if (entry) entries.push(entry);
        break;
      }
    }
  }
  return entries;
}

function blobText(data: unknown): string | undefined {
  if (typeof data === 'string') return data;
  if (data instanceof Uint8Array) return Buffer.from(data).toString('utf8');
  return undefined;
}

/** Read and map new blobs since rowid `state.lastRowid`. Binary / encrypted
 *  rows that fail to parse are skipped but advance the cursor. */
function readNewEntries(state: ReaderState, dbPath: string, onEntries: (entries: readonly CursorCotEntry[]) => void): void {
  const db = openDatabaseSyncNow(dbPath, { readOnly: true });
  if (!db) return;
  try {
    const rows = db
      .prepare('SELECT rowid AS rowid, data AS data FROM blobs WHERE rowid > ? ORDER BY rowid LIMIT ?')
      .all(state.lastRowid, BATCH_LIMIT) as Array<{ rowid: number | bigint; data: unknown }>;
    for (const row of rows) {
      state.lastRowid = Number(row.rowid);
      const text = blobText(row.data);
      if (text === undefined) continue;
      let blob: any;
      try { blob = JSON.parse(text); } catch { continue; }
      const entries = entriesFromBlob(blob);
      if (entries.length > 0) {
        try { onEntries(entries); } catch { /* cosmetic channel — never break the read loop */ }
      }
    }
  } finally {
    try { db.close(); } catch { /* ignore */ }
  }
}

export interface StartCursorCotOptions {
  /** Override the ~/.cursor/chats root (tests point this at a temp dir). */
  chatsRoot?: string;
  /** Explicit store.db path; skips the root scan when provided. */
  dbPath?: string;
}

/** Start (or restart) the CoT reader for a chat. Baselines at the current
 *  max rowid so historical blobs are not replayed; subsequent blobs poll at
 *  1s. The timer is unref'd so it can never hold the process alive. */
export function startCursorCot(
  chatId: string,
  onEntries: (entries: readonly CursorCotEntry[]) => void,
  options: StartCursorCotOptions = {},
): void {
  if (!chatId) return;
  let existing = readers.get(chatId);
  if (existing) clearTimer(existing);
  const dbPath = options.dbPath ?? findCursorStoreDb(chatId, options.chatsRoot);
  if (!dbPath) return;
  const db = openDatabaseSyncNow(dbPath, { readOnly: true });
  let baseline = 0;
  if (db) {
    try {
      const row = db.prepare('SELECT max(rowid) AS m FROM blobs').get() as { m?: number | bigint };
      baseline = Number(row?.m ?? 0);
    } finally {
      try { db.close(); } catch { /* ignore */ }
    }
  }
  const state: ReaderState = {
    chatId, timer: undefined as unknown as NodeJS.Timeout,
    lastRowid: baseline, chatsRoot: options.chatsRoot,
  };
  state.timer = setInterval(() => {
    const path = options.dbPath ?? findCursorStoreDb(chatId, options.chatsRoot) ?? dbPath;
    try { readNewEntries(state, path, onEntries); } catch { /* cosmetic channel */ }
  }, POLL_INTERVAL_MS);
  if (typeof state.timer.unref === 'function') state.timer.unref();
  readers.set(chatId, state);
}

function clearTimer(state: ReaderState): void {
  clearInterval(state.timer);
}

/** Stop the CoT reader for a chat, if any. */
export function stopCursorCot(chatId: string): void {
  const state = readers.get(chatId);
  if (!state) return;
  clearTimer(state);
  readers.delete(chatId);
}

/** Reset the rowid cursor to the current end of the store, so subsequent
 *  reads only expose blobs written after this call. No-op when no reader is
 *  running — startCursorCot itself baselines at max rowid. */
export function rebaselineCursorCot(chatId: string): void {
  const existing = readers.get(chatId);
  if (!existing) return;
  const dbPath = findCursorStoreDb(chatId);
  const db = dbPath ? openDatabaseSyncNow(dbPath, { readOnly: true }) : null;
  if (!db) return;
  try {
    const row = db.prepare('SELECT max(rowid) AS m FROM blobs').get() as { m?: number | bigint };
    existing.lastRowid = Number(row?.m ?? 0);
  } finally {
    try { db.close(); } catch { /* ignore */ }
  }
}

/** Stop every active CoT reader (full teardown). */
export function stopAllCursorCot(): void {
  for (const state of readers.values()) clearTimer(state);
  readers.clear();
}

export const cursorCotChatIdFromTranscriptPath = (transcriptPath: string): string =>
  basename(dirname(transcriptPath));
