import { describe, it, expect, afterEach } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { openDatabaseSync } from '../src/services/sqlite-compat.js';
import { startCursorCot, stopAllCursorCot, findCursorStoreDb, type CursorCotEntry } from '../src/services/cursor-cot.js';

function line(obj: unknown): string {
  return JSON.stringify(obj) + '\n';
}

function assistantBlob(blocks: unknown[]) {
  return { role: 'assistant', content: blocks };
}

let dirs: string[] = [];

function setupStore(chatId: string): { dir: string; dbPath: string } {
  const dir = mkdtempSync(join(tmpdir(), 'cursor-cot-'));
  dirs.push(dir);
  const chatsRoot = join(dir, '.cursor', 'chats');
  const chatDir = join(chatsRoot, 'projhash', chatId);
  mkdirSync(chatDir, { recursive: true });
  return { dir: chatsRoot, dbPath: join(chatDir, 'store.db') };
}

afterEach(() => {
  stopAllCursorCot();
  for (const dir of dirs) rmSync(dir, { recursive: true, force: true });
  dirs = [];
});

describe('findCursorStoreDb', () => {
  it('finds <chatsRoot>/<hash>/<chatId>/store.db', async () => {
    const chatId = 'a80c1234-0000-4000-8000-000000000000';
    const { dir, dbPath } = setupStore(chatId);
    const db = await openDatabaseSync(dbPath);
    db.exec('CREATE TABLE blobs (data TEXT)');
    db.close();
    expect(findCursorStoreDb(chatId, dir)).toBe(dbPath);
    expect(findCursorStoreDb(chatId, join(dir, 'missing'))).toBeUndefined();
  });
});

describe('cursor CoT mapping', () => {
  it('maps reasoning / tool-call / tool-result / text blobs in order', async () => {
    const chatId = 'b80c1234-0000-4000-8000-000000000001';
    const { dir, dbPath } = setupStore(chatId);
    const db = await openDatabaseSync(dbPath);
    db.exec('CREATE TABLE blobs (data TEXT)');
    db.close();

    const done = new Promise<CursorCotEntry[]>((resolve) => {
      const got: CursorCotEntry[] = [];
      startCursorCot(chatId, (entries) => {
        got.push(...entries);
        if (got.some(e => e.kind === 'tool_result')) resolve(got);
      }, { chatsRoot: dir });
    });

    const callId = 'call-abc\nfc_1';
    const liveDb = await openDatabaseSync(dbPath);
    const insert = liveDb.prepare('INSERT INTO blobs (data) VALUES (?)');
    insert.run(line(assistantBlob([
      { type: 'reasoning', text: 'thinking first' },
      { type: 'text', text: 'a narration' },
      { type: 'tool-call', toolCallId: callId, toolName: 'Shell', args: { command: 'ls -la /tmp' } },
    ])));
    insert.run(line({
      role: 'tool',
      content: [{ type: 'tool-result', toolCallId: callId, result: 'total 0' }],
    }));
    liveDb.close();

    const got = await done;
    expect(got.map(e => e.kind)).toEqual(['thinking', 'text', 'tool_call', 'tool_result']);
    expect(got[0]).toMatchObject({ kind: 'thinking', text: 'thinking first' });
    const toolCall = got[2];
    if (toolCall.kind === 'tool_call') {
      expect(toolCall.id).toBe('call-abc_fc_1');
      expect(toolCall.name).toBe('Shell');
      expect(toolCall.subject).toBe('ls -la /tmp');
    }
    expect(got[3]).toMatchObject({ kind: 'tool_result', result: 'total 0' });
  });

  it('skips empty reasoning and unparsable blobs', async () => {
    const chatId = 'b80c1234-0000-4000-8000-000000000002';
    const { dir, dbPath } = setupStore(chatId);
    const db = await openDatabaseSync(dbPath);
    db.exec('CREATE TABLE blobs (data TEXT)');
    db.close();

    const got: CursorCotEntry[] = [];
    startCursorCot(chatId, (entries) => got.push(...entries), { chatsRoot: dir });

    const liveDb = await openDatabaseSync(dbPath);
    const insert = liveDb.prepare('INSERT INTO blobs (data) VALUES (?)');
    insert.run(line(assistantBlob([{ type: 'reasoning', text: '' }])));
    insert.run('not json\n');
    liveDb.close();

    await new Promise(r => setTimeout(r, 1500));
    expect(got).toEqual([]);
  });

  it('baselines at the existing end so historical blobs do not replay', async () => {
    const chatId = 'b80c1234-0000-4000-8000-000000000003';
    const { dir, dbPath } = setupStore(chatId);
    const db = await openDatabaseSync(dbPath);
    db.exec('CREATE TABLE blobs (data TEXT)');
    db.prepare('INSERT INTO blobs (data) VALUES (?)')
      .run(line(assistantBlob([{ type: 'reasoning', text: 'old thinking' }])));
    db.close();

    const got: CursorCotEntry[] = [];
    startCursorCot(chatId, (entries) => got.push(...entries), { chatsRoot: dir });
    await new Promise(r => setTimeout(r, 1500));
    expect(got).toEqual([]);
  });
});
