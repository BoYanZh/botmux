import { describe, it, expect, afterEach } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { openDatabaseSync } from '../src/services/sqlite-compat.js';
import { startCursorCot, stopAllCursorCot, type CursorCotEntry } from '../src/services/cursor-cot.js';

function line(obj: unknown): string {
  return JSON.stringify(obj) + '\n';
}

let dirs: string[] = [];

function setupStore(): { chatsRoot: string; dbPath: string; chatId: string } {
  const dir = mkdtempSync(join(tmpdir(), 'cursor-cot-'));
  dirs.push(dir);
  const chatId = 'b80c1234-0000-4000-8000-000000000000';
  const chatsRoot = join(dir, '.cursor', 'chats');
  const chatDir = join(chatsRoot, 'projhash', chatId);
  mkdirSync(chatDir, { recursive: true });
  return { chatsRoot, dbPath: join(chatDir, 'store.db'), chatId };
}

async function openFreshStore(dbPath: string) {
  const db = await openDatabaseSync(dbPath);
  db.exec('CREATE TABLE blobs (data TEXT)');
  return db;
}

function startCollector(chatsRoot: string, chatId: string): { got: CursorCotEntry[]; waitFor: (kind: CursorCotEntry['kind']) => Promise<void> } {
  const got: CursorCotEntry[] = [];
  let resolver: (() => void) | undefined;
  let wanted: CursorCotEntry['kind'] | undefined;
  startCursorCot(chatId, (entries) => {
    got.push(...entries);
    if (wanted && got.some(e => e.kind === wanted)) {
      wanted = undefined;
      resolver?.();
    }
  }, { chatsRoot });
  return {
    got,
    waitFor: (kind) => new Promise<void>((resolve) => {
      wanted = kind;
      resolver = resolve;
      if (got.some(e => e.kind === kind)) resolve();
    }),
  };
}

afterEach(() => {
  stopAllCursorCot();
  for (const dir of dirs) rmSync(dir, { recursive: true, force: true });
  dirs = [];
});

describe('cursor CoT mapping', () => {
  it('maps reasoning / tool-call / tool-result / text blocks in order', async () => {
    const { chatsRoot, dbPath, chatId } = setupStore();
    const init = await openFreshStore(dbPath);
    init.close();

    const collector = startCollector(chatsRoot, chatId);
    const callId = 'call-abc\nfc_1';
    const db = await openDatabaseSync(dbPath);
    const insert = db.prepare('INSERT INTO blobs (data) VALUES (?)');
    insert.run(line({
      id: 'asst-1', role: 'assistant',
      content: [
        { type: 'reasoning', text: 'thinking first' },
        { type: 'text', text: 'a narration' },
        { type: 'tool-call', toolCallId: callId, toolName: 'Shell', args: { command: 'ls -la /tmp' } },
      ],
    }));
    insert.run(line({
      id: 'tool-1', role: 'tool',
      content: [{ type: 'tool-result', toolCallId: callId, result: 'total 0' }],
    }));
    db.close();

    await collector.waitFor('tool_result');
    expect(collector.got.map(e => e.kind)).toEqual(['thinking', 'text', 'tool_call', 'tool_result']);
    expect(collector.got[0]).toMatchObject({ kind: 'thinking', text: 'thinking first' });
    const toolCall = collector.got[2];
    if (toolCall.kind === 'tool_call') {
      expect(toolCall.id).toBe('call-abc_fc_1');
      expect(toolCall.name).toBe('Shell');
      expect(toolCall.subject).toBe('ls -la /tmp');
    }
    expect(collector.got[3]).toMatchObject({ kind: 'tool_result', result: 'total 0' });
  });

  it('F1: user and system rows never produce entries', async () => {
    const { chatsRoot, dbPath, chatId } = setupStore();
    const init = await openFreshStore(dbPath);
    init.close();

    const collector = startCollector(chatsRoot, chatId);
    const db = await openDatabaseSync(dbPath);
    const insert = db.prepare('INSERT INTO blobs (data) VALUES (?)');
    // Raw user prompt + hidden botmux envelope — must stay out of the bubble.
    insert.run(line({
      id: 'user-1', role: 'user',
      content: [{ type: 'text', text: '<user_query>do the thing</user_query>\n<botmux_routing>hidden envelope' }],
    }));
    insert.run(line({
      id: 'sys-1', role: 'system',
      content: [{ type: 'text', text: 'system text' }],
    }));
    // One assistant blob proves the reader tick ran.
    insert.run(line({
      id: 'asst-1', role: 'assistant',
      content: [{ type: 'reasoning', text: 'real thinking' }],
    }));
    db.close();

    await collector.waitFor('thinking');
    expect(collector.got).toHaveLength(1);
    expect(collector.got[0]).toMatchObject({ kind: 'thinking', text: 'real thinking' });
  });

  it('skips empty reasoning and unparsable blobs', async () => {
    const { chatsRoot, dbPath, chatId } = setupStore();
    const init = await openFreshStore(dbPath);
    init.close();

    const collector = startCollector(chatsRoot, chatId);
    const db = await openDatabaseSync(dbPath);
    const insert = db.prepare('INSERT INTO blobs (data) VALUES (?)');
    insert.run(line({ id: 'asst-1', role: 'assistant', content: [{ type: 'reasoning', text: '' }] }));
    insert.run('not json\n');
    insert.run(line({ id: 'asst-2', role: 'assistant', content: [{ type: 'reasoning', text: 'later thinking' }] }));
    db.close();

    await collector.waitFor('thinking');
    expect(collector.got).toEqual([{ kind: 'thinking', text: 'later thinking' }]);
  });

  it('does not replay the same blob after a rowid re-sweep', async () => {
    const { chatsRoot, dbPath, chatId } = setupStore();
    const init = await openFreshStore(dbPath);
    init.close();

    const collector = startCollector(chatsRoot, chatId);
    const db = await openDatabaseSync(dbPath);
    db.prepare('INSERT INTO blobs (data) VALUES (?)')
      .run(line({ id: 'asst-1', role: 'assistant', content: [{ type: 'reasoning', text: 'once only' }] }));
    db.close();

    await collector.waitFor('thinking');
    expect(collector.got).toHaveLength(1);

    // Simulate rowid reuse below the cursor: tick re-sweeps from 0, the
    // stable blob id must dedupe it.
    const second = await openDatabaseSync(dbPath);
    second.exec('DELETE FROM blobs');
    second.prepare('INSERT INTO blobs (data) VALUES (?)')
      .run(line({ id: 'asst-1', role: 'assistant', content: [{ type: 'reasoning', text: 'once only' }] }));
    second.close();

    await new Promise(r => setTimeout(r, 1800));
    expect(collector.got).toHaveLength(1);
  });

  it('returns false when the store cannot be resolved', () => {
    const ok = startCursorCot('missing-chat-id', () => {}, { chatsRoot: join(tmpdir(), 'no-such-root') });
    expect(ok).toBe(false);
  });
});
