/**
 * Unit tests for the Linux user-systemd autostart integration, focused on the
 * crash-watchdog units added so a dead fleet self-heals.
 *
 * Run: pnpm vitest run test/autostart-watchdog.test.ts
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, rmSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// Capture every spawnSync call and report success so the autostart code thinks
// systemctl accepted enable/disable without actually touching the host.
const calls: { cmd: string; args: string[] }[] = [];
vi.mock('node:child_process', () => ({
  spawnSync: vi.fn((cmd: string, args: string[]) => {
    calls.push({ cmd, args: [...(args || [])] });
    return { status: 0, stdout: '', stderr: '' };
  }),
}));

import { enableAutostart, disableAutostart } from '../src/autostart.js';

let home: string;
function opts() {
  return {
    pkgRoot: join(home, 'pkg'),
    configDir: join(home, '.botmux'),
    logDir: join(home, '.botmux', 'logs'),
  };
}

describe('Linux autostart crash watchdog', () => {
  afterEach(() => {
    if (home) rmSync(home, { recursive: true, force: true });
    calls.length = 0;
  });

  it('enable writes and enables the watchdog service + timer', () => {
    home = mkdtempSync(join(tmpdir(), 'botmux-autostart-'));
    // Run only the Linux path regardless of the host running the tests.
    const o = opts();
    vi.stubEnv('HOME', home);
    // platform() reads process.platform; force the Linux branch by stubbing.
    vi.spyOn(process, 'platform', 'get').mockReturnValue('linux');
    // userSystemdAvailable() shells out; our mock returns status 0 already.

    enableAutostart(o);

    const unitDir = join(home, '.config', 'systemd', 'user');
    expect(existsSync(join(unitDir, 'botmux-watchdog.service'))).toBe(true);
    expect(existsSync(join(unitDir, 'botmux-watchdog.timer'))).toBe(true);

    // The timer is enabled --now so it starts watching immediately.
    const enableCall = calls.find(
      (c) => c.cmd === 'systemctl' && c.args.includes('botmux-watchdog.timer'),
    );
    expect(enableCall?.args).toEqual(
      expect.arrayContaining(['--user', 'enable', '--now', 'botmux-watchdog.timer']),
    );

    // The watchdog service runs the same idempotent start as the boot unit.
    const service = readFileSync(join(unitDir, 'botmux-watchdog.service'), 'utf-8');
    expect(service).toContain('Type=oneshot');
    expect(service).toMatch(/ExecStart=.* start\n/);

    // The timer fires periodically and survives reboots.
    const timer = readFileSync(join(unitDir, 'botmux-watchdog.timer'), 'utf-8');
    expect(timer).toContain('OnUnitActiveSec=30s');
    expect(timer).toContain('Persistent=true');
  });

  it('disable stops and removes the watchdog so it cannot resurrect the fleet', () => {
    home = mkdtempSync(join(tmpdir(), 'botmux-autostart-'));
    const o = opts();
    vi.stubEnv('HOME', home);
    vi.spyOn(process, 'platform', 'get').mockReturnValue('linux');

    enableAutostart(o);
    calls.length = 0;
    const unitDir = join(home, '.config', 'systemd', 'user');

    disableAutostart(o);

    const disableCall = calls.find(
      (c) => c.cmd === 'systemctl' && c.args.includes('botmux-watchdog.timer'),
    );
    expect(disableCall?.args).toEqual(
      expect.arrayContaining(['--user', 'disable', '--now', 'botmux-watchdog.timer']),
    );
    expect(existsSync(join(unitDir, 'botmux-watchdog.service'))).toBe(false);
    expect(existsSync(join(unitDir, 'botmux-watchdog.timer'))).toBe(false);
  });
});
