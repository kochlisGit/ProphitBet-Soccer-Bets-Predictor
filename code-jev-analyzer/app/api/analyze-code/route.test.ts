import { describe, it, expect, vi } from 'vitest';
import { NextRequest } from 'next/server';
import { POST } from './route';
import { analyzeCode } from '../../../lib/analyze-code';
vi.mock('../../../lib/analyze-code', () => ({ analyzeCode: vi.fn() }));
const request = (body: string) => new NextRequest('https://example.test/api/analyze-code', {method:'POST', body});
describe('honest evaluation failure', () => {
  it('does not fabricate probabilities when inference fails', async () => {
    vi.mocked(analyzeCode).mockRejectedValueOnce(new Error('private provider credential'));
    const response = await POST(request(JSON.stringify({code:'const a = 1;', filename:'test.ts'})));
    const data = await response.json();
    expect(response.status).toBe(503);
    expect(data.answers).toBeUndefined();
    expect(JSON.stringify(data)).not.toContain('private provider credential');
  });
  it('rejects malformed JSON', async () => {
    expect((await POST(request('{'))).status).toBe(400);
  });
  it('does not pretend regex rewrites repair code', async () => {
    expect((await POST(request(JSON.stringify({code:'a',filename:'a.ts',action:'fix'})))).status).toBe(422);
  });
});
