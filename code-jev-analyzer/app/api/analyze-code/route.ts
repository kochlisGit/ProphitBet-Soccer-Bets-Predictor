import { NextRequest, NextResponse } from 'next/server';
import { analyzeCode } from '../../../lib/analyze-code';

export async function POST(request: NextRequest) {
  let body;
  try {
    const raw = await request.text();
    if (raw.length > 100_000) return NextResponse.json({ error: 'Input too large.' }, { status: 413 });
    body = JSON.parse(raw);
  } catch {
    return NextResponse.json({ error: 'Invalid JSON.' }, { status: 400 });
  }
  const { code, filename, action } = body ?? {};
  if (typeof code !== 'string' || !code.trim() || typeof filename !== 'string' || !filename.trim() || filename.length > 255) {
    return NextResponse.json({ error: 'Non-empty code and filename are required.' }, { status: 400 });
  }
  if (action && action !== 'analyze') {
    return NextResponse.json({ error: 'Automatic regex fixes are disabled: they cannot safely validate or repair code.' }, { status: 422 });
  }
  try {
    const result = await analyzeCode(code, filename);
    return NextResponse.json({ ...result, engine: 'typesafe-ai/jev', status: 'evaluated',
      limitation: 'AI code review is advisory. It does not verify football facts or guarantee correctness.' });
  } catch {
    // Never replace failed inference with fabricated probabilities or expose provider credentials.
    return NextResponse.json({ status: 'unavailable', engine: 'typesafe-ai/jev',
      error: 'JEv evaluation unavailable. Check server gateway credentials, credits and provider availability. No validation was performed.' },
      { status: 503 });
  }
}
