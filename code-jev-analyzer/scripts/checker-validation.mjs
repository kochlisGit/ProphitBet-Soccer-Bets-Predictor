export function healthySamples(payload, expected) {
  if (payload?.status !== 'success' || payload?.data?.resultType !== 'vector') return false;
  const rows = payload.data.result;
  return Array.isArray(rows) && rows.length > 0 && rows.every(row => {
    const raw = row?.value?.[1];
    return typeof raw === 'string' && raw.trim() !== '' && Number.isFinite(Number(raw)) && Number(raw) === expected;
  });
}
export function validAnswers(answers, questions) {
  return Object.keys(questions).every(key => answers?.[key]?.type === 'boolean'
    && Number.isFinite(answers[key].probability) && answers[key].probability >= 0 && answers[key].probability <= 1);
}
