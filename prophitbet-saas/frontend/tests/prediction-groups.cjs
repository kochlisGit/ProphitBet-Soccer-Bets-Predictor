const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../lib/prediction-groups.ts'), 'utf8');
const exportsObject = {};
new Function('exports', ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText)(exportsObject);
const group = exportsObject.groupMatchPredictions;
const a = { id: '1', league_id: 1, home_team: 'A', away_team: 'B', match_date: '2026-10-02T10:00:00Z', market_type: 'result', created_at: '2026-09-29' };
const b = { ...a, id: '2', market_type: 'over_under' };
assert.equal(group([a, b]).length, 1);
assert.equal(group([b, a])[0].id, '1');
assert.equal(group([a, b])[0].markets.length, 2);
assert.equal(group([a, { ...a, id: '3', created_at: '2026-09-30' }])[0].id, '3');
assert.equal(group([a, { ...a, league_id: 2 }]).length, 2);
assert.equal(group([a, { ...a, match_date: '2026-10-03T10:00:00Z' }]).length, 2);
assert.equal(group([a, { ...b, match_date: '2026-10-02T12:00:00+02:00' }]).length, 1);
assert.equal(group([{ ...a, match_date: null }, { ...b, match_date: null }]).length, 2);
console.log('8 prediction grouping checks passed');
