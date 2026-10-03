const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const assert = require('node:assert/strict');
const exportsObject = {};
new Function('exports', 'require', ts.transpileModule(fs.readFileSync(path.join(__dirname, '../lib/utils.ts'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText)(exportsObject, require);
const match = exportsObject.matchesLeagueFilter;
const english = { league_id: 6, league_name: 'Premier-League', country: 'England' };
const russian = { league_id: 26, league_name: 'Premier-League', country: 'Russia' };
assert.equal(match(english, 'Premier League'), true);
assert.equal(match(russian, 'Premier League'), false);
assert.equal(match(russian, 'Premier-League', 6), false);
assert.equal(match({league_name:'Premier-League'}, 'Premier-League', 6), false);
assert.equal(match(russian, 'league:26'), true);
assert.equal(match(english, 'league:26'), false);
assert.equal(match(russian, 'Russian Premier League'), true);
assert.equal(match({...english, country:undefined}, 'Premier League'), false);
assert.equal(match({league_name:'Serie-A', country:'Brazil'}, 'Serie-A'), false);
assert.equal(match({league_name:'Premiership', country:'Scotland'}, 'PSL'), false);
assert.equal(match(russian, 'ALL'), true);
console.log('11 league identity regression checks passed');
