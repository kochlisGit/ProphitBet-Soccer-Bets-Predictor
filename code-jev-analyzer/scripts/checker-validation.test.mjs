import test from 'node:test';
import assert from 'node:assert/strict';
import {healthySamples,validAnswers} from './checker-validation.mjs';
test('missing telemetry is not a pass', () => {
  for (const result of [[],[{value:[0,'NaN']}],[{value:[0,'']}],[{value:[0,'0']}]] ) {
    assert.equal(healthySamples({status:'success',data:{resultType:'vector',result}},1),false);
  }
  assert.equal(healthySamples({status:'success',data:{resultType:'vector',result:[{value:[0,'1']}]}},1),true);
});
test('AI cannot substitute malformed or missing probabilities', () => {
  const questions={risk:{}};
  for(const answers of [{},{risk:{type:'boolean',probability:NaN}},{risk:{type:'boolean',probability:2}},{risk:{type:'score',probability:.5}}]) assert.equal(validAnswers(answers,questions),false);
  assert.equal(validAnswers({risk:{type:'boolean',probability:.5}},questions),true);
});
