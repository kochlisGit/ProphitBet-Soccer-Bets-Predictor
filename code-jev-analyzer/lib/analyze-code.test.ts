import { describe, it, expect } from 'vitest';
import { Experimental_EvaluationMockModelV4 } from 'ai/test';
import { analyzeCode, getFlagStatus } from './analyze-code';

describe('getFlagStatus threshold logic', () => {
  it('returns false for probabilities below 0.4', () => {
    expect(getFlagStatus(0.0)).toBe(false);
    expect(getFlagStatus(0.2)).toBe(false);
    expect(getFlagStatus(0.39)).toBe(false);
  });

  it('returns "review" for probabilities between 0.4 and 0.8 (inclusive of 0.4, exclusive of 0.8)', () => {
    expect(getFlagStatus(0.4)).toBe('review');
    expect(getFlagStatus(0.65)).toBe('review');
    expect(getFlagStatus(0.799)).toBe('review');
  });

  it('returns true for probabilities >= 0.8', () => {
    expect(getFlagStatus(0.8)).toBe(true);
    expect(getFlagStatus(0.95)).toBe(true);
    expect(getFlagStatus(1.0)).toBe(true);
  });
});

describe('analyzeCode with Experimental_EvaluationMockModelV4', () => {
  it('correctly handles a clearly clean file', async () => {
    const cleanMockModel = new Experimental_EvaluationMockModelV4({
      provider: 'mock-provider',
      modelId: 'mock-jev',
      supportedQuestionTypes: ['boolean', 'score'],
      doEvaluate: async () => ({
        answers: {
          hardcodedSecret: { type: 'boolean', probability: 0.02 },
          unhandledError: { type: 'boolean', probability: 0.1 },
          missingInputValidation: { type: 'boolean', probability: 0.05 },
          sqlInjectionRisk: { type: 'boolean', probability: 0.01 },
          leagueIsolationRisk: { type: 'boolean', probability: 0.1 },
          unverifiedDataRisk: { type: 'boolean', probability: 0.1 },
          riskLevel: { type: 'score', score: 0 },
        },
        warnings: [],
      }),
    });

    const code = `
      export function add(a: number, b: number): number {
        return a + b;
      }
    `;

    const result = await analyzeCode(code, 'math.ts', cleanMockModel);

    // Raw answers check
    expect(result.answers.hardcodedSecret.probability).toBe(0.02);
    expect(result.answers.unhandledError.probability).toBe(0.1);
    expect(result.answers.missingInputValidation.probability).toBe(0.05);
    expect(result.answers.sqlInjectionRisk.probability).toBe(0.01);
    expect(result.answers.riskLevel.score).toBe(0);

    // Summary should be empty because no question met or exceeded 0.4 threshold
    expect(result.summary).toHaveLength(0);
  });

  it('correctly handles a clearly flagged file with probability >= 0.8', async () => {
    const flaggedMockModel = new Experimental_EvaluationMockModelV4({
      provider: 'mock-provider',
      modelId: 'mock-jev',
      supportedQuestionTypes: ['boolean', 'score'],
      doEvaluate: async () => ({
        answers: {
          hardcodedSecret: { type: 'boolean', probability: 0.96 },
          unhandledError: { type: 'boolean', probability: 0.15 },
          missingInputValidation: { type: 'boolean', probability: 0.3 },
          sqlInjectionRisk: { type: 'boolean', probability: 0.88 },
          leagueIsolationRisk: { type: 'boolean', probability: 0.1 },
          unverifiedDataRisk: { type: 'boolean', probability: 0.1 },
          riskLevel: { type: 'score', score: 3 },
        },
        warnings: [],
      }),
    });

    const code = `
      const DB_PASSWORD = "super_secret_password_123";
      export async function getUser(req: Request) {
        const id = new URL(req.url).searchParams.get("id");
        return db.query("SELECT * FROM users WHERE id = " + id);
      }
    `;

    const result = await analyzeCode(code, 'user.ts', flaggedMockModel);

    // Both hardcodedSecret and sqlInjectionRisk should be flagged: true
    expect(result.summary).toHaveLength(2);

    const secretIssue = result.summary.find((i) => i.question === 'hardcodedSecret');
    expect(secretIssue).toBeDefined();
    expect(secretIssue?.flagged).toBe(true);
    expect(secretIssue?.probability).toBe(0.96);

    const sqlIssue = result.summary.find((i) => i.question === 'sqlInjectionRisk');
    expect(sqlIssue).toBeDefined();
    expect(sqlIssue?.flagged).toBe(true);
    expect(sqlIssue?.probability).toBe(0.88);

    expect(result.answers.riskLevel.score).toBe(3);
  });

  it('correctly handles a borderline file that routes to review (0.4 <= p < 0.8)', async () => {
    const borderlineMockModel = new Experimental_EvaluationMockModelV4({
      provider: 'mock-provider',
      modelId: 'mock-jev',
      supportedQuestionTypes: ['boolean', 'score'],
      doEvaluate: async () => ({
        answers: {
          hardcodedSecret: { type: 'boolean', probability: 0.12 },
          unhandledError: { type: 'boolean', probability: 0.62 },
          missingInputValidation: { type: 'boolean', probability: 0.5 },
          sqlInjectionRisk: { type: 'boolean', probability: 0.25 },
          leagueIsolationRisk: { type: 'boolean', probability: 0.1 },
          unverifiedDataRisk: { type: 'boolean', probability: 0.1 },
          riskLevel: { type: 'score', score: 2 },
        },
        warnings: [],
      }),
    });

    const code = `
      export async function fetchUserData(input: any) {
        const res = await fetch('/api/user/' + input.id);
        const data = await res.json();
        return data;
      }
    `;

    const result = await analyzeCode(code, 'api.ts', borderlineMockModel);

    // Both unhandledError and missingInputValidation should have flagged: 'review'
    expect(result.summary).toHaveLength(2);

    const errorIssue = result.summary.find((i) => i.question === 'unhandledError');
    expect(errorIssue).toBeDefined();
    expect(errorIssue?.flagged).toBe('review');
    expect(errorIssue?.probability).toBe(0.62);

    const validationIssue = result.summary.find((i) => i.question === 'missingInputValidation');
    expect(validationIssue).toBeDefined();
    expect(validationIssue?.flagged).toBe('review');
    expect(validationIssue?.probability).toBe(0.5);

    expect(result.answers.riskLevel.score).toBe(2);
  });
});
