# Code Jev Analyzer 🛡️

A Next.js 14+ (App Router) server-side service deployed on Vercel that evaluates code files and pull request diffs using **TypeSafe AI's Jev** model via **Vercel AI Gateway**.

All inference runs exclusively server-side on Vercel using Vercel OIDC authentication (`VERCEL_OIDC_TOKEN`) with zero data retention.

---

## Table of Contents

- [Overview](#overview)
- [Why Jev Returns Typed Answers Instead of Prose](#why-jev-returns-typed-answers-instead-of-prose)
- [Evaluation Questions & Atomicity](#evaluation-questions--atomicity)
- [How Thresholds Work](#how-thresholds-work)
- [Adding a New Question](#adding-a-new-question)
- [Vercel AI Gateway & Pricing](#vercel-ai-gateway--pricing)
- [Authentication & Environments](#authentication--environments)
- [API Reference](#api-reference)
- [GitHub Actions PR Integration](#github-actions-pr-integration)
- [Development & Testing](#development--testing)

---

## Overview

Traditional LLM code reviews return conversational prose that is difficult for automated CI/CD pipelines to parse reliably. `code-jev-analyzer` utilizes the Vercel AI SDK's `experimental_evaluate` function with TypeSafe AI's **Jev** evaluation model (`typesafe-ai/jev`) to inspect code against precise, atomic questions, returning mathematical probability distributions and calibrated rubric scores.

---

## Why Jev Returns Typed Answers Instead of Prose

1. **Deterministic & Machine-Readable**: Jev answers specific typed questions (boolean or score rubric) with calibrated probabilities ($P(\text{true}) \in [0, 1]$), eliminating markdown formatting variations, conversational preamble, and hallucinated commentary.
2. **Zero Regex/Scraping Needed**: Downstream tools and CI workflows can directly branch on numerical thresholds rather than relying on brittle string matching or regex parsers.
3. **Atomic Evaluation**: Instead of asking one broad "Review this code" prompt, Jev evaluates independent questions in isolation. This prevents cognitive dilution and ensures each potential vulnerability is inspected systematically.
4. **Calibrated Confidence**: Rather than a vague "looks like there might be an issue", Jev provides exact probabilities, allowing teams to tune precision vs. recall.

---

## Evaluation Questions & Atomicity

Each question in `lib/analyze-code.ts` focuses strictly on a **single concern**:

| Question Key | Type | Description |
| :--- | :--- | :--- |
| `hardcodedSecret` | `boolean` | Hardcoded credential, API key, or secret present |
| `unhandledError` | `boolean` | An error path (network call, parse, DB query) exists with no handling |
| `missingInputValidation` | `boolean` | External input is used without validation or sanitization |
| `sqlInjectionRisk` | `boolean` | SQL is built via string concatenation or interpolation instead of parameterized queries |
| `riskLevel` | `score` (4-level) | `0: trivial/cosmetic` → `1: minor` → `2: notable (could break a feature)` → `3: severe (could break prod or leak data)` |

---

## How Thresholds Work

The `analyzeCode` helper maps the model's raw probability values to actionable statuses:

- **Flagged (`flagged: true`)**: Probability $\ge 0.8$
  - High confidence vulnerability or defect. Recommended to block merge or require immediate remediation.
- **Review (`flagged: 'review'`)**: $0.4 \le \text{Probability} < 0.8$
  - Borderline or ambiguous pattern. Routes to a human reviewer for triage.
- **Clean (`flagged: false`)**: Probability $< 0.4$
  - Below threshold, considered benign. Excluded from the issue summary.

---

## Adding a New Question

Questions must remain **atomic** (one concern per question). To add a new check:

1. Open `lib/analyze-code.ts`.
2. Add your question to the `QUESTIONS` object:

```typescript
export const QUESTIONS = {
  // ... existing questions ...
  insecureDependency: {
    type: 'boolean',
    instructions: 'The code imports or references a known insecure or deprecated package.',
  },
} as const;
```

3. Update the `BooleanQuestionKey` union type if it's a boolean check:

```typescript
export type BooleanQuestionKey =
  | 'hardcodedSecret'
  | 'unhandledError'
  | 'missingInputValidation'
  | 'sqlInjectionRisk'
  | 'insecureDependency';
```

4. The threshold logic in `analyzeCode` will automatically evaluate and flag issues for the new check.

---

## Vercel AI Gateway & Pricing

- **Zero Data Retention**: All calls specify `providerOptions: { gateway: { zeroDataRetention: true } }`, guaranteeing your code is not stored or trained on.
- **Cost Coverage**: Usage is covered by Vercel AI Gateway's **$5/month free credit**, which is sufficient for thousands of daily file evaluations. No credit card is required to get started unless volume exceeds the free tier.

---

## Authentication & Environments

> [!NOTE]
> Deployed instances on Vercel receive `VERCEL_OIDC_TOKEN` **automatically** from the Vercel runtime. You do not need to configure API keys or secrets manually in production.

### Local Development / One-off Testing Only

In production, all inference runs server-side on Vercel. For one-off local testing during development:

1. Link your local project:
   ```bash
   vercel link
   ```
2. Pull development environment variables:
   ```bash
   vercel env pull
   ```
   This downloads a temporary `VERCEL_OIDC_TOKEN` into `.env.local`.

---

## API Reference

### `POST /api/analyze-code`

#### Request Body
```json
{
  "code": "const key = 'sk_live_123456789';",
  "filename": "config.ts"
}
```

#### Response
```json
{
  "answers": {
    "hardcodedSecret": { "type": "boolean", "probability": 0.96 },
    "unhandledError": { "type": "boolean", "probability": 0.05 },
    "missingInputValidation": { "type": "boolean", "probability": 0.12 },
    "sqlInjectionRisk": { "type": "boolean", "probability": 0.01 },
    "riskLevel": { "type": "score", "score": 3 }
  },
  "summary": [
    {
      "question": "hardcodedSecret",
      "probability": 0.96,
      "flagged": true,
      "description": "A hardcoded credential, API key, or secret is present in the code."
    }
  ]
}
```

---

## GitHub Actions PR Integration

A GitHub Actions workflow is provided at `.github/workflows/analyze.yml`.

### Setup
1. In your GitHub repository settings, go to **Settings > Secrets and variables > Actions**.
2. Add a new repository secret:
   - **Name**: `ANALYZER_URL`
   - **Value**: `https://<your-project>.vercel.app`
3. On every Pull Request, the workflow diffs changed files, submits each file to the deployed Vercel endpoint, and posts an interactive summary comment.

---

## Development & Testing

Unit tests run via Vitest using the mock evaluation model `Experimental_EvaluationMockModelV4`:

```bash
# Run unit tests
npm test

# Run build
npm run build
```
