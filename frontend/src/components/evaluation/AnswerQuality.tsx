import { Scale } from 'lucide-react'
import type { AnswerSummary, EvaluationRunSummary } from '../../types/api'
import { formatPercent } from '../../utils/evaluation'
import { Badge } from '../ui/Badge'
import { Card, CardHeader } from '../ui/Card'

const SCORES: { key: keyof AnswerSummary; label: string; description: string }[] = [
  {
    key: 'faithfulness',
    label: 'Faithfulness',
    description: 'Statements in the answers that the retrieved passages support.',
  },
  {
    key: 'answer_relevance',
    label: 'Answer relevance',
    description: 'How directly the answers address their questions.',
  },
  {
    key: 'context_relevance',
    label: 'Context relevance',
    description: 'Passages given to the model that help answer the question.',
  },
  {
    key: 'correctness',
    label: 'Correctness',
    description: 'Agreement with the reference answers written for the dataset.',
  },
]

function Tile({ label, value, description }: { label: string; value: string; description: string }) {
  return (
    <div className="rounded-lg border border-border px-4 py-3">
      <p className="text-xs text-fg-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-fg">{value}</p>
      <p className="mt-1 text-xs text-fg-subtle">{description}</p>
    </div>
  )
}

/** LLM-judged answer scores, kept visibly apart from the retrieval metrics. */
export function AnswerQuality({ run }: { run: EvaluationRunSummary }) {
  const answers = run.metrics.answers
  const judge = answers?.models?.judge

  return (
    <Card>
      <CardHeader
        title="Answer quality"
        description="The full pipeline answered every question; a language model then judged the answers."
        action={<Badge tone="warning">LLM-judged</Badge>}
      />
      <div className="space-y-4 p-5">
        {!run.config.answer_quality || !answers ? (
          <p className="flex gap-2 text-sm text-fg-muted">
            <Scale className="mt-0.5 size-4 shrink-0 text-fg-subtle" aria-hidden />
            Not measured in this run. Tick “Judge answers” before running to generate an answer
            for every question and score it (about two model calls per question; needs a Gemini
            API key).
          </p>
        ) : (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {SCORES.map((score) => (
                <Tile
                  key={score.key}
                  label={score.label}
                  value={formatPercent(answers.summary[score.key] as number | null)}
                  description={score.description}
                />
              ))}
            </div>
            <div className="rounded-lg bg-surface-muted/60 px-4 py-3 text-sm text-fg">
              <p>
                <span className="font-semibold">
                  {formatPercent(answers.summary.refusal_accuracy)}
                </span>{' '}
                of questions handled correctly: answered when the papers cover them, refused when
                they don’t.
              </p>
              <p className="mt-1 text-xs text-fg-muted">
                Answerable questions answered: {formatPercent(answers.summary.answerable_answered)}{' '}
                · Unanswerable questions refused:{' '}
                {formatPercent(answers.summary.unanswerable_refused)} · Checked in code, not by
                the model.
              </p>
            </div>
            <p className="text-xs text-fg-subtle">
              Scores are averaged over the {answers.summary.answered} answers given
              {answers.stopped_after !== undefined &&
                ` (the run stopped after ${answers.stopped_after} questions)`}
              . The judge{judge ? ` (${judge})` : ''} is a language model, usually the one that
              wrote the answers, so treat these as indicative rather than ground truth.
            </p>
          </>
        )}
      </div>
    </Card>
  )
}
