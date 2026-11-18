import React, { useState } from 'react';
import { API_BASE } from '../api';

/*
 * Specialist confirmation of a child's real outcome, per disorder.
 * These labels are the training data for the game's models; they never change the child's report.
 */
const DISORDERS = [['DYSLEXIA', 'Dyslexia'], ['DYSGRAPHIA', 'Dysgraphia'], ['DYSCALCULIA', 'Dyscalculia']];

export default function OutcomeLabels({ session, onSaved }) {
  const [labels, setLabels] = useState(session.labels || {});
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);

  const save = async (disorder, value) => {
    setBusy(disorder);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/assessments/${session.id}/labels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ disorder, value, source: 'specialist' }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `Server answered ${res.status}`);
      const saved = await res.json();
      setLabels(saved.labels || {});
      onSaved?.(saved);
    } catch (e) {
      setError(`Not saved: ${e.message}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div style={{ width: '100%', borderTop: '1px solid var(--border-faint)', paddingTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ fontSize: '0.76rem', color: 'var(--text-tertiary)' }}>
        Confirmed outcome (specialist): used to retrain the screening models
      </div>
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
        {DISORDERS.map(([key, label]) => {
          const cur = labels[key];
          const pill = (value, text) => {
            const on = cur && Number(cur.value) === value;
            return (
              <button
                type="button"
                disabled={busy === key}
                onClick={() => save(key, value)}
                aria-pressed={!!on}
                style={{
                  fontSize: '0.74rem', padding: '3px 9px', borderRadius: 999, cursor: 'pointer',
                  border: `1px solid ${on ? (value ? 'var(--risk-high-border)' : 'var(--risk-low-border)') : 'var(--border-subtle)'}`,
                  background: on ? (value ? 'var(--risk-high-dim)' : 'var(--risk-low-dim)') : 'transparent',
                  color: on ? (value ? 'var(--risk-high)' : 'var(--risk-low)') : 'var(--text-secondary)',
                }}
              >
                {text}
              </button>
            );
          };
          return (
            <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', minWidth: 74 }}>{label}</span>
              {pill(1, 'Has it')}
              {pill(0, 'Does not')}
              {cur && cur.source === 'parent' && <span style={{ fontSize: '0.68rem', color: 'var(--text-tertiary)' }}>(parent-reported)</span>}
            </div>
          );
        })}
      </div>
      {error && <div style={{ color: 'var(--risk-high)', fontSize: '0.76rem' }}>{error}</div>}
    </div>
  );
}
