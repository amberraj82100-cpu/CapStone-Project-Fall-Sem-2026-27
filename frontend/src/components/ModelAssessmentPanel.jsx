import React from 'react';
import { Cpu } from 'lucide-react';

/* Shows the ML service's per-domain results stored on the report (report.mlAssessment). */
const LABEL = { DYSLEXIA: 'Dyslexia', DYSGRAPHIA: 'Dysgraphia', DYSCALCULIA: 'Dyscalculia' };
const RISK = {
  LOW_RISK: ['Low risk', 'var(--risk-low)'],
  MODERATE_TENDENCY: ['Moderate tendency', 'var(--risk-mod)'],
  HIGH_RISK: ['High risk', 'var(--risk-high)'],
};

export default function ModelAssessmentPanel({ ml }) {
  if (!ml || !ml.domains) {
    return (
      <p style={{ color: 'var(--text-tertiary)', fontSize: '0.84rem', marginBottom: 40 }}>
        The screening models were not available for this session, so the results above use the rule-based scoring only.
      </p>
    );
  }
  return (
    <div style={{ marginBottom: 44 }}>
      <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
        <Cpu size={18} /> AI Model Assessment
      </h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginBottom: 16, maxWidth: '80ch' }}>
        Where a model trained on real data could read this session, its risk level is used in the domain cards above.
        Models trained on simulated data are shown for information only.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
        {Object.entries(ml.domains).map(([key, d]) => {
          const [riskText, riskColor] = RISK[d.risk_level] || ['—', 'var(--text-tertiary)'];
          const contrib = Object.entries(d.modality_contributions_logodds || {});
          return (
            <div key={key} style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-faint)', borderRadius: 'var(--r-lg)', padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ fontWeight: 700 }}>{LABEL[key] || key}</div>
              {d.status === 'ok' ? (
                <>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                    <span style={{ fontSize: '1.8rem', fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>{Math.round(d.probability * 100)}%</span>
                    <span style={{ color: riskColor, fontWeight: 700, fontSize: '0.86rem' }}>{riskText}</span>
                  </div>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)' }}>
                    Model {d.model_version}{d.synthetic_training_data ? ' · trained on simulated data (demo, not used in the report)' : ''}
                  </div>
                  {contrib.length > 0 && (
                    <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: 3 }}>
                      <span style={{ color: 'var(--text-tertiary)' }}>What drove it (positive raises risk):</span>
                      {contrib.map(([m, v]) => (
                        <span key={m} style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                          <span>{m}</span>
                          <span style={{ color: v > 0 ? 'var(--risk-high)' : 'var(--risk-low)', fontVariantNumeric: 'tabular-nums' }}>{v > 0 ? '+' : ''}{v.toFixed(2)}</span>
                        </span>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)' }}>{d.message || 'Not scored.'}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
