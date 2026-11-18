import React from 'react';
import { RotateCcw } from 'lucide-react';

/* Registry + retraining history from the ML service (/api/models). */
const NAMES = { DYSLEXIA: 'Dyslexia', DYSGRAPHIA: 'Dysgraphia', DYSCALCULIA: 'Dyscalculia' };
const STATUS = {
  promoted: ['New model promoted', 'var(--risk-low)'],
  kept_champion: ['Kept the live model', 'var(--text-secondary)'],
  waiting_for_data: ['Waiting for labelled sessions', 'var(--risk-mod)'],
};
const f3 = (v) => (v == null ? '—' : (+v).toFixed(3));
const when = (iso) => (iso ? new Date(iso).toLocaleString() : '—');

export default function RetrainingPanel({ models }) {
  if (!models) return null;
  const { registry, history = [], policy = {} } = models;
  const last = history[history.length - 1];
  return (
    <section className="md-section" id="retraining">
      <h2><RotateCcw size={18} /> Game data and retraining</h2>
      <p className="md-note">
        When a specialist confirms a child’s outcome, that session becomes training data. After every {policy.retrain_every_new_labels} new
        confirmed outcomes for a disorder, the same fusion pipeline retrains on the game’s own data. The new model replaces the live one
        only if it scores at least {policy.promote_margin} ROC-AUC higher on children the live model never saw, and at least {policy.min_auc_to_deploy} overall.
        Training starts once a disorder has {policy.min_labelled} confirmed sessions with {policy.min_per_class} or more in each group.
      </p>
      <div className="md-grid3">
        {Object.entries(registry).map(([d, entry]) => {
          const r = last?.results?.[d];
          const [st, col] = STATUS[r?.status] || ['Not run yet', 'var(--text-tertiary)'];
          return (
            <article key={d} className="md-card">
              <div className="md-card-top">
                <h3>{NAMES[d]}</h3>
                <span className="md-chip">{entry.active ? `Live: ${entry.active}` : 'Live: rules'}</span>
              </div>
              <dl className="md-kv">
                <dt>Last retraining</dt><dd>{when(r?.run_at)}</dd>
                <dt>Outcome</dt><dd style={{ color: col }}>{st}</dd>
                <dt>Confirmed sessions</dt><dd>{r ? `${r.labelled_sessions} (${r.at_risk} at risk)` : '—'}</dd>
                {r?.challenger_roc_auc != null && <><dt>New model ROC-AUC</dt><dd>{f3(r.challenger_roc_auc)}</dd></>}
                {r?.champion_roc_auc != null && <><dt>Live model ROC-AUC</dt><dd>{f3(r.champion_roc_auc)}</dd></>}
              </dl>
              {r?.champion_score_basis && <p className="md-note">Live model scored on {r.champion_score_basis}.</p>}
              {r?.reason && <p className="md-note">{r.reason}</p>}
              {r?.message && <p className="md-note">{r.message}</p>}
              <div className="md-tablewrap">
                <table className="md-table md-compact">
                  <thead><tr><th>Version</th><th>Data</th><th>ROC-AUC</th></tr></thead>
                  <tbody>
                    {entry.versions.slice().reverse().map((v) => (
                      <tr key={v.id} className={v.id === entry.active ? 'picked' : ''}>
                        <td title={`Created ${new Date(v.created_at).toLocaleString()}`}>{v.id.replace(/^[a-z]+-/, '')}</td>
                        <td>{v.kind === 'game' ? `game, ${v.n_train} sessions` : v.synthetic ? 'synthetic' : 'research'}</td>
                        <td>{f3(v.metrics?.roc_auc)}</td>
                      </tr>
                    ))}
                    {entry.versions.length === 0 && <tr><td colSpan={3}>No trained model yet; rule-based scoring is live.</td></tr>}
                  </tbody>
                </table>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
