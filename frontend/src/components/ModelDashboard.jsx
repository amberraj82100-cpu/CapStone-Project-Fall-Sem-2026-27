import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, AlertTriangle, CheckCircle2, FlaskConical, Layers, Loader2, RefreshCw } from 'lucide-react';

import { ML_API } from '../mlApi';
import RetrainingPanel from './RetrainingPanel';

/* Fixed family order and colours (validated categorical palette, dark-surface steps) */
const FAMILIES = ['Unimodal', 'Early fusion', 'Late fusion', 'Hybrid fusion'];
const FAMILY_COLOR = { Unimodal: '#3987e5', 'Early fusion': '#d95926', 'Late fusion': '#199e70', 'Hybrid fusion': '#c98500' };
const FAMILY_TEXT = {
  Unimodal: 'One model per modality, using only that evidence. The baseline: it tells us what each modality is worth on its own.',
  'Early fusion': 'All modalities’ features joined into one input, then one classifier. It can learn how modalities interact, such as slow answers together with many misses.',
  'Late fusion': 'One model per modality, then their risk scores are combined: a plain average, an average weighted by each modality’s accuracy, or a small model that learns the combination (stacking). Copes with a missing modality.',
  'Hybrid fusion': 'Stacking over the per-modality scores and the best early-fusion score. The most complex option.',
};
const EXPERIMENTS = [
  {
    id: 'dyslexia', title: 'Dyslexia', chip: 'Real · Dyt game data', kind: 'real',
    data: 'Gamified reading test, 5,039 children aged 7 to 17, 10.7% at risk. Modalities: demographics, task accuracy and interaction behaviour.',
    reading: 'Early fusion gained most, likely because one model can combine accuracy with clicking behaviour on the same question, while late fusion only sees each modality’s final score. Hybrid edged ahead by a margin well inside the noise, so the simpler early-fusion model is preferred.',
  },
  {
    id: 'dysgraphia', title: 'Dysgraphia', chip: 'Real · handwriting images', kind: 'real',
    data: '249 handwriting images, 46% potential dysgraphia. Modalities: 25 handwriting-geometry measures and deep visual features from a pretrained CNN (EfficientNet-B0).',
    reading: 'Both modalities look at the same handwriting, so they overlap. Joining them still added about 0.01, likely because the geometry measures (letter-size variability, spacing, stroke tremor) catch details the CNN averages away. Late fusion added almost nothing. The score may be inflated: the model leans on pen thickness, which may differ by how each group was recorded.',
  },
  {
    id: 'dyscalculia_castaldi2024', title: 'Dyscalculia', chip: 'Real · 37 children', kind: 'small',
    data: 'Castaldi et al. 2024: 37 children, 15 with dyscalculia. Modalities: time perception and cognitive profile (age, IQ, gender).',
    reading: 'Time perception carries the signal; age, IQ and gender add none. With so few children, late fusion’s combiner overfits and loses ground, so the time-perception model alone is the pick. This study used tasks the game does not have, so it is not deployed.',
  },
  {
    id: 'dyscalculia_synthetic', title: 'Dyscalculia', chip: 'Synthetic · pipeline test', kind: 'synthetic',
    data: '3,000 simulated children playing the game’s number tasks, 13% labelled at risk with 5% label noise. Modalities: demographics, task accuracy, response time and number-sense profile.',
    reading: 'Each modality is a noisy view of the same simulated abilities, so every fusion type gains about the same small amount. The explanations rank the planted effects correctly (arithmetic first, subitizing last). These scores say nothing about real children.',
  },
];
const EXP_LABEL = Object.fromEntries(EXPERIMENTS.map((e) => [e.id, e.kind === 'synthetic' ? 'Dyscalculia (synthetic)' : e.kind === 'small' ? 'Dyscalculia (Castaldi 2024)' : e.title]));
const FIGURE_TITLES = {
  'xai_shap_beeswarm.png': 'SHAP: how each feature pushes the risk',
  'xai_shap_geometry_beeswarm.png': 'SHAP: handwriting-geometry features',
  'xai_occlusion_maps.png': 'Where in the handwriting the CNN looks',
  'xai_question_importance.png': 'Which game questions matter most',
  'xai_shap_local_example.png': 'Why one child was flagged',
  'roc_best_per_family.png': 'ROC curves, best model per family',
  'confusion_best.png': 'Confusion matrix of the best model',
  'fusion_weights.png': 'Late-fusion weights per modality',
};

const f3 = (v) => (v == null || Number.isNaN(+v) ? '—' : (+v).toFixed(3));
const signed = (v) => (v == null ? '—' : `${+v >= 0 ? '+' : '−'}${Math.abs(+v).toFixed(3)}`);

/* ---------- helpers ---------- */
function useWidth() {
  const ref = useRef(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!ref.current) return undefined;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}
let _ctx;
function textWidth(t, px = 12) {
  _ctx = _ctx || document.createElement('canvas').getContext('2d');
  _ctx.font = `${px}px Lexend, Inter, sans-serif`;
  // Lexend renders wider than the canvas fallback may measure; pad so labels never clip
  return _ctx.measureText(t).width * 1.18 + 6;
}

/* Dot-and-whisker chart: one row per item, value ± sd, chance line at 0.5 */
function DotPlot({ rows, ariaLabel, highlight }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const W = Math.max(280, width || 600);
  const labW = Math.max(...rows.map((r) => textWidth(r.label) * (highlight === r.key ? 1.1 : 1) + (highlight === r.key ? 16 : 0)), 40);
  const stacked = labW > W * 0.42;
  const rowH = stacked ? 40 : 28;
  const top = 6, bottom = 28, valW = 46;
  const left = stacked ? 6 : labW + 14;
  const H = top + bottom + rows.length * rowH;
  const lo = Math.max(0, Math.floor((Math.min(0.5, ...rows.map((r) => r.v - (r.sd || 0))) - 0.02) * 10) / 10);
  const x = (v) => left + ((Math.min(1, Math.max(lo, v)) - lo) / (1 - lo)) * (W - valW - 8 - left);
  const ticks = [];
  for (let t = lo; t <= 1.0001; t += (1 - lo) > 0.5 ? 0.1 : 0.05) ticks.push(+t.toFixed(2));
  const y = (i) => top + i * rowH + (stacked ? 28 : rowH / 2);
  return (
    <div ref={ref} className="md-chart">
      {width > 0 && (
        <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={top} y2={H - bottom + 4} className="md-grid" />
              <text x={x(t)} y={H - 8} textAnchor="middle" className="md-tick">{t.toFixed(t * 10 % 1 ? 2 : 1)}</text>
            </g>
          ))}
          <line x1={x(0.5)} x2={x(0.5)} y1={top} y2={H - bottom + 4} className="md-ref" />
          {rows.map((r, i) => (
            <g key={r.key} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={0} y={top + i * rowH} width={W} height={rowH} fill={hover === i ? 'rgba(255,255,255,0.04)' : 'transparent'} />
              <text x={stacked ? left : left - 10} y={stacked ? y(i) - 15 : y(i)} dy="0.32em" textAnchor={stacked ? 'start' : 'end'}
                className="md-label" style={{ fontWeight: highlight === r.key ? 700 : 400 }}>
                {highlight === r.key ? '✓ ' : ''}{r.label}
              </text>
              <line x1={x(r.v - (r.sd || 0))} x2={x(r.v + (r.sd || 0))} y1={y(i)} y2={y(i)} className="md-whisker" />
              <circle cx={x(r.v)} cy={y(i)} r={highlight === r.key ? 7 : 5.5} fill={r.color} stroke="var(--bg-surface)" strokeWidth={2} />
              <text x={W - valW} y={y(i)} dy="0.32em" className="md-val">{f3(r.v)}</text>
            </g>
          ))}
        </svg>
      )}
      {hover != null && rows[hover] && (
        <div className="md-tip" style={{ top: y(hover) + 14, left: Math.min(Math.max(0, x(rows[hover].v) - 40), Math.max(0, W - 260)) }}>
          <div>{rows[hover].tip || rows[hover].label}</div>
          <span>{f3(rows[hover].v)} ± {f3(rows[hover].sd || 0)}</span>
          {rows[hover].extra && <span>{rows[hover].extra}</span>}
        </div>
      )}
    </div>
  );
}

/* Horizontal bars from zero */
function Bars({ rows, color, fmt = f3, ariaLabel }) {
  const [ref, width] = useWidth();
  const W = Math.max(240, width || 320);
  const labW = Math.max(...rows.map((r) => textWidth(r.label)), 30);
  const stacked = labW > W * 0.45;
  const rowH = stacked ? 36 : 24, left = stacked ? 0 : labW + 10, valW = 52;
  const H = rows.length * rowH + 4;
  const max = Math.max(1e-9, ...rows.map((r) => r.v));
  const len = (v) => Math.max(1, (Math.max(0, v) / max) * (W - left - valW - 6));
  return (
    <div ref={ref}>
      {width > 0 && (
        <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel}>
          {rows.map((r, i) => {
            const yb = i * rowH + (stacked ? 18 : 4);
            return (
              <g key={r.label}>
                <text x={stacked ? 0 : left - 8} y={stacked ? yb - 6 : yb + 8} dy="0.32em" textAnchor={stacked ? 'start' : 'end'} className="md-label">{r.label}</text>
                <rect x={left} y={yb} height={16} rx={3} width={len(r.v)} fill={color} />
                <text x={left + len(r.v) + 6} y={yb + 8} dy="0.32em" className="md-val">{fmt(r.v)}</text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

function Chip({ kind, children }) {
  return <span className={`md-chip ${kind || ''}`}>{children}</span>;
}

/* ---------- page ---------- */
export default function ModelDashboard() {
  const [data, setData] = useState(null);
  const [models, setModels] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [exp, setExp] = useState('dyslexia');
  const [metric, setMetric] = useState('roc_auc');

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${ML_API}/api/dashboard`);
      if (!res.ok) throw new Error(`ML service answered ${res.status}`);
      setData(await res.json());
      const m = await fetch(`${ML_API}/api/models`).catch(() => null);
      if (m && m.ok) setModels(await m.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => { load(); }, []);

  const t = data?.tables;
  const gameExps = (t?.experiments || []).filter((e) => e.data_kind === 'game').map((e) => ({
    id: e.experiment, title: e.label, chip: 'Game data', kind: 'game',
    data: `${e.n_samples} game sessions with a confirmed outcome, ${e.n_positive} at risk.`,
    reading: 'Trained automatically on the game’s own labelled sessions by the retraining job.',
  }));
  const allExps = [...EXPERIMENTS, ...gameExps];
  const expLabel = (id) => EXP_LABEL[id] || (gameExps.find((g) => g.id === id) || {}).title || id;
  const byExp = (name) => (t ? t[name].filter((r) => r.experiment === exp) : []);
  const fusionAhead = t ? t.experiments.filter((e) => +e.fusion_gain_auc > 0.005).length : null;
  const fusionDeployed = t ? t.deployment.filter((d) => !String(d.model).startsWith('Unimodal')).length : null;

  const exploreRows = useMemo(() => byExp('model_metrics')
    .map((r) => ({
      key: r.model, v: +r[metric], sd: +r[`${metric}_std`] || 0, color: FAMILY_COLOR[r.family],
      label: r.family === 'Unimodal' ? r.short_name : `${r.family.replace(' fusion', '')} · ${r.short_name}`,
      tip: `${r.family} · ${r.short_name}`, extra: `Sensitivity ${f3(r.sensitivity)} · specificity ${f3(r.specificity)}`,
    }))
    .sort((a, b) => b.v - a.v), [t, exp, metric]); // eslint-disable-line react-hooks/exhaustive-deps
  const chosen = t ? (t.selection.find((s) => s.experiment === exp) || {}).chosen_model : null;

  if (loading && !data) {
    return (
      <div className="flex-center" style={{ minHeight: '60vh', flexDirection: 'column', gap: 14 }}>
        <Loader2 size={36} color="var(--accent)" style={{ animation: 'spin 1.2s linear infinite' }} />
        <p style={{ color: 'var(--text-secondary)' }}>Loading model results…</p>
        <style>{'@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }'}</style>
      </div>
    );
  }
  if (error && !data) {
    return (
      <div className="md-page">
        <style>{CSS}</style>
        <div className="md-card md-error">
          <AlertTriangle size={20} color="var(--risk-high)" />
          <div>
            <h3>Can’t reach the ML service</h3>
            <p>{error}. Start it with <code>python -m uvicorn app:app --app-dir ml/service --port 8000</code> from the project folder, then retry.</p>
            <button className="btn-secondary" onClick={load}><RefreshCw size={14} /> Retry</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="md-page">
      <style>{CSS}</style>

      {/* Header */}
      <header className="md-head">
        <div className="md-head-row">
          <h1>Model Dashboard</h1>
          <button className="btn-secondary md-refresh" onClick={load} disabled={loading}>
            <RefreshCw size={14} /> {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
        <p className="md-lede">
          We screen for three learning disorders by combining several kinds of evidence from the same child.
          Fusion beat the best single kind of evidence by more than 0.005 ROC-AUC in <b>{fusionAhead}</b> of <b>{t.experiments.length}</b> experiments,
          and the game runs <b>{fusionDeployed}</b> fusion models out of <b>{t.deployment.length}</b>.
        </p>
        <p className="md-sub">
          All figures are held-out estimates from nested cross-validation: models, fusion weights and decision thresholds were chosen on training folds only.
          ROC-AUC is the chance that a child at risk scores higher than one who is not (0.5 is guessing). Balanced accuracy averages the share of at-risk
          children caught and the share of others cleared, so it isn’t inflated when one class is rare.
          {data.generated_at && <> Results last updated {new Date(data.generated_at).toLocaleString()}.</>}
        </p>
      </header>

      {/* Models in the game */}
      <section className="md-section" id="in-game">
        <h2><Activity size={18} /> Models running in the game</h2>
        <p className="md-note">The model each disorder uses right now. Within each experiment we deploy the simplest model that scored within 0.005 ROC-AUC of the best one; smaller gaps are inside the fold-to-fold noise.</p>
        <div className="md-grid3">
          {t.deployment.map((d) => (
            <article key={d.disorder} className="md-card">
              <div className="md-card-top">
                <h3>{d.disorder}</h3>
                <Chip kind={d.trained_on === 'synthetic' ? 'warn' : d.trained_on === 'rules' ? 'muted' : d.trained_on === 'game' ? 'game' : ''}>
                  {{ synthetic: 'Trained on synthetic data', rules: 'Rule-based for now', game: 'Trained on game data', real: 'Trained on research data' }[d.trained_on]}
                </Chip>
              </div>
              <p className="md-model">{d.model}{d.version ? ` · ${d.version}` : ''}</p>
              <div className="md-big"><span>{f3(d.roc_auc)}</span><small>ROC-AUC</small></div>
              <dl className="md-kv">
                <dt>Balanced accuracy</dt><dd>{f3(d.balanced_accuracy)}</dd>
                <dt>Sensitivity (at-risk caught)</dt><dd>{f3(d.sensitivity)}</dd>
                <dt>Specificity (others cleared)</dt><dd>{f3(d.specificity)}</dd>
                <dt>Moderate risk from p ≥</dt><dd>{f3(d.moderate_from)}</dd>
                <dt>High risk from p ≥</dt><dd>{f3(d.high_from)}</dd>
              </dl>
              <p className="md-note">Reads: {d.game_inputs}</p>
              <p className="md-reason"><b>Why this model: </b>{d.reason}</p>
              {d.disorder === 'Dysgraphia' && <p className="md-note warn">The score may be inflated: the model leans on pen thickness, which may differ by how each group was recorded.</p>}
              {d.trained_on === 'synthetic' && <p className="md-note warn">Trained on simulated children until the game has collected labelled real data. Treat its output as a demo.</p>}
            </article>
          ))}
        </div>
      </section>

      <RetrainingPanel models={models} />

      {/* What we tried */}
      <section className="md-section" id="approach">
        <h2><Layers size={18} /> What we tried</h2>
        <p className="md-note">Every disorder is observed through several kinds of evidence (modalities), such as what a child got right, how they clicked, or how their handwriting looks. We trained four families of models and compared them all on children the models had never seen.</p>
        <div className="md-fams">
          {FAMILIES.map((f) => (
            <div key={f} className="md-fam" style={{ borderTopColor: FAMILY_COLOR[f] }}>
              <h3><i style={{ background: FAMILY_COLOR[f] }} />{f}</h3>
              <p>{FAMILY_TEXT[f]}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Experiment by experiment */}
      <section className="md-section" id="stories">
        <h2><FlaskConical size={18} /> Experiment by experiment</h2>
        {allExps.map((e) => {
          const fam = Object.fromEntries(t.family_summary.filter((r) => r.experiment === e.id).map((r) => [r.family, r]));
          const sel = t.selection.find((s) => s.experiment === e.id) || {};
          if (!Object.keys(fam).length) return null;
          return (
            <article key={e.id} className="md-story">
              <div className="md-story-text">
                <h3>{e.title} <Chip kind={e.kind === 'synthetic' ? 'warn' : e.kind === 'small' ? 'muted' : e.kind === 'game' ? 'game' : ''}>{e.chip}</Chip></h3>
                <p className="md-note">{e.data}</p>
                <ul className="md-steps">
                  {FAMILIES.map((f) => fam[f] && (
                    <li key={f}>
                      <span><i style={{ background: FAMILY_COLOR[f] }} />{f === 'Unimodal' ? 'Best single' : f}</span>
                      <span>
                        {fam[f].best_short}: ROC-AUC <b>{f3(fam[f].roc_auc)}</b>
                        {f !== 'Unimodal' && <em className={+fam[f].gain_vs_best_unimodal > 0.005 ? 'good' : +fam[f].gain_vs_best_unimodal < -0.005 ? 'bad' : ''}> {signed(fam[f].gain_vs_best_unimodal)}</em>}
                      </span>
                    </li>
                  ))}
                </ul>
                <div className="md-pick">
                  <span><CheckCircle2 size={14} color="var(--risk-low)" /> <b>Chosen:</b> {sel.chosen_model}</span>
                  <span>{sel.reason}</span>
                </div>
                <p className="md-note"><b>Our reading of this run:</b> {e.reading}</p>
              </div>
              <DotPlot
                ariaLabel={`ROC-AUC of the best model in each family for ${expLabel(e.id)}`}
                highlight={sel.chosen_family}
                rows={FAMILIES.filter((f) => fam[f]).map((f) => ({
                  key: f, label: f, v: +fam[f].roc_auc, sd: +fam[f].roc_auc_std, color: FAMILY_COLOR[f],
                  tip: fam[f].best_model, extra: `${fam[f].n_models} model(s) tried · balanced acc. ${f3(fam[f].balanced_accuracy)}`,
                }))}
              />
            </article>
          );
        })}
      </section>

      {/* Explore */}
      <section className="md-section" id="explore">
        <h2>Explore every model</h2>
        <div className="md-controls">
          <label>Experiment
            <select value={exp} onChange={(e) => setExp(e.target.value)}>
              {allExps.map((e) => <option key={e.id} value={e.id}>{expLabel(e.id)}</option>)}
            </select>
          </label>
          <label>Metric
            <select value={metric} onChange={(e) => setMetric(e.target.value)}>
              <option value="roc_auc">ROC-AUC</option>
              <option value="balanced_accuracy">Balanced accuracy</option>
            </select>
          </label>
          <ul className="md-legend">
            {FAMILIES.map((f) => <li key={f}><i style={{ background: FAMILY_COLOR[f] }} />{f}</li>)}
          </ul>
        </div>
        <DotPlot rows={exploreRows} highlight={chosen} ariaLabel={`${metric === 'roc_auc' ? 'ROC-AUC' : 'Balanced accuracy'} of every model for ${expLabel(exp)}`} />
        <p className="md-note">Dots are the mean over outer test folds; whiskers span ± one standard deviation; the dashed line is chance (0.5). ✓ marks the model we picked.</p>

        <div className="md-grid3 md-xai">
          <div>
            <h3>Which modality matters</h3>
            <p className="md-note">Measured as {(byExp('modality_importance')[0] || {}).importance_kind}.</p>
            <Bars color="#3987e5" ariaLabel="Modality importance"
              rows={byExp('modality_importance').map((r) => ({ label: r.modality, v: +r.importance })).sort((a, b) => b.v - a.v)} />
          </div>
          <div>
            <h3>How late fusion weights each modality</h3>
            <p className="md-note">AUC-weighted voting weight, averaged over folds.</p>
            <Bars color="#199e70" fmt={(v) => (+v).toFixed(2)} ariaLabel="Late-fusion weights"
              rows={byExp('modality_importance').map((r) => ({ label: r.modality, v: +r.late_fusion_weight })).sort((a, b) => b.v - a.v)} />
          </div>
          <div>
            <h3>Top features by mean |SHAP|</h3>
            <p className="md-note">{exp === 'dysgraphia' ? 'Handwriting-geometry features only; the 1,280 CNN dimensions have no individual meaning.' : 'Average size of each feature’s push on the predicted risk.'}</p>
            <Bars color="#3987e5" ariaLabel="Top features"
              rows={byExp('top_features').map((r) => ({ label: r.feature, v: +r.mean_abs_shap })).sort((a, b) => b.v - a.v)} />
          </div>
        </div>

        {(data.figures[exp] || []).length > 0 && (
          <div className="md-figs">
            {data.figures[exp].map((name) => (
              <figure key={name}>
                <img src={`${ML_API}/api/figures/${exp}/${name}`} alt={FIGURE_TITLES[name] || name} loading="lazy" />
                <figcaption>{FIGURE_TITLES[name] || name}</figcaption>
              </figure>
            ))}
          </div>
        )}

        <h3 className="md-h3">Full metrics</h3>
        <div className="md-tablewrap">
          <table className="md-table">
            <thead>
              <tr><th>Family</th><th>Model</th><th>ROC-AUC</th><th>sd</th><th>Balanced acc.</th><th>sd</th><th>Sensitivity</th><th>Specificity</th><th>F1</th><th>Accuracy</th></tr>
            </thead>
            <tbody>
              {byExp('model_metrics').sort((a, b) => b.roc_auc - a.roc_auc).map((r) => (
                <tr key={r.model} className={r.model === chosen ? 'picked' : ''}>
                  <td><i style={{ background: FAMILY_COLOR[r.family] }} />{r.family}</td><td>{r.short_name}</td>
                  <td>{f3(r.roc_auc)}</td><td>{f3(r.roc_auc_std)}</td><td>{f3(r.balanced_accuracy)}</td><td>{f3(r.balanced_accuracy_std)}</td>
                  <td>{f3(r.sensitivity)}</td><td>{f3(r.specificity)}</td><td>{f3(r.f1)}</td><td>{f3(r.accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="md-note">The highlighted row is the model we picked. Plain accuracy is shown for completeness only: for dyslexia, always answering “no risk” already scores 0.893.</p>
      </section>

      <details className="md-defs">
        <summary>Definitions and data sources</summary>
        <dl>
          <dt>Sensitivity</dt><dd>Share of at-risk children the model flags.</dd>
          <dt>Specificity</dt><dd>Share of children not at risk that the model clears.</dd>
          <dt>Nested cross-validation</dt><dd>The data is split into outer folds; inside each training part a second split chooses the algorithms, fusion weights and thresholds. Only the outer test folds are scored.</dd>
          <dt>Dyslexia data</dt><dd>Dyt-desktop and Dyt-tablet gamified-test datasets (5,039 children, 32 questions).</dd>
          <dt>Dysgraphia data</dt><dd>249 handwriting images, potential vs low-potential dysgraphia.</dd>
          <dt>Dyscalculia data</dt><dd>Castaldi et al. 2024 (Zenodo 10966052, CC BY 4.0), and a simulated set generated from the game’s number tasks.</dd>
        </dl>
      </details>
    </div>
  );
}

const CSS = `
.md-page { max-width: 1180px; margin: 0 auto; padding: 32px max(16px, 3vw) 64px; display: flex; flex-direction: column; gap: 40px; color: var(--text-primary); }
.md-head { display: flex; flex-direction: column; gap: 10px; }
.md-head-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
.md-head h1 { margin: 0; font-size: clamp(1.6rem, 3vw, 2.2rem); letter-spacing: -0.03em; }
.md-refresh { font-size: 0.84rem; padding: 7px 14px; }
.md-lede { margin: 0; max-width: 78ch; font-size: 1.02rem; line-height: 1.6; color: var(--text-primary); }
.md-lede b { color: var(--accent); font-variant-numeric: tabular-nums; }
.md-sub, .md-note { margin: 0; color: var(--text-secondary); font-size: 0.82rem; line-height: 1.55; max-width: 90ch; }
.md-note.warn { color: var(--risk-mod); }
.md-section { display: flex; flex-direction: column; gap: 14px; }
.md-section > h2 { margin: 0; font-size: 1.2rem; display: flex; align-items: center; gap: 8px; }
.md-h3 { margin: 12px 0 0; font-size: 1rem; }
.md-grid3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
.md-grid3 > div { min-width: 0; display: flex; flex-direction: column; gap: 8px; }
.md-grid3 h3 { margin: 0; font-size: 0.95rem; }
.md-card { background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 20px; display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.md-card-top { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.md-card h3 { margin: 0; font-size: 1.1rem; }
.md-error { flex-direction: row; gap: 14px; align-items: flex-start; }
.md-error p { color: var(--text-secondary); font-size: 0.9rem; }
.md-chip { font-size: 0.72rem; padding: 2px 9px; border-radius: 999px; border: 1px solid var(--border-medium); color: var(--text-secondary); white-space: nowrap; font-weight: 600; }
.md-chip.warn { border-color: var(--risk-mod-border); color: var(--risk-mod); background: var(--risk-mod-dim); }
.md-chip.game { border-color: var(--dysgraphia-border); color: var(--dysgraphia); background: var(--dysgraphia-dim); }
.md-chip.muted { border-color: var(--border-medium); color: var(--text-tertiary); }
.md-model { margin: 0; color: var(--text-secondary); font-size: 0.82rem; overflow-wrap: anywhere; }
.md-big { display: flex; align-items: baseline; gap: 8px; }
.md-big span { font-size: 2.4rem; font-weight: 700; letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }
.md-big small { color: var(--text-tertiary); font-size: 0.8rem; }
.md-kv { display: grid; grid-template-columns: 1fr auto; gap: 4px 14px; margin: 0; font-size: 0.82rem; }
.md-kv dt { color: var(--text-secondary); }
.md-kv dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; }
.md-reason { margin: 0; font-size: 0.82rem; line-height: 1.5; }
.md-fams { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
.md-fam { border-top: 3px solid; padding-top: 10px; display: flex; flex-direction: column; gap: 6px; }
.md-fam h3 { margin: 0; font-size: 0.95rem; display: flex; align-items: center; gap: 8px; }
.md-fam p { margin: 0; font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5; }
.md-page i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; flex-shrink: 0; }
.md-story { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 28px; padding: 22px 0; border-top: 1px solid var(--border-faint); }
@media (max-width: 820px) { .md-story { grid-template-columns: minmax(0, 1fr); } }
.md-story-text { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.md-story h3 { margin: 0; font-size: 1.1rem; display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.md-steps { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 6px; font-size: 0.84rem; }
.md-steps li { display: grid; grid-template-columns: 8.5em 1fr; gap: 10px; }
.md-steps li > span:first-child { color: var(--text-secondary); display: flex; align-items: center; }
.md-steps b { font-variant-numeric: tabular-nums; }
.md-steps em { font-style: normal; color: var(--text-tertiary); font-variant-numeric: tabular-nums; }
.md-steps em.good { color: var(--risk-low); }
.md-steps em.bad { color: var(--risk-high); }
.md-pick { background: var(--glass-2); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 12px 14px; font-size: 0.84rem; line-height: 1.5; display: flex; flex-direction: column; gap: 4px; }
.md-pick span:first-child { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.md-chart { position: relative; min-height: 120px; }
.md-chart svg, .md-grid3 svg { display: block; }
.md-grid { stroke: var(--border-faint); }
.md-ref { stroke: var(--text-tertiary); stroke-dasharray: 4 3; }
.md-whisker { stroke: var(--text-secondary); stroke-width: 1.5; }
.md-tick { fill: var(--text-tertiary); font-size: 11px; }
.md-label { fill: var(--text-secondary); font-size: 12px; }
.md-val { fill: var(--text-primary); font-size: 12px; font-variant-numeric: tabular-nums; }
.md-tip { position: absolute; pointer-events: none; background: var(--bg-overlay); border: 1px solid var(--border-medium); box-shadow: var(--shadow-md); border-radius: 8px; padding: 8px 12px; font-size: 0.78rem; max-width: 260px; display: flex; flex-direction: column; gap: 2px; z-index: 5; }
.md-tip span { color: var(--text-secondary); }
.md-controls { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; }
.md-controls label { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: var(--text-secondary); }
.md-controls select { background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-medium); border-radius: 8px; padding: 6px 10px; font: inherit; }
.md-legend { display: flex; flex-wrap: wrap; gap: 14px; list-style: none; margin: 0; padding: 0; font-size: 0.78rem; color: var(--text-secondary); }
.md-legend li { display: flex; align-items: center; }
.md-xai { margin-top: 12px; }
.md-figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; align-items: start; }
.md-figs figure { margin: 0; background: #fcfcfb; border-radius: 10px; padding: 10px; display: flex; flex-direction: column; gap: 6px; }
.md-figs img { width: 100%; height: auto; display: block; }
.md-figs figcaption { color: #52514e; font-size: 0.78rem; }
.md-tablewrap { overflow-x: auto; }
.md-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; font-variant-numeric: tabular-nums; }
.md-table th, .md-table td { padding: 7px 10px; border-bottom: 1px solid var(--border-faint); text-align: right; white-space: nowrap; }
.md-table th:nth-child(-n+2), .md-table td:nth-child(-n+2) { text-align: left; }
.md-table th { color: var(--text-tertiary); font-weight: 600; }
.md-compact th, .md-compact td { white-space: normal; padding: 6px 6px; }
.md-table tr.picked td { background: var(--accent-subtle); font-weight: 600; }
.md-defs { font-size: 0.84rem; color: var(--text-secondary); }
.md-defs summary { cursor: pointer; color: var(--text-primary); }
.md-defs dl { display: grid; grid-template-columns: minmax(140px, max-content) 1fr; gap: 6px 16px; }
.md-defs dd { margin: 0; line-height: 1.5; }
`;
