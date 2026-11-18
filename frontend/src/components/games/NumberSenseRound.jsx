import React, { useEffect, useMemo, useRef, useState } from 'react';

/*
 * Dyscalculia: a quick round of short trials of the game's five number skills.
 * Every trial is logged in the format the dyscalculia model reads (ml/src/dyscalculia_features.py):
 *   subitize   n_dots, response          magnitude  left, right
 *   numberline target, estimate, line_max sequence / arithmetic  correct + rt_ms only
 */
const DEFAULT_COUNTS = { subitize: 10, magnitude: 16, numberline: 10, sequence: 6, arithmetic: 10 };
const TASK_INTRO = {
  subitize: 'Stars will flash for a moment. How many did you see?',
  magnitude: 'Which side has MORE dots? Don’t count, just look.',
  numberline: 'Slide the marker to where the number belongs.',
  sequence: 'Which number fills the gap?',
  arithmetic: 'Add the numbers.',
};

const rnd = (a, b) => a + Math.floor(Math.random() * (b - a + 1));
const shuffle = (xs) => xs.map((x) => [Math.random(), x]).sort((p, q) => p[0] - q[0]).map((p) => p[1]);

function options(answer, spread = 3, min = 0) {
  const set = new Set([answer]);
  while (set.size < 4) {
    const o = answer + rnd(-spread, spread);
    if (o >= min && o !== answer) set.add(o);
  }
  return shuffle([...set]);
}

function makeTrials(counts) {
  const out = [];
  const sub = [1, 2, 3, 4, 5, 6, 7, 8, 9, 4].slice(0, counts.subitize);
  shuffle(sub).forEach((n) => out.push({ task: 'subitize', n_dots: n }));
  const ratios = [0.5, 0.6, 0.67, 0.75, 0.8, 0.86, 0.9, 0.92];
  for (let k = 0; k < counts.magnitude; k++) {
    const big = rnd(10, 24);
    const small = Math.max(1, Math.round(big * ratios[k % ratios.length]));
    out.push(Math.random() < 0.5 ? { task: 'magnitude', left: big, right: small } : { task: 'magnitude', left: small, right: big });
  }
  shuffle(Array.from({ length: 19 }, (_, i) => i + 1)).slice(0, counts.numberline)
    .forEach((x) => out.push({ task: 'numberline', target: x, line_max: 20 }));
  for (let k = 0; k < counts.sequence; k++) {
    const step = rnd(2, 5), start = rnd(1, 10), gap = rnd(1, 3);
    const seq = [0, 1, 2, 3, 4].map((i) => start + i * step);
    out.push({ task: 'sequence', seq, gap, answer: seq[gap], choices: options(seq[gap], step + 1, 1) });
  }
  for (let k = 0; k < counts.arithmetic; k++) {
    const a = rnd(1, 9), b = rnd(1, 9);
    out.push({ task: 'arithmetic', a, b, answer: a + b, choices: options(a + b, 3, 1) });
  }
  // keep each task's trials together so the child gets one short instruction per task
  const order = ['subitize', 'magnitude', 'numberline', 'sequence', 'arithmetic'];
  return order.flatMap((t) => out.filter((x) => x.task === t));
}

/* deterministic-looking scattered dots inside a box */
function Dots({ n, w = 220, h = 160, color = 'var(--dyscalculia)' }) {
  const pts = useMemo(() => {
    const ps = [];
    let guard = 0;
    while (ps.length < n && guard++ < 2000) {
      const p = { x: 16 + Math.random() * (w - 32), y: 16 + Math.random() * (h - 32), r: 7 + Math.random() * 5 };
      if (ps.every((q) => Math.hypot(q.x - p.x, q.y - p.y) > q.r + p.r + 4)) ps.push(p);
    }
    return ps;
  }, [n, w, h]);
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: 'var(--bg-elevated)', borderRadius: 12 }} aria-hidden="true">
      {pts.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r={p.r} fill={color} />)}
    </svg>
  );
}

export default function NumberSenseRound({ question, onAnswer }) {
  const counts = { ...DEFAULT_COUNTS, ...(question?.payload?.trialCounts || {}) };
  const flashMs = question?.payload?.flashMs || 1500;
  const trials = useMemo(() => makeTrials(counts), []); // eslint-disable-line react-hooks/exhaustive-deps
  const [i, setI] = useState(0);
  const [showDots, setShowDots] = useState(true);
  const [slider, setSlider] = useState(10);
  const log = useRef([]);
  const shownAt = useRef(performance.now());
  const tr = trials[i];
  const firstOfTask = i === 0 || trials[i - 1].task !== tr?.task;

  useEffect(() => {
    shownAt.current = performance.now();
    setSlider(10);
    if (tr?.task === 'subitize') {
      setShowDots(true);
      const h = setTimeout(() => setShowDots(false), flashMs);
      return () => clearTimeout(h);
    }
    return undefined;
  }, [i]); // eslint-disable-line react-hooks/exhaustive-deps

  const record = (fields) => {
    const rt = Math.round(performance.now() - shownAt.current);
    const { seq, gap, answer, choices, a, b, ...keep } = tr; // eslint-disable-line no-unused-vars
    log.current.push({ ...keep, trial: log.current.filter((x) => x.task === tr.task).length, rt_ms: rt, ...fields });
    if (i + 1 < trials.length) {
      setI(i + 1);
    } else {
      const all = log.current;
      const acc = all.filter((x) => x.correct).length / all.length;
      onAnswer({
        userAnswer: `${all.filter((x) => x.correct).length}/${all.length} trials correct`,
        correctAnswer: `${all.length} trials`,
        isCorrect: acc >= 0.7,
        rawData: { trials: all },
      });
    }
  };

  if (!tr) return null;
  const btn = { minWidth: 64, fontSize: '1.2rem', padding: '10px 18px' };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 18 }}>
      <div style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)' }}>Round {i + 1} / {trials.length}</div>
      <p style={{ margin: 0, fontWeight: firstOfTask ? 700 : 500, color: firstOfTask ? 'var(--text-primary)' : 'var(--text-secondary)', textAlign: 'center' }}>
        {TASK_INTRO[tr.task]}
      </p>

      {tr.task === 'subitize' && (
        <>
          {showDots ? <Dots key={i} n={tr.n_dots} /> : <div style={{ width: 220, height: 160, borderRadius: 12, background: 'var(--bg-elevated)' }} />}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center', maxWidth: 420 }}>
            {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
              <button key={n} type="button" className="btn-secondary" style={btn} disabled={showDots}
                onClick={() => record({ response: n, correct: n === tr.n_dots ? 1 : 0 })}>{n}</button>
            ))}
          </div>
        </>
      )}

      {tr.task === 'magnitude' && (
        <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', justifyContent: 'center' }}>
          {['left', 'right'].map((side) => (
            <button key={side} type="button" onClick={() => {
              const chosen = tr[side], other = tr[side === 'left' ? 'right' : 'left'];
              record({ correct: chosen > other ? 1 : 0 });
            }} style={{ background: 'none', padding: 0, cursor: 'pointer', borderRadius: 12 }} aria-label={`${side} side`}>
              <Dots key={`${i}-${side}`} n={tr[side]} />
            </button>
          ))}
        </div>
      )}

      {tr.task === 'numberline' && (
        <div style={{ width: '100%', maxWidth: 520, display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center' }}>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--dyscalculia)' }}>{tr.target}</div>
          <input type="range" min={0} max={20} step={0.1} value={slider} onChange={(e) => setSlider(Number(e.target.value))}
            style={{ width: '100%' }} aria-label="Number line from 0 to 20" />
          <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', color: 'var(--text-tertiary)' }}><span>0</span><span>20</span></div>
          <button type="button" className="btn-primary" onClick={() => record({ estimate: slider, correct: Math.abs(slider - tr.target) <= 1.5 ? 1 : 0 })}>
            Place it here
          </button>
        </div>
      )}

      {(tr.task === 'sequence' || tr.task === 'arithmetic') && (
        <>
          <div style={{ fontSize: '2rem', fontWeight: 800, letterSpacing: '0.04em' }}>
            {tr.task === 'sequence' ? tr.seq.map((v, k) => (k === tr.gap ? '__' : v)).join(',  ') : `${tr.a} + ${tr.b} = ?`}
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            {tr.choices.map((c) => (
              <button key={c} type="button" className="btn-secondary" style={btn} onClick={() => record({ correct: c === tr.answer ? 1 : 0 })}>{c}</button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
