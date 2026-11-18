import React, { useEffect, useRef, useState } from 'react';
import { Eraser, Check } from 'lucide-react';

/*
 * Dysgraphia: copy a short sentence by hand.
 * Sends the drawing as a PNG (dark ink on white, like a scanned page) for the handwriting model,
 * plus real stroke measures: jitter = mean distance of points from their locally smoothed path,
 * hesitations = pauses over 400 ms with the pen down.
 */
const W = 900;
const H = 260;

function strokeJitter(strokes) {
  let sum = 0, n = 0;
  strokes.forEach((s) => {
    for (let i = 2; i < s.length - 2; i++) {
      const mx = (s[i - 2].x + s[i - 1].x + s[i + 1].x + s[i + 2].x) / 4;
      const my = (s[i - 2].y + s[i - 1].y + s[i + 1].y + s[i + 2].y) / 4;
      sum += Math.hypot(s[i].x - mx, s[i].y - my);
      n++;
    }
  });
  return n ? sum / n : 0;
}

function hesitations(strokes) {
  let count = 0;
  strokes.forEach((s) => {
    for (let i = 1; i < s.length; i++) {
      const moved = Math.hypot(s[i].x - s[i - 1].x, s[i].y - s[i - 1].y);
      if (s[i].t - s[i - 1].t > 400 && moved < 3) count++;
    }
  });
  return count;
}

export default function CopySentenceGame({ question, onAnswer }) {
  const sentence = question?.payload?.sentence || 'The cat sat on the mat.';
  const canvasRef = useRef(null);
  const strokesRef = useRef([]);
  const drawing = useRef(false);
  const startedAt = useRef(null);
  const [inkCount, setInkCount] = useState(0);

  const paper = () => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, W, H);
    // faint writing line, light enough that the handwriting model treats it as paper
    ctx.strokeStyle = '#e6e9f0';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(30, H * 0.68);
    ctx.lineTo(W - 30, H * 0.68);
    ctx.stroke();
  };

  useEffect(() => { paper(); }, []);

  const pos = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * W, y: ((e.clientY - r.top) / r.height) * H, t: performance.now() };
  };

  const down = (e) => {
    e.preventDefault();
    canvasRef.current.setPointerCapture(e.pointerId);
    drawing.current = true;
    if (!startedAt.current) startedAt.current = performance.now();
    strokesRef.current.push([pos(e)]);
  };
  const move = (e) => {
    if (!drawing.current) return;
    const p = pos(e);
    const s = strokesRef.current[strokesRef.current.length - 1];
    const last = s[s.length - 1];
    s.push(p);
    const ctx = canvasRef.current.getContext('2d');
    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  };
  const up = () => {
    if (!drawing.current) return;
    drawing.current = false;
    setInkCount(strokesRef.current.length);
  };

  const clear = () => {
    strokesRef.current = [];
    startedAt.current = null;
    setInkCount(0);
    paper();
  };

  const done = () => {
    const strokes = strokesRef.current;
    onAnswer({
      userAnswer: `${strokes.length} strokes`,
      correctAnswer: sentence,
      isCorrect: strokes.length >= 5,
      motorJitterScore: Number(strokeJitter(strokes).toFixed(3)),
      hesitationCount: hesitations(strokes),
      rawData: {
        handwritingImage: canvasRef.current.toDataURL('image/png'),
        sentence,
        strokeCount: strokes.length,
        drawingMs: startedAt.current ? Math.round(performance.now() - startedAt.current) : 0,
      },
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, alignItems: 'center' }}>
      <div style={{ fontSize: '1.6rem', fontWeight: 700, letterSpacing: '0.02em', textAlign: 'center' }}>
        “{sentence}”
      </div>
      <p style={{ color: 'var(--text-secondary)', margin: 0, textAlign: 'center' }}>
        Write the sentence on the white paper with your finger, stylus or mouse.
      </p>
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerLeave={up}
        style={{ width: '100%', maxWidth: W, aspectRatio: `${W} / ${H}`, borderRadius: 12, touchAction: 'none', cursor: 'crosshair', boxShadow: 'var(--shadow-sm)' }}
        aria-label="Writing area"
      />
      <div style={{ display: 'flex', gap: 12 }}>
        <button type="button" className="btn-secondary" onClick={clear}><Eraser size={16} /> Start again</button>
        <button type="button" className="btn-primary" onClick={done} disabled={inkCount < 3}><Check size={16} /> I’m done</button>
      </div>
    </div>
  );
}
