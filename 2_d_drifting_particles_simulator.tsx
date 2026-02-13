import { useEffect, useMemo, useRef, useState } from "react";

type Vec2 = { x: number; y: number };

function clamp(v: number, a: number, b: number) {
  return Math.max(a, Math.min(b, v));
}

function mulberry32(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng: () => number) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function worldToCanvas(p: Vec2, w: number, h: number, bounds: Bounds) {
  const sx = (p.x - bounds.minX) / (bounds.maxX - bounds.minX);
  const sy = (p.y - bounds.minY) / (bounds.maxY - bounds.minY);
  return { x: sx * w, y: (1 - sy) * h };
}

function canvasToWorld(p: Vec2, w: number, h: number, bounds: Bounds) {
  const sx = clamp(p.x / w, 0, 1);
  const sy = clamp(1 - p.y / h, 0, 1);
  return {
    x: bounds.minX + sx * (bounds.maxX - bounds.minX),
    y: bounds.minY + sy * (bounds.maxY - bounds.minY),
  };
}

type Bounds = { minX: number; maxX: number; minY: number; maxY: number };

type SimState = {
  rng: () => number;
  x: Float32Array;
  v: Float32Array;
  y: Float32Array;
  yCount: number;
  yCap: number;
  step: number;
  meanV: number;
  bounds: Bounds;
  ws?: {
    logits?: Float32Array;
    Arow?: Float32Array;
    Acol?: Float32Array;
    spos?: Float32Array;
    sneg?: Float32Array;
  };
};

function ensureWorkspace(sim: SimState, n: number, m: number) {
  const size = n * m;
  if (!sim.ws) sim.ws = {};
  if (!sim.ws.logits || sim.ws.logits.length !== size) sim.ws.logits = new Float32Array(size);
  if (!sim.ws.Arow || sim.ws.Arow.length !== size) sim.ws.Arow = new Float32Array(size);
  if (!sim.ws.Acol || sim.ws.Acol.length !== size) sim.ws.Acol = new Float32Array(size);
  if (!sim.ws.spos || sim.ws.spos.length !== n) sim.ws.spos = new Float32Array(n);
  if (!sim.ws.sneg || sim.ws.sneg.length !== n) sim.ws.sneg = new Float32Array(n);
}

function computeV_paperAligned(sim: SimState, temperature: number) {
  const x = sim.x;
  const y = sim.y;
  const n = x.length / 2;
  const nPos = sim.yCount;
  const nNeg = n;
  const m = nPos + nNeg;

  ensureWorkspace(sim, n, m);
  const logits = sim.ws!.logits!;
  const Arow = sim.ws!.Arow!;
  const Acol = sim.ws!.Acol!;
  const spos = sim.ws!.spos!;
  const sneg = sim.ws!.sneg!;

  const invT = -1.0 / Math.max(temperature, 1e-6);

  for (let i = 0; i < n; i++) {
    const xi = x[2 * i + 0];
    const yi = x[2 * i + 1];

    for (let j = 0; j < nPos; j++) {
      const dx = xi - y[2 * j + 0];
      const dy = yi - y[2 * j + 1];
      const dist = Math.hypot(dx, dy);
      logits[i * m + j] = dist * invT;
    }

    for (let k = 0; k < nNeg; k++) {
      const dx = xi - x[2 * k + 0];
      const dy = yi - x[2 * k + 1];
      let dist = Math.hypot(dx, dy);
      if (k === i) dist = 1e6;
      logits[i * m + (nPos + k)] = dist * invT;
    }
  }

  for (let i = 0; i < n; i++) {
    let maxv = -Infinity;
    const base = i * m;
    for (let j = 0; j < m; j++) {
      const v = logits[base + j];
      if (v > maxv) maxv = v;
    }
    let sum = 0;
    for (let j = 0; j < m; j++) {
      const e = Math.exp(logits[base + j] - maxv);
      Arow[base + j] = e;
      sum += e;
    }
    const inv = 1.0 / (sum + 1e-12);
    for (let j = 0; j < m; j++) Arow[base + j] *= inv;
  }

  for (let j = 0; j < m; j++) {
    let maxv = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = logits[i * m + j];
      if (v > maxv) maxv = v;
    }
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const e = Math.exp(logits[i * m + j] - maxv);
      Acol[i * m + j] = e;
      sum += e;
    }
    const inv = 1.0 / (sum + 1e-12);
    for (let i = 0; i < n; i++) Acol[i * m + j] *= inv;
  }

  for (let i = 0; i < n; i++) {
    let sp = 0;
    let sn = 0;
    const base = i * m;

    for (let j = 0; j < m; j++) {
      const a = Math.sqrt((Arow[base + j] * Acol[base + j]) + 1e-24);
      Arow[base + j] = a;
      if (j < nPos) sp += a;
      else sn += a;
    }

    spos[i] = sp;
    sneg[i] = sn;
  }

  let mean = 0;
  for (let i = 0; i < n; i++) {
    const base = i * m;

    let px = 0;
    let py = 0;
    const negScale = sneg[i];
    for (let j = 0; j < nPos; j++) {
      const w = Arow[base + j] * negScale;
      px += w * y[2 * j + 0];
      py += w * y[2 * j + 1];
    }

    let nx = 0;
    let ny = 0;
    const posScale = spos[i];
    for (let k = 0; k < nNeg; k++) {
      const w = Arow[base + (nPos + k)] * posScale;
      nx += w * x[2 * k + 0];
      ny += w * x[2 * k + 1];
    }

    const vx = px - nx;
    const vy = py - ny;
    sim.v[2 * i + 0] = vx;
    sim.v[2 * i + 1] = vy;
    mean += Math.hypot(vx, vy);
  }

  sim.meanV = mean / n;
}

export default function DriftingParticles2D() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);

  const [seed, setSeed] = useState(0);
  const [nParticles, setNParticles] = useState(350);

  const [temperature, setTemperature] = useState(0.30);
  const [stepSize, setStepSize] = useState(0.06);
  const [noiseScale, setNoiseScale] = useState(0.0);

  const [drawVectors, setDrawVectors] = useState(false);
  const [pause, setPause] = useState(false);

  const [clusters, setClusters] = useState(4);
  const [basePosCount, setBasePosCount] = useState(600);

  const [editPositives, setEditPositives] = useState(false);
  const [eraseMode, setEraseMode] = useState(false);
  const [brushRadius, setBrushRadius] = useState(0.10);
  const brushArea = Math.PI * brushRadius * brushRadius;
  const brushParticlesPerStamp = Math.max(1, Math.round(brushArea * (230 / (1 + brushRadius * 10))));

  const simRef = useRef<SimState>({
    rng: mulberry32(0),
    x: new Float32Array(0),
    v: new Float32Array(0),
    y: new Float32Array(0),
    yCount: 0,
    yCap: 0,
    step: 0,
    meanV: 0,
    bounds: { minX: -2.5, maxX: 2.5, minY: -2.5, maxY: 2.5 },
  });

  const centers = useMemo(() => {
    const base = [
      { x: -1.2, y: 0.9 },
      { x: 1.1, y: 0.9 },
      { x: -0.2, y: -1.15 },
      { x: 1.25, y: -0.8 },
      { x: -1.3, y: -0.2 },
      { x: 0.1, y: 1.2 },
      { x: 0.9, y: -0.1 },
      { x: -0.7, y: 0.2 },
    ];
    const k = clamp(clusters, 1, base.length);
    return base.slice(0, k);
  }, [clusters]);

  function ensureYCapacity(sim: SimState, needed: number) {
    if (sim.yCap >= needed) return;
    let cap = Math.max(256, sim.yCap);
    while (cap < needed) cap *= 2;
    const next = new Float32Array(cap * 2);
    next.set(sim.y.subarray(0, sim.yCount * 2));
    sim.y = next;
    sim.yCap = cap;
  }

  function resetParticles() {
    const sim = simRef.current;
    sim.rng = mulberry32(seed);
    sim.x = new Float32Array(nParticles * 2);
    sim.v = new Float32Array(nParticles * 2);
    for (let i = 0; i < nParticles; i++) {
      sim.x[2 * i + 0] = (sim.rng() - 0.5) * 5.0;
      sim.x[2 * i + 1] = (sim.rng() - 0.5) * 5.0;
      sim.v[2 * i + 0] = 0;
      sim.v[2 * i + 1] = 0;
    }
    sim.step = 0;
    sim.meanV = 0;
    sim.ws = undefined;
  }

  function resetPositivesToClusters() {
    const sim = simRef.current;
    const rng = sim.rng;
    const nPos = basePosCount;
    ensureYCapacity(sim, nPos);
    const perCenter = Math.floor(nPos / centers.length);
    let k = 0;
    for (let c = 0; c < centers.length; c++) {
      const cx = centers[c].x;
      const cy = centers[c].y;
      const count = c === centers.length - 1 ? nPos - perCenter * (centers.length - 1) : perCenter;
      for (let j = 0; j < count; j++) {
        sim.y[2 * k + 0] = cx + 0.27 * randn(rng);
        sim.y[2 * k + 1] = cy + 0.27 * randn(rng);
        k++;
      }
    }
    sim.yCount = nPos;
  }

  function clearPositives() {
    const sim = simRef.current;
    sim.yCount = 0;
  }

  function addPositiveSpray(center: Vec2) {
    const sim = simRef.current;
    const rng = sim.rng;
    const radius = brushRadius;
    const count = Math.max(1, Math.round(Math.PI * radius * radius * (230 / (1 + radius * 10))));
    ensureYCapacity(sim, sim.yCount + count);
    for (let i = 0; i < count; i++) {
      const theta = rng() * Math.PI * 2;
      const rr = Math.sqrt(rng()) * radius;
      const x = clamp(center.x + rr * Math.cos(theta), sim.bounds.minX, sim.bounds.maxX);
      const y = clamp(center.y + rr * Math.sin(theta), sim.bounds.minY, sim.bounds.maxY);
      sim.y[2 * sim.yCount + 0] = x;
      sim.y[2 * sim.yCount + 1] = y;
      sim.yCount += 1;
    }
  }

  function eraseInRadius(p: Vec2) {
    const sim = simRef.current;
    if (sim.yCount === 0) return;
    const rad2 = brushRadius * brushRadius;
    let write = 0;
    for (let read = 0; read < sim.yCount; read++) {
      const dx = sim.y[2 * read + 0] - p.x;
      const dy = sim.y[2 * read + 1] - p.y;
      const d2 = dx * dx + dy * dy;
      if (d2 > rad2) {
        sim.y[2 * write + 0] = sim.y[2 * read + 0];
        sim.y[2 * write + 1] = sim.y[2 * read + 1];
        write++;
      }
    }
    sim.yCount = write;
  }

  useEffect(() => {
    resetParticles();
  }, [seed, nParticles]);

  useEffect(() => {
    resetPositivesToClusters();
  }, [centers.length, basePosCount, seed]);

  function stepSim() {
    const sim = simRef.current;
    computeV_paperAligned(sim, temperature);

    const x = sim.x;
    const v = sim.v;
    const rng = sim.rng;

    for (let i = 0; i < x.length / 2; i++) {
      x[2 * i + 0] += stepSize * v[2 * i + 0] + noiseScale * randn(rng);
      x[2 * i + 1] += stepSize * v[2 * i + 1] + noiseScale * randn(rng);
      x[2 * i + 0] = clamp(x[2 * i + 0], sim.bounds.minX, sim.bounds.maxX);
      x[2 * i + 1] = clamp(x[2 * i + 1], sim.bounds.minY, sim.bounds.maxY);
    }

    sim.step += 1;
  }

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.floor(rect.width * dpr);
    const h = Math.floor(rect.height * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    ctx.clearRect(0, 0, w, h);

    const sim = simRef.current;

    ctx.fillStyle = "rgba(50, 110, 255, 0.28)";
    for (let j = 0; j < sim.yCount; j++) {
      const p = worldToCanvas({ x: sim.y[2 * j + 0], y: sim.y[2 * j + 1] }, w, h, sim.bounds);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.3 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = "rgba(20, 20, 20, 0.95)";
    for (let i = 0; i < sim.x.length / 2; i++) {
      const p = worldToCanvas({ x: sim.x[2 * i + 0], y: sim.x[2 * i + 1] }, w, h, sim.bounds);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3.0 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }

    if (drawVectors) {
      ctx.strokeStyle = "rgba(0, 0, 0, 0.22)";
      ctx.lineWidth = 1.0 * dpr;
      const stride = Math.max(1, Math.floor((sim.x.length / 2) / 140));
      for (let i = 0; i < sim.x.length / 2; i += stride) {
        const p0 = worldToCanvas({ x: sim.x[2 * i + 0], y: sim.x[2 * i + 1] }, w, h, sim.bounds);
        const vx = sim.v[2 * i + 0];
        const vy = sim.v[2 * i + 1];
        const scale = 0.35;
        const p1 = worldToCanvas({ x: sim.x[2 * i + 0] + scale * vx, y: sim.x[2 * i + 1] + scale * vy }, w, h, sim.bounds);
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
      }
    }

    ctx.fillStyle = "rgba(0,0,0,0.75)";
    ctx.font = `${14 * dpr}px ui-sans-serif, system-ui`;
    ctx.fillText(`step=${sim.step}   mean|V|=${sim.meanV.toFixed(3)}   T=${temperature.toFixed(2)}   pos=${sim.yCount}`, 12 * dpr, 22 * dpr);

    if (editPositives) {
      ctx.fillStyle = "rgba(0,0,0,0.60)";
      ctx.font = `${12 * dpr}px ui-sans-serif, system-ui`;
      ctx.fillText(eraseMode ? "edit: erase" : "edit: draw", 12 * dpr, 42 * dpr);
    }
  }

  useEffect(() => {
    const loop = () => {
      if (!pause) stepSim();
      draw();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [pause, temperature, stepSize, noiseScale, drawVectors, editPositives, eraseMode, brushRadius]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let isDown = false;
    let lastP: Vec2 | null = null;

    const onPointerDown = (e: PointerEvent) => {
      if (!editPositives) return;
      isDown = true;
      canvas.setPointerCapture(e.pointerId);
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.floor(rect.width * dpr);
      const h = Math.floor(rect.height * dpr);
      const p = canvasToWorld({ x: (e.clientX - rect.left) * dpr, y: (e.clientY - rect.top) * dpr }, w, h, simRef.current.bounds);
      if (eraseMode) eraseInRadius(p);
      else addPositiveSpray(p);
      lastP = p;
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!editPositives || !isDown) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.floor(rect.width * dpr);
      const h = Math.floor(rect.height * dpr);
      const p = canvasToWorld({ x: (e.clientX - rect.left) * dpr, y: (e.clientY - rect.top) * dpr }, w, h, simRef.current.bounds);

      if (lastP) {
        const dx = p.x - lastP.x;
        const dy = p.y - lastP.y;
        const d = Math.hypot(dx, dy);
        const spacing = Math.max(0.015, brushRadius * 0.35);
        const steps = Math.max(1, Math.min(40, Math.floor(d / spacing)));
        for (let s = 1; s <= steps; s++) {
          const t = s / (steps + 1);
          const q = { x: lastP.x + t * dx, y: lastP.y + t * dy };
          if (eraseMode) eraseInRadius(q);
          else addPositiveSpray(q);
        }
      } else {
        if (eraseMode) eraseInRadius(p);
        else addPositiveSpray(p);
      }
      lastP = p;
    };

    const onPointerUp = (e: PointerEvent) => {
      if (!editPositives) return;
      isDown = false;
      lastP = null;
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch {}
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);

    return () => {
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerUp);
    };
  }, [editPositives, eraseMode, brushRadius]);

  return (
    <div className="w-full h-full p-4 md:p-5 flex flex-col gap-4 bg-gradient-to-b from-slate-100 via-sky-50 to-indigo-100">
      <div className="rounded-2xl border border-white/60 bg-white/80 backdrop-blur shadow-sm px-4 py-3 flex items-center justify-between gap-3">
        <div className="flex flex-col">
          <div className="text-xl font-semibold text-slate-900">Drifting particles in 2D</div>
          <div className="text-xs text-slate-600">Paper-aligned transport dynamics with interactive positive set painting</div>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-3 py-2 rounded-xl border border-slate-200 bg-white text-slate-900 hover:bg-slate-50 transition" onClick={() => setPause((p) => !p)}>
            {pause ? "Run" : "Pause"}
          </button>
          <button
            className="px-3 py-2 rounded-xl border border-slate-200 bg-white text-slate-900 hover:bg-slate-50 transition"
            onClick={() => {
              resetParticles();
              resetPositivesToClusters();
            }}
          >
            Reset
          </button>
          <button className="px-3 py-2 rounded-xl border border-slate-200 bg-white text-slate-900 hover:bg-slate-50 transition" onClick={() => setSeed((s) => (s + 1) % 100000)}>
            New seed
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="lg:col-span-2 rounded-2xl shadow-lg border border-white/60 bg-white/80 backdrop-blur overflow-hidden">
          <div className="h-[520px]">
            <canvas ref={canvasRef} className="w-full h-full" />
          </div>
        </div>

        <div className="rounded-2xl shadow-lg border border-white/60 bg-white/85 backdrop-blur p-4 flex flex-col gap-4">
          <div className="rounded-xl border border-indigo-100 bg-indigo-50/70 p-3 text-slate-700 space-y-2">
            <div className="text-xs uppercase tracking-wide font-semibold text-indigo-700">Algorithm 2 Drift</div>
            <div className="text-sm font-mono text-slate-800 leading-relaxed">
              logits = -||x - y|| / T, then row-softmax and column-softmax on [y<sub>pos</sub>, y<sub>neg</sub>]
            </div>
            <div className="text-sm font-mono text-slate-900 leading-relaxed">
              V = (W<sub>pos</sub> @ y<sub>pos</sub>) - (W<sub>neg</sub> @ y<sub>neg</sub>), y<sub>neg</sub> = x
            </div>
            <div className="pt-2 border-t border-indigo-200">
              <a 
                href="https://arxiv.org/abs/2602.04770" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-xs text-indigo-600 hover:text-indigo-800 underline font-medium inline-flex items-center gap-1"
              >
                <span>📄</span>
                <span>Generative Modeling via Drifting (arXiv:2602.04770)</span>
              </a>
            </div>
          </div>

          <div className="flex items-center justify-between gap-3">
            <div className="text-sm">Draw drift vectors</div>
            <input type="checkbox" checked={drawVectors} onChange={(e) => setDrawVectors(e.target.checked)} />
          </div>

          <div className={`rounded-xl border p-3 transition ${editPositives ? "border-blue-300 bg-blue-50/80 shadow-sm" : "border-slate-200 bg-slate-50/80"}`}>
            <div className="flex items-center justify-between gap-3">
              <div className="flex flex-col">
                <div className="text-sm font-semibold text-slate-900">Edit blue points</div>
                <div className="text-xs text-slate-600">Interactive spray brush for sculpting attractors in real time</div>
              </div>
              <input type="checkbox" checked={editPositives} onChange={(e) => setEditPositives(e.target.checked)} />
            </div>
            <div className="mt-2 text-xs text-slate-600">
              Mode: <span className="font-semibold">{eraseMode ? "Erase cloud" : "Spray add"}</span> | Brush area: <span className="font-semibold">{brushArea.toFixed(3)}</span> | Particles/stamp: <span className="font-semibold">~{brushParticlesPerStamp}</span>
            </div>
          </div>

          <div className="flex items-center justify-between gap-3">
            <div className="text-sm">Erase mode</div>
            <input type="checkbox" checked={eraseMode} onChange={(e) => setEraseMode(e.target.checked)} disabled={!editPositives} />
          </div>

          <div className="space-y-2 rounded-xl border border-blue-200 bg-blue-50/60 p-3">
            <div className="text-sm font-semibold text-blue-900">Brush radius: {brushRadius.toFixed(2)}</div>
            <div className="text-xs text-blue-800">
              Higher radius sprays more points over a wider and sparser area, like an airbrush.
            </div>
            <input
              className="w-full"
              type="range"
              min={0.03}
              max={0.30}
              step={0.01}
              value={brushRadius}
              onChange={(e) => setBrushRadius(parseFloat(e.target.value))}
              disabled={!editPositives}
            />
          </div>

          <div className="flex items-center gap-2">
            <button className="px-3 py-2 rounded-xl border border-slate-200 bg-white text-slate-900 hover:bg-slate-50 transition disabled:opacity-50" onClick={() => clearPositives()} disabled={!editPositives}>
              Clear blue
            </button>
            <button
              className="px-3 py-2 rounded-xl border border-slate-200 bg-white text-slate-900 hover:bg-slate-50 transition"
              onClick={() => resetPositivesToClusters()}
            >
              Reset blue
            </button>
          </div>

          <div className="space-y-2">
            <div className="text-sm font-medium">Temperature (T): {temperature.toFixed(2)}</div>
            <input className="w-full" type="range" min={0.05} max={1.20} step={0.01} value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} />
          </div>

          <div className="space-y-2">
            <div className="text-sm font-medium">Step size: {stepSize.toFixed(3)}</div>
            <input className="w-full" type="range" min={0.005} max={0.20} step={0.005} value={stepSize} onChange={(e) => setStepSize(parseFloat(e.target.value))} />
          </div>

          <div className="space-y-2">
            <div className="text-sm font-medium">Brownian noise: {noiseScale.toFixed(3)}</div>
            <input className="w-full" type="range" min={0.0} max={0.08} step={0.002} value={noiseScale} onChange={(e) => setNoiseScale(parseFloat(e.target.value))} />
          </div>

          <div className="space-y-2">
            <div className="text-sm font-medium">Particles: {nParticles}</div>
            <input className="w-full" type="range" min={80} max={800} step={10} value={nParticles} onChange={(e) => setNParticles(parseInt(e.target.value, 10))} />
          </div>

          <div className="space-y-2">
            <div className="text-sm font-medium">Base positives (reset): {basePosCount}</div>
            <input className="w-full" type="range" min={150} max={1400} step={25} value={basePosCount} onChange={(e) => setBasePosCount(parseInt(e.target.value, 10))} />
          </div>

          <div className="space-y-2">
            <div className="text-sm font-medium">Cluster count (reset): {clusters}</div>
            <input className="w-full" type="range" min={1} max={8} step={1} value={clusters} onChange={(e) => setClusters(parseInt(e.target.value, 10))} />
          </div>

          <div className="text-xs text-neutral-600">
            Tip: enable Edit blue points and draw on the canvas. Lower T makes interactions more local. Higher T makes them more global.
          </div>
        </div>
      </div>
    </div>
  );
}
