// Monte Carlo price paths in the browser. Mirrors src/mc/validate.py and src/mc/models.py (same
// step rules, zero drift); a test compares them through Node. Settings that need fitting (the
// Student-t degrees of freedom, the jump sizes) come from Python in the page data.

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeRandom(seed) {
  const u = mulberry32(seed);
  let spare = null;
  const normal = () => {  // Box-Muller, two draws at a time
    if (spare !== null) { const s = spare; spare = null; return s; }
    let a = 0; while (a === 0) a = u();
    const r = Math.sqrt(-2 * Math.log(a)), th = 2 * Math.PI * u();
    spare = r * Math.sin(th);
    return r * Math.cos(th);
  };
  const gamma = (k) => {  // Marsaglia-Tsang, shape k, scale 1
    if (k < 1) return gamma(k + 1) * Math.pow(u(), 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = normal(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const w = u();
      if (w < 1 - 0.0331 * x ** 4 || Math.log(w) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  };
  const studentT = (df) => normal() / Math.sqrt(2 * gamma(df / 2) / df);
  const poisson = (lam) => { const l = Math.exp(-lam); let k = 0, p = 1; do { k++; p *= u(); } while (p > l); return k - 1; };
  return { u, normal, studentT, poisson };
}

// Daily log steps of one path under a model. `p` holds the fitted settings, `c` the centred
// history (for the bootstraps).
function steps(model, p, c, days, rnd) {
  const out = new Float64Array(days), drift = -0.5 * p.sd * p.sd;
  if (model === "gbm" || (model === "jump" && !p.jumpOk)) {
    for (let t = 0; t < days; t++) out[t] = drift + p.sd * rnd.normal();
  } else if (model === "t") {
    const scale = p.sd * Math.sqrt((p.df - 2) / p.df);
    for (let t = 0; t < days; t++) out[t] = drift + scale * rnd.studentT(p.df);
  } else if (model === "jump") {
    for (let t = 0; t < days; t++) {
      const n = rnd.poisson(p.lam);
      out[t] = p.sdD * rnd.normal() + p.mJ * n + p.sJ * Math.sqrt(n) * rnd.normal() + drift - p.lam * p.mJ;
    }
  } else if (model === "bootstrap") {
    for (let t = 0; t < days; t++) out[t] = c[Math.floor(rnd.u() * c.length)];
  } else if (model === "block") {
    for (let t = 0; t < days; t += 20) {
      const start = Math.floor(rnd.u() * c.length);
      for (let j = 0; j < 20 && t + j < days; j++) out[t + j] = c[(start + j) % c.length];
    }
  }
  return out;
}

// Centre history as the models do: mean −σ²/2, so the expected price stays at today's.
function centre(hist) {
  const n = hist.length, m = hist.reduce((a, b) => a + b, 0) / n;
  const v = hist.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1);
  return hist.map((x) => x - m - 0.5 * v);
}

// Prices, one Float64Array of days + 1 per path; column 0 is s0.
function simulate(model, p, hist, s0, days, nPaths, seed = 1) {
  const rnd = makeRandom(seed), c = centre(hist), paths = [];
  for (let i = 0; i < nPaths; i++) {
    const st = steps(model, p, c, days, rnd), path = new Float64Array(days + 1);
    path[0] = s0;
    let acc = 0;
    for (let t = 0; t < days; t++) { acc += st[t]; path[t + 1] = s0 * Math.exp(acc); }
    paths.push(path);
  }
  return paths;
}

function quantile(sorted, q) {  // numpy's default (linear interpolation), q in [0, 1]
  const i = q * (sorted.length - 1), lo = Math.floor(i), f = i - lo;
  return lo + 1 < sorted.length ? sorted[lo] + f * (sorted[lo + 1] - sorted[lo]) : sorted[lo];
}

// VaR and CVaR as positive loss amounts (mirrors mc/risk.py var_cvar).
function varCvar(pnl, level) {
  const loss = pnl.map((x) => -x), s = [...loss].sort((a, b) => a - b), v = quantile(s, level);
  const tail = loss.filter((x) => x >= v);
  return [v, tail.reduce((a, b) => a + b, 0) / tail.length];
}

// Share of paths touching `barrier` at any point, and finishing beyond it (mc/risk.py).
function probTouch(paths, barrier) {
  const up = barrier > paths[0][0];
  return paths.filter((p) => p.some((x) => up ? x >= barrier : x <= barrier)).length / paths.length;
}
function probEnd(paths, level) {
  const up = level > paths[0][0];
  return paths.filter((p) => up ? p[p.length - 1] >= level : p[p.length - 1] <= level).length / paths.length;
}

if (typeof module !== "undefined") module.exports = { makeRandom, steps, centre, simulate, quantile, varCvar, probTouch, probEnd };
