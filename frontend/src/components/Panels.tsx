import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  POLICY_DESCRIPTION,
  POLICY_LABEL,
  POLICY_ORDER,
  type PolicyKey,
  type SimulateResult,
} from "../api";

// ── Header ────────────────────────────────────────────────────────────────
export function Header({ apiOk }: { apiOk: boolean }) {
  return (
    <header className="app-header">
      <div className="logo">☀</div>
      <div>
        <h1>ESS 운영 도구</h1>
        <p className="sub">
          태양광 발전 예측 기반 ESS 운영 정책 비교 · naive / xgb_lookahead / mpc_xgb
        </p>
      </div>
      <div className={`pill ${apiOk ? "ok" : "bad"}`}>
        <span className="dot" />
        {apiOk ? "API 연결됨" : "API 응답 없음"}
      </div>
    </header>
  );
}

// ── 정책 설명 (접이식) ────────────────────────────────────────────────────
export function PolicyInfo() {
  return (
    <details className="policy-info">
      <summary>정책 설명 — 3가지 운영 방식</summary>
      <ul>
        {POLICY_ORDER.map((p) => (
          <li key={p}>
            <strong>{POLICY_LABEL[p]}</strong>
            <span>{POLICY_DESCRIPTION[p]}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

// ── 숫자 카운트업 훅 ──────────────────────────────────────────────────────
function useCountUp(target: number, decimals: number, duration = 900): string {
  const [val, setVal] = useState(0);
  const fromRef = useRef(0);
  useEffect(() => {
    const from = fromRef.current;
    const t0 = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / duration);
      const e = 1 - Math.pow(1 - p, 3);
      setVal(from + (target - from) * e);
      if (p < 1) raf = requestAnimationFrame(tick);
      else fromRef.current = target;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return val.toFixed(decimals);
}

// ── KPI 카드 ──────────────────────────────────────────────────────────────
interface Kpi {
  label: string;
  value: number;
  decimals: number;
  unit?: string;
  accent: string;
  delta: string;
  deltaUp: boolean;
}

function bestBy(
  result: SimulateResult,
  key: string,
  dir: "max" | "min"
): { policy: PolicyKey; value: number } {
  const entries = POLICY_ORDER.map((p) => ({
    policy: p,
    value: Number(result.results[p].summary[key]),
  }));
  return entries.reduce((a, b) =>
    dir === "max" ? (b.value > a.value ? b : a) : b.value < a.value ? b : a
  );
}

function buildKpis(result: SimulateResult): Kpi[] {
  const naive = result.results.naive.summary;
  const rev = bestBy(result, "net_revenue_krw", "max");
  const suf = bestBy(result, "self_sufficiency_rate_pct", "max");
  const imp = bestBy(result, "total_import_mwh", "min");
  const cycles = result.results[rev.policy].summary.battery_cycles;

  const revGainPct =
    naive.net_revenue_krw !== 0
      ? ((rev.value - naive.net_revenue_krw) / Math.abs(naive.net_revenue_krw)) * 100
      : 0;
  const sufGain = suf.value - naive.self_sufficiency_rate_pct;

  return [
    {
      label: `최적 순수익 · ${POLICY_LABEL[rev.policy]}`,
      value: rev.value / 1_000_000,
      decimals: 2,
      unit: "백만원",
      accent: "var(--solar)",
      delta: `naive 대비 ${revGainPct >= 0 ? "+" : ""}${revGainPct.toFixed(1)}%`,
      deltaUp: revGainPct >= 0,
    },
    {
      label: "자급률",
      value: suf.value,
      decimals: 1,
      unit: "%",
      accent: "var(--energy)",
      delta: `${sufGain >= 0 ? "+" : ""}${sufGain.toFixed(1)}%p`,
      deltaUp: sufGain >= 0,
    },
    {
      label: "배터리 사이클",
      value: cycles,
      decimals: 2,
      accent: "var(--blue)",
      delta: "수명 영향 지표",
      deltaUp: true,
    },
    {
      label: "총 매입 전력 (최소)",
      value: imp.value,
      decimals: 2,
      unit: "MWh",
      accent: "var(--red)",
      delta: `${POLICY_LABEL[imp.policy]} best`,
      deltaUp: true,
    },
  ];
}

function KpiCard({ kpi, index }: { kpi: Kpi; index: number }) {
  const shown = useCountUp(kpi.value, kpi.decimals);
  return (
    <motion.div
      className="kpi"
      style={{ ["--accent" as string]: kpi.accent }}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.08 * index + 0.05, duration: 0.4, ease: [0.2, 0.7, 0.3, 1] }}
      whileHover={{ y: -5 }}
    >
      <div className="k-label">{kpi.label}</div>
      <div className="k-val">
        {shown}
        {kpi.unit && <span className="unit">{kpi.unit}</span>}
      </div>
      <div className={`k-delta ${kpi.deltaUp ? "up" : "down"}`}>
        {kpi.deltaUp ? "▲" : "▼"} {kpi.delta}
      </div>
    </motion.div>
  );
}

export function KpiCards({ result }: { result: SimulateResult }) {
  const kpis = buildKpis(result);
  return (
    <div className="kpis">
      {kpis.map((k, i) => (
        <KpiCard key={k.label} kpi={k} index={i} />
      ))}
    </div>
  );
}

// ── 핵심 지표 표 ──────────────────────────────────────────────────────────
type Dir = "max" | "min" | null;
const METRIC_ROWS: { label: string; key: string; decimals: number; dir: Dir; comma?: boolean }[] = [
  { label: "순수익 (원)", key: "net_revenue_krw", decimals: 0, dir: "max", comma: true },
  { label: "자급률 (%)", key: "self_sufficiency_rate_pct", decimals: 2, dir: "max" },
  { label: "총 매입 (MWh)", key: "total_import_mwh", decimals: 2, dir: "min" },
  { label: "총 판매 (MWh)", key: "total_export_mwh", decimals: 2, dir: null },
  { label: "배터리 사이클", key: "battery_cycles", decimals: 3, dir: null },
];

function fmt(v: number, decimals: number, comma?: boolean): string {
  return comma
    ? v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
    : v.toFixed(decimals);
}

export function MetricsTable({ result }: { result: SimulateResult }) {
  return (
    <div className="panel">
      <table className="metrics">
        <thead>
          <tr>
            <th>지표</th>
            {POLICY_ORDER.map((p) => (
              <th key={p}>{POLICY_LABEL[p]}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {METRIC_ROWS.map((row) => {
            const vals = POLICY_ORDER.map((p) => Number(result.results[p].summary[row.key]));
            let bestIdx = -1;
            if (row.dir === "max") bestIdx = vals.indexOf(Math.max(...vals));
            else if (row.dir === "min") bestIdx = vals.indexOf(Math.min(...vals));
            return (
              <tr key={row.key}>
                <td>{row.label}</td>
                {vals.map((v, i) => (
                  <td key={i} className={i === bestIdx ? "best" : undefined}>
                    {fmt(v, row.decimals, row.comma)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="caption">초록 칩 = 해당 지표 기준 best (순수익·자급률 max / 총 매입 min)</p>
    </div>
  );
}
