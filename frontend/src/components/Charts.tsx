import { useState, type ReactNode } from "react";
import { motion } from "framer-motion";
// react-plotly.js/factory + dist-min: 전체 plotly.js 소스 대신 경량 번들 사용.
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import {
  POLICY_COLOR,
  POLICY_LABEL,
  POLICY_ORDER,
  type PolicyKey,
  type SimulateResult,
} from "../api";

const Plot = createPlotlyComponent(Plotly);

const FONT = { family: "Segoe UI, Malgun Gothic, sans-serif", color: "#9AA4B2", size: 11 };
const GRID = "#2A3344";
const CONFIG = { displayModeBar: false, responsive: true } as const;
const STYLE = { width: "100%", height: "280px" };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function baseLayout(extra: Record<string, any> = {}): Record<string, any> {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: FONT,
    margin: { l: 48, r: 20, t: 12, b: 38 },
    xaxis: { gridcolor: GRID, zerolinecolor: GRID },
    yaxis: { gridcolor: GRID, zerolinecolor: GRID },
    legend: { orientation: "h", y: 1.13, font: { size: 10 } },
    hoverlabel: { bgcolor: "#1B2230", bordercolor: GRID, font: { color: "#E6E9EF" } },
    hovermode: "x unified",
    ...extra,
  };
}

function ChartCard({ title, action, children }: { title: string; action?: ReactNode; children: ReactNode }) {
  return (
    <motion.div className="chart-card" whileHover={{ y: -3 }} transition={{ duration: 0.22 }}>
      <div className="ch-head">
        <span>{title}</span>
        {action}
      </div>
      {children}
    </motion.div>
  );
}

// ── (1) 예측 vs 실측 ──────────────────────────────────────────────────────
export function PredictionChart({ result }: { result: SimulateResult }) {
  const steps = result.predictions.map((p) => p.step);
  const pred = result.predictions.map((p) => p.predicted_power_mwh);
  const actual = result.actuals.map((a) => a.actual_power_mwh);
  const simLen = result.meta.sim_length;

  const data = [
    {
      x: steps.slice(0, simLen),
      y: pred.slice(0, simLen),
      name: "예측(시뮬)",
      mode: "lines",
      line: { color: "#4C8DFF", width: 2.5 },
    },
    {
      x: steps.slice(simLen - 1),
      y: pred.slice(simLen - 1),
      name: "예측(이후)",
      mode: "lines",
      line: { color: "#4C8DFF", width: 1.6, dash: "dash" },
      opacity: 0.6,
    },
    {
      x: steps,
      y: actual,
      name: "실측",
      mode: "lines",
      line: { color: "#FF6B6B", width: 1.6, dash: "dot" },
    },
  ];
  const layout = baseLayout({
    xaxis: { gridcolor: GRID, title: { text: "시뮬 step (h)", font: FONT } },
    shapes: [
      {
        type: "rect",
        x0: steps[0],
        x1: steps[Math.min(simLen, steps.length) - 1],
        y0: 0,
        y1: 1,
        yref: "paper",
        fillcolor: "#4C8DFF",
        opacity: 0.06,
        line: { width: 0 },
      },
    ],
  });
  return (
    <ChartCard title="발전량 예측 vs 실측 (48h)">
      <Plot data={data as never} layout={layout as never} config={CONFIG} style={STYLE} useResizeHandler />
    </ChartCard>
  );
}

// ── (2) 정책별 SOC ────────────────────────────────────────────────────────
export function SocChart({ result }: { result: SimulateResult }) {
  const data = POLICY_ORDER.map((p) => {
    const h = result.results[p].hourly;
    return {
      x: h.map((x) => x.step),
      y: h.map((x) => x.soc),
      name: POLICY_LABEL[p],
      mode: "lines",
      line: { color: POLICY_COLOR[p], width: p === "mpc_xgb" ? 2.6 : 2.0 },
    };
  });
  const layout = baseLayout({
    yaxis: { gridcolor: GRID, range: [0, 1], title: { text: "SOC", font: FONT } },
    xaxis: { gridcolor: GRID, title: { text: "시뮬 step (h)", font: FONT } },
    shapes: [0.1, 0.9].map((y) => ({
      type: "line",
      x0: 0,
      x1: result.meta.sim_length - 1,
      y0: y,
      y1: y,
      line: { color: "#566", dash: "dash", width: 1 },
    })),
  });
  return (
    <ChartCard title="정책별 SOC 추이 (24h)">
      <Plot data={data as never} layout={layout as never} config={CONFIG} style={STYLE} useResizeHandler />
    </ChartCard>
  );
}

// ── (3) 순수익 vs 자급률 ──────────────────────────────────────────────────
export function RevenueChart({ result }: { result: SimulateResult }) {
  const labels = POLICY_ORDER.map((p) => POLICY_LABEL[p]);
  const rev = POLICY_ORDER.map((p) => result.results[p].summary.net_revenue_krw / 1_000_000);
  const suf = POLICY_ORDER.map((p) => result.results[p].summary.self_sufficiency_rate_pct);
  const data = [
    {
      x: labels,
      y: rev,
      name: "순수익(백만원)",
      type: "bar",
      marker: { color: POLICY_ORDER.map((p) => POLICY_COLOR[p]) },
    },
    {
      x: labels,
      y: suf,
      name: "자급률(%)",
      yaxis: "y2",
      mode: "lines+markers",
      line: { color: "#4C8DFF", width: 2 },
      marker: { size: 9 },
    },
  ];
  const layout = baseLayout({
    hovermode: "closest",
    yaxis: { gridcolor: GRID, title: { text: "백만원", font: FONT } },
    yaxis2: { overlaying: "y", side: "right", showgrid: false, title: { text: "%", font: FONT } },
  });
  return (
    <ChartCard title="정책별 순수익 vs 자급률">
      <Plot data={data as never} layout={layout as never} config={CONFIG} style={STYLE} useResizeHandler />
    </ChartCard>
  );
}

// ── (4) 시간대별 매매 (탭 전환) ───────────────────────────────────────────
const PERIOD_FILL: Record<string, string> = {
  off_peak: "rgba(76,141,255,0.06)",
  mid_peak: "rgba(255,184,0,0.07)",
  max_peak: "rgba(255,107,107,0.09)",
};

export function TradingChart({ result }: { result: SimulateResult }) {
  const [selected, setSelected] = useState<PolicyKey>("mpc_xgb");
  const hourly = result.results[selected].hourly;
  const steps = hourly.map((h) => h.step);
  const sell = hourly.map((h) => h.grid_sell_mwh);
  const buy = hourly.map((h) => -h.grid_buy_mwh);
  const soc = hourly.map((h) => h.soc);

  const shapes = hourly
    .filter((h) => h.period && PERIOD_FILL[h.period])
    .map((h) => ({
      type: "rect",
      x0: h.step - 0.5,
      x1: h.step + 0.5,
      y0: 0,
      y1: 1,
      yref: "paper",
      fillcolor: PERIOD_FILL[h.period as string],
      line: { width: 0 },
      layer: "below",
    }));

  const data = [
    { x: steps, y: sell, name: "판매", type: "bar", marker: { color: "#34D399" } },
    { x: steps, y: buy, name: "매입", type: "bar", marker: { color: "#FF6B6B" } },
    {
      x: steps,
      y: soc,
      name: "SOC",
      yaxis: "y2",
      mode: "lines+markers",
      line: { color: "#E6E9EF", width: 1.6 },
      marker: { size: 4 },
    },
  ];
  const layout = baseLayout({
    barmode: "relative",
    transition: { duration: 500, easing: "cubic-in-out" },
    xaxis: { gridcolor: GRID, title: { text: "시뮬 step (h)", font: FONT } },
    yaxis: { gridcolor: GRID, title: { text: "MWh (+판매/−매입)", font: FONT } },
    yaxis2: { overlaying: "y", side: "right", range: [0, 1], showgrid: false },
    shapes,
  });

  const action = (
    <div className="seg">
      {POLICY_ORDER.map((p) => (
        <button
          key={p}
          className={p === selected ? "active" : ""}
          onClick={() => setSelected(p)}
        >
          {POLICY_LABEL[p]}
        </button>
      ))}
    </div>
  );

  return (
    <ChartCard title="시간대별 매매" action={action}>
      <Plot data={data as never} layout={layout as never} config={CONFIG} style={STYLE} useResizeHandler />
    </ChartCard>
  );
}
