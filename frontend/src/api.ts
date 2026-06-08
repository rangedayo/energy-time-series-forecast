// 백엔드 BFF(/sim/meta, /simulate) 클라이언트 + 응답 타입.
// 응답 구조의 진실의 원천은 app_streamlit/orchestrator.py:run_mpc_simulation 의 반환 dict.

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY ?? "dev-key-change-me";

export type PolicyKey = "naive" | "xgb_lookahead" | "mpc_xgb";

export interface SimMeta {
  regions: string[];
  start_min: string; // ISO
  start_max: string; // ISO
}

export interface PredictionPoint {
  timestamp: string;
  predicted_power_mwh: number;
  step: number;
}

export interface ActualPoint {
  timestamp: string;
  actual_power_mwh: number;
}

export interface HourlyPoint {
  step: number;
  soc: number;
  grid_sell_mwh: number;
  grid_buy_mwh: number;
  period: string | null;
}

export interface PolicySummary {
  net_revenue_krw: number;
  self_sufficiency_rate_pct: number;
  total_import_mwh: number;
  total_export_mwh: number;
  battery_cycles: number;
  [k: string]: number;
}

export interface PolicyResult {
  hourly: HourlyPoint[];
  summary: PolicySummary;
}

export interface SimulateResult {
  meta: {
    start_time: string;
    region: string;
    initial_soc: number;
    policies: PolicyKey[];
    horizon_used: number;
    sim_length: number;
  };
  predictions: PredictionPoint[];
  actuals: ActualPoint[];
  results: Record<PolicyKey, PolicyResult>;
  elapsed_ms: number;
}

export interface SimulateInput {
  region: string;
  start_time: string; // ISO, 정시
  initial_soc: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* ignore parse error */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export function fetchMeta(): Promise<SimMeta> {
  return request<SimMeta>("/sim/meta");
}

export function runSimulation(input: SimulateInput): Promise<SimulateResult> {
  return request<SimulateResult>("/simulate", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

// 정책 표시명 / 색상 — app_streamlit/ui/naming.py 와 일치.
export const POLICY_ORDER: PolicyKey[] = ["naive", "xgb_lookahead", "mpc_xgb"];

// 운영자 친화 한글 표시명 — app_streamlit/ui/naming.py 와 일치.
export const POLICY_LABEL: Record<PolicyKey, string> = {
  naive: "기본 운영",
  xgb_lookahead: "단기 예측 기반",
  mpc_xgb: "수익 최적화 (MPC)",
};

export const POLICY_DESCRIPTION: Record<PolicyKey, string> = {
  naive: "예측 없이 SOC를 0.20~0.80 범위로 고정 유지",
  xgb_lookahead: "다음 1시간 발전량 예측을 보고 SOC 목표 조정",
  mpc_xgb: "24시간 예측 기반 LP 최적화 (Rolling Horizon)",
};

export const POLICY_COLOR: Record<PolicyKey, string> = {
  naive: "#9AA4B2",
  xgb_lookahead: "#FFB800",
  mpc_xgb: "#00D9C0",
};
