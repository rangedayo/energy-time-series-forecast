import { useMemo, useState } from "react";
import type { SimMeta, SimulateInput } from "../api";

interface Props {
  meta: SimMeta | null;
  loading: boolean;
  error: string | null;
  onRun: (input: SimulateInput) => void;
}

function isoDate(iso: string): string {
  return iso.slice(0, 10);
}

function clampDate(value: string, lo: string, hi: string): string {
  if (value < lo) return lo;
  if (value > hi) return hi;
  return value;
}

export default function Sidebar({ meta, loading, error, onRun }: Props) {
  const regions = meta?.regions ?? [];
  const dateMin = meta ? isoDate(meta.start_min) : "";
  const dateMax = meta ? isoDate(meta.start_max) : "";

  const defaultRegion = regions.includes("전라남도") ? "전라남도" : regions[0] ?? "";
  const defaultDate = useMemo(
    () => (meta ? clampDate("2022-06-15", dateMin, dateMax) : ""),
    [meta, dateMin, dateMax]
  );

  const [region, setRegion] = useState(defaultRegion);
  const [date, setDate] = useState(defaultDate);
  const [hour, setHour] = useState(9);
  const [soc, setSoc] = useState(50); // 0~100

  // meta 가 늦게 도착하면 기본값 동기화
  useMemo(() => {
    if (region === "" && defaultRegion) setRegion(defaultRegion);
    if (date === "" && defaultDate) setDate(defaultDate);
  }, [defaultRegion, defaultDate]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleRun(e: React.MouseEvent<HTMLButtonElement>) {
    // 리플
    const btn = e.currentTarget;
    const r = document.createElement("span");
    r.className = "ripple";
    const rect = btn.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    r.style.width = r.style.height = `${size}px`;
    r.style.left = `${e.clientX - rect.left - size / 2}px`;
    r.style.top = `${e.clientY - rect.top - size / 2}px`;
    btn.appendChild(r);
    setTimeout(() => r.remove(), 600);

    const hh = String(hour).padStart(2, "0");
    onRun({ region, start_time: `${date}T${hh}:00:00`, initial_soc: soc / 100 });
  }

  return (
    <aside className="sidebar">
      <h2>입력</h2>

      <div className="field">
        <label>지역 (region)</label>
        <select
          className="control"
          value={region}
          onChange={(e) => setRegion(e.target.value)}
          disabled={!meta}
        >
          {regions.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label>시뮬 시작 날짜</label>
        <input
          className="control"
          type="date"
          value={date}
          min={dateMin}
          max={dateMax}
          onChange={(e) => setDate(e.target.value)}
          disabled={!meta}
        />
      </div>

      <div className="field">
        <label>시뮬 시작 시각</label>
        <select
          className="control"
          value={hour}
          onChange={(e) => setHour(Number(e.target.value))}
        >
          {Array.from({ length: 24 }, (_, h) => (
            <option key={h} value={h}>
              {String(h).padStart(2, "0")}:00
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label>초기 SOC</label>
        <div className="slider-row">
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={soc}
            style={{ ["--p" as string]: `${soc}%` }}
            onChange={(e) => setSoc(Number(e.target.value))}
          />
          <span className="soc-val">{(soc / 100).toFixed(2)}</span>
        </div>
      </div>

      <button className="run-btn" onClick={handleRun} disabled={loading || !meta}>
        {loading ? "⏳ 실행 중..." : "⚡ 실행"}
      </button>

      {meta && (
        <p className="hint">
          사용 가능 범위
          <br />
          {dateMin} ~ {dateMax}
          <br />
          <br />
          3개 정책 시뮬레이션 (~5초)
        </p>
      )}

      {error && <div className="err">{error}</div>}
    </aside>
  );
}
