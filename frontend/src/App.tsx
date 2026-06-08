import { useEffect, useState } from "react";
import { fetchMeta, runSimulation, type SimMeta, type SimulateInput, type SimulateResult } from "./api";
import Sidebar from "./components/Sidebar";
import { Header, KpiCards, MetricsTable, PolicyInfo } from "./components/Panels";
import { PredictionChart, RevenueChart, SocChart, TradingChart } from "./components/Charts";

export default function App() {
  const [meta, setMeta] = useState<SimMeta | null>(null);
  const [apiOk, setApiOk] = useState(false);
  const [result, setResult] = useState<SimulateResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMeta()
      .then((m) => {
        setMeta(m);
        setApiOk(true);
      })
      .catch((e: Error) => {
        setApiOk(false);
        setError(
          `API 서버에 연결할 수 없습니다 (${e.message}).\n` +
            "터미널에서 'uvicorn app.main:app' 실행 중인지 확인하세요."
        );
      });
  }, []);

  async function handleRun(input: SimulateInput) {
    setLoading(true);
    setError(null);
    try {
      const res = await runSimulation(input);
      setResult(res);
      setApiOk(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Header apiOk={apiOk} />
      <div className="shell">
        <Sidebar meta={meta} loading={loading} error={error} onRun={handleRun} />
        <main className="content">
          {result ? (
            <>
              <KpiCards result={result} />
              <PolicyInfo />
              <div className="section-title">핵심 지표</div>
              <MetricsTable result={result} />
              <div className="section-title">차트</div>
              <div className="charts">
                <PredictionChart result={result} />
                <SocChart result={result} />
                <RevenueChart result={result} />
                <TradingChart result={result} />
              </div>
            </>
          ) : (
            <div className="empty">
              좌측에서 입력을 설정한 후 <strong>실행</strong> 버튼을 눌러주세요.
            </div>
          )}
        </main>
      </div>
    </>
  );
}
