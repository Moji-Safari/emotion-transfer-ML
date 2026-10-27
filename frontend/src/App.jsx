import { useState } from 'react';
import Header from './components/Header';
import HostNecklace from './components/HostNecklace';
import GuestWristband from './components/GuestWristband';
import HumanMock from './components/HumanMock';
import PredictionForm from './components/PredictionForm';
import Footer from './components/Footer';
import * as api from './api';

const DEFAULT_BASELINE = {
  heart_rate: 68,
  hrv_rmssd: 55,
  hrv_sdnn: 82,
  respiratory_rate: 14,
  respiratory_energy: 0.012,
  respiratory_variability: 0.35,
  skin_temp_c: 33.0,
};

const DEFAULT_CURRENT = {
  heart_rate: 72,
  hrv_rmssd: 42,
  hrv_sdnn: 65,
  respiratory_rate: 16,
  respiratory_energy: 0.02,
  respiratory_variability: 0.5,
  skin_temp_c: 33.0,
};

function jitter(conditions, pct = 0.08) {
  const out = {};
  for (const k of Object.keys(conditions)) {
    const range = Math.abs(conditions[k]) * pct;
    const noise = (Math.random() * 2 - 1) * range;
    out[k] = Math.round((conditions[k] + noise) * 100) / 100;
  }
  return out;
}

function generateBaselineList(base = DEFAULT_BASELINE) {
  const list = [];
  for (let i = 0; i < 5; i++) list.push(jitter(base, 0.08));
  return list;
}

export default function App() {
  const [state, setState] = useState('baseline');
  const [baselineList, setBaselineList] = useState(() => generateBaselineList());
  const [current, setCurrent] = useState({ ...DEFAULT_CURRENT });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function regenerateBaseline() {
    setBaselineList(generateBaselineList());
    setResult(null);
    setError(null);
  }

  async function handlePredict() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.predictAffectiveState({
        baseline: baselineList,
        current,
      });
      setResult(res);
      setState(res.state);
    } catch (e) {
      console.error('Predict error:', e);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleMock(mockConditions) {
    setCurrent({ ...mockConditions });
    setResult(null);
    setError(null);
    setState('baseline');
  }

  function handleReset() {
    setBaselineList(generateBaselineList());
    setCurrent({ ...DEFAULT_CURRENT });
    setResult(null);
    setError(null);
    setState('baseline');
  }

  return (
    <div className="app">
      <Header />

      <div className="dashboard-three">
        <HostNecklace state={state} />
        <GuestWristband state={state} />
        <HumanMock onGenerate={handleMock} onReset={handleReset} />
      </div>

      <PredictionForm
        baselineList={baselineList}
        onGenerateBaseline={regenerateBaseline}
        conditions={current}
        setConditions={setCurrent}
        onPredict={handlePredict}
        onReset={handleReset}
        loading={loading}
        lastResult={result}
      />

      {error && <div className="error-block">{error}</div>}

      <Footer />
    </div>
  );
}