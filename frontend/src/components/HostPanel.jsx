import { useState, useMemo } from 'react'

// Simulated EDA waveform generator.
// Baseline mode: low, steady. Stress mode: high mean, more variance.
function generateEdaWindow(mode, length = 120) {
  const baseMean = mode === 'stress' ? 0.55 : 0.30
  const noise = mode === 'stress' ? 0.06 : 0.01
  const slope = mode === 'stress' ? 0.0015 : 0.0

  const values = []
  for (let i = 0; i < length; i++) {
    const trend = slope * i
    const n = (Math.random() - 0.5) * noise
    values.push(baseMean + trend + n)
  }
  return values
}

// Compute the same 4 features the backend expects.
function computeFeatures(window) {
  const mean = window.reduce((a, b) => a + b, 0) / window.length
  const variance =
    window.reduce((a, b) => a + (b - mean) ** 2, 0) / window.length
  const std = Math.sqrt(variance)
  let sumAbsDiff = 0
  for (let i = 1; i < window.length; i++) {
    sumAbsDiff += Math.abs(window[i] - window[i - 1])
  }
  const meanAbsChange = sumAbsDiff / (window.length - 1)
  // Fake TEMP — assumes stable ~33°C
  const tempMean = 33.0 + (Math.random() - 0.5) * 0.2
  return [mean, std, meanAbsChange, tempMean]
}

export default function HostPanel({
  mode, setMode,
  onSendWindow,
  onCalibrate,
  lastFeatures,
}) {
  const [window_, setWindow_] = useState(() => generateEdaWindow(mode))

  // Regenerate waveform when mode changes
  useMemo(() => {
    setWindow_(generateEdaWindow(mode))
  }, [mode])

  const pathD = useMemo(() => {
    const w = 300
    const h = 140
    const pad = 10
    const min = Math.min(...window_)
    const max = Math.max(...window_)
    const range = max - min || 1
    return window_
      .map((v, i) => {
        const x = pad + (i / (window_.length - 1)) * (w - 2 * pad)
        const y = h - pad - ((v - min) / range) * (h - 2 * pad)
        return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`
      })
      .join(' ')
  }, [window_])

  const features = computeFeatures(window_)

  return (
    <div className="panel">
      <div className="panel-title">Host Device · Simulated</div>

      <div className="waveform-label">EDA · 30s window · 4 Hz</div>
      <div className="waveform">
        <svg viewBox="0 0 300 140" preserveAspectRatio="none">
          <path
            d={pathD}
            fill="none"
            stroke="#00704A"
            strokeWidth="1.5"
          />
        </svg>
      </div>

      <div>
        <div className="waveform-label" style={{ marginBottom: 8 }}>
          Simulated state
        </div>
        <div className="mode-toggle">
          <button
            className={mode === 'baseline' ? 'active' : ''}
            onClick={() => setMode('baseline')}
          >
            Baseline
          </button>
          <button
            className={mode === 'stress' ? 'active' : ''}
            onClick={() => setMode('stress')}
          >
            Stress
          </button>
        </div>
      </div>

      <div className="button-row">
        <button
          className="secondary"
          onClick={() => onCalibrate(mode)}
        >
          Capture Baseline
        </button>
        <button
          onClick={() => onSendWindow(features)}
        >
          Send Window
        </button>
      </div>

      {lastFeatures && (
        <div className="waveform-label">
          Last sent: [{lastFeatures.map((f) => f.toFixed(3)).join(', ')}]
        </div>
      )}
    </div>
  )
}