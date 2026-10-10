export default function DecisionPanel({ result, error, loading }) {
  if (error) {
    return (
      <div className="panel">
        <div className="panel-title">Recognition</div>
        <div className="error">{error}</div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="panel">
        <div className="panel-title">Recognition</div>
        <div className="empty">
          No window sent yet.
          <br />
          Click "Send Window" on the host panel.
        </div>
      </div>
    )
  }

  const pct = (result.stress_probability * 100).toFixed(1)

  return (
    <div className="panel">
      <div className="panel-title">Recognition</div>

      <div className="probability">
        <div className="value">{pct}%</div>
        <div className="label">Stress Probability</div>
      </div>

      <div className="confidence-bar">
        <div
          className="fill"
          style={{ width: `${result.stress_probability * 100}%` }}
        />
      </div>

      <div className="band-display">
        <div className="band-name">{result.band}</div>
        <div className="band-desc">
          {result.calibrated ? 'Calibrated' : 'Uncalibrated'}
        </div>
      </div>

      {loading && (
        <div className="waveform-label" style={{ textAlign: 'center' }}>
          Updating…
        </div>
      )}
    </div>
  )
}