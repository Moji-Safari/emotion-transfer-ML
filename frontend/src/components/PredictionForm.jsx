export default function PredictionForm({
  baselineList,
  onGenerateBaseline,
  conditions,
  setConditions,
  onPredict,
  onReset,
  loading,
  lastResult,
}) {
  const update = (key, value) =>
    setConditions((c) => ({ ...c, [key]: value }));

  return (
    <div className="prediction-panel">
      <div className="form-header">
        <div className="selector-label">Subject body condition</div>
        <div className="baseline-info">
          baseline readings: {baselineList.length}
          <button className="link-button" onClick={onGenerateBaseline}>
            Regenerate baseline
          </button>
        </div>
      </div>

      <div className="form-grid">
        <div className="form-row">
          <label>Heart rate (bpm)</label>
          <input
            type="number"
            min="50"
            max="120"
            step="1"
            value={conditions.heart_rate}
            onChange={(e) => update("heart_rate", parseFloat(e.target.value))}
          />
        </div>

        <div className="form-row">
          <label>HRV RMSSD (ms)</label>
          <input
            type="number"
            min="10"
            max="100"
            step="0.1"
            value={conditions.hrv_rmssd}
            onChange={(e) => update("hrv_rmssd", parseFloat(e.target.value))}
          />
        </div>

        <div className="form-row">
          <label>HRV SDNN (ms)</label>
          <input
            type="number"
            min="20"
            max="150"
            step="0.1"
            value={conditions.hrv_sdnn}
            onChange={(e) => update("hrv_sdnn", parseFloat(e.target.value))}
          />
        </div>

        <div className="form-row">
          <label>Respiratory rate</label>
          <input
            type="number"
            min="8"
            max="30"
            step="0.1"
            value={conditions.respiratory_rate}
            onChange={(e) =>
              update("respiratory_rate", parseFloat(e.target.value))
            }
          />
        </div>

        <div className="form-row">
          <label>Respiratory energy</label>
          <input
            type="number"
            min="0"
            max="0.3"
            step="0.001"
            value={conditions.respiratory_energy}
            onChange={(e) =>
              update("respiratory_energy", parseFloat(e.target.value))
            }
          />
        </div>

        <div className="form-row">
          <label>Respiratory variability</label>
          <input
            type="number"
            min="0"
            max="3"
            step="0.01"
            value={conditions.respiratory_variability}
            onChange={(e) =>
              update("respiratory_variability", parseFloat(e.target.value))
            }
          />
        </div>

        <div className="form-row">
          <label>Skin temperature (°C)</label>
          <input
            type="number"
            min="28"
            max="36"
            step="0.1"
            value={conditions.skin_temp_c}
            onChange={(e) =>
              update("skin_temp_c", parseFloat(e.target.value))
            }
          />
        </div>
      </div>

      <div className="form-actions">
        <button
          className="secondary-button"
          onClick={onReset}
          disabled={loading}
        >
          Reset
        </button>
        <button
          className="primary-button"
          onClick={onPredict}
          disabled={loading}
        >
          {loading ? "Predicting…" : "Predict state"}
        </button>
      </div>

      {lastResult && (
        <div className="result-block">
          <div className="result-state">{lastResult.state}</div>
          <div className="result-prob">
            confidence: {(lastResult.confidence * 100).toFixed(1)}%
          </div>

          <div className="result-prob-dist">
            {Object.entries(lastResult.probabilities).map(([name, p]) => (
              <div key={name} className="prob-row">
                <span>{name}</span>
                <span className="prob-value">{(p * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>

          <div className="result-note">
            calibrated against {lastResult.calibration.n_baseline} baseline readings
          </div>
        </div>
      )}
    </div>
  );
}