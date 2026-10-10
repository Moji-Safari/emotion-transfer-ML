import { useEffect, useState } from 'react'

export default function GuestPanel({ pattern }) {
  const [animating, setAnimating] = useState(false)

  // Trigger a pulse animation whenever a new pattern arrives
  useEffect(() => {
    if (!pattern) return
    setAnimating(true)
    const total = pattern.duration_ms
    const t = setTimeout(() => setAnimating(false), total)
    return () => clearTimeout(t)
  }, [pattern])

  if (!pattern) {
    return (
      <div className="panel">
        <div className="panel-title">Guest Device · Haptic</div>
        <div className="empty">
          No haptic pattern yet.
          <br />
          Send a window from the host to generate one.
        </div>
      </div>
    )
  }

  // Render each pulse as a vertical bar whose height encodes intensity.
  const maxIntensity = Math.max(...pattern.pulses.map((p) => p.intensity))

  return (
    <div className="panel">
      <div className="panel-title">Guest Device · Haptic</div>

      <div className="pulse-visualizer">
        {pattern.pulses.map((pulse, i) => {
          const heightPct = (pulse.intensity / maxIntensity) * 100
          return (
            <div
              key={i}
              className="pulse-bar"
              style={{
                height: animating ? `${heightPct}%` : '4px',
                opacity: animating ? 1 : 0.35,
              }}
            />
          )
        })}
      </div>

      <div className="band-display">
        <div className="band-name">{pattern.band}</div>
        <div className="band-desc">{pattern.description}</div>
      </div>

      <div className="pulse-meta">
        <div className="meta-item">
          <div className="meta-label">Frequency</div>
          <div className="meta-value">
            {pattern.frequency_hz.toFixed(2)} Hz
          </div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Duration</div>
          <div className="meta-value">{pattern.duration_ms} ms</div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Intensity</div>
          <div className="meta-value">
            {pattern.intensity.toFixed(2)}
          </div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Pulses</div>
          <div className="meta-value">{pattern.pulses.length}</div>
        </div>
      </div>

      <button
        disabled={!animating}
        onClick={() => {
          if ('vibrate' in navigator) {
            // Use the real haptic API if available (mobile)
            const vibPattern = pattern.pulses
              .sort((a, b) => a.t_ms - b.t_ms)
              .map((p, i, arr) => {
                const next = arr[i + 1]
                const gap = next
                  ? next.t_ms - p.t_ms
                  : 200
                return [Math.round(gap * p.intensity), gap]
              })
              .flat()
            navigator.vibrate(vibPattern)
          } else {
            // Fallback: visual feedback only
            setAnimating(false)
            setTimeout(() => setAnimating(true), 50)
          }
        }}
      >
        {animating ? 'Playing…' : 'Replay Pattern'}
      </button>
    </div>
  )
}