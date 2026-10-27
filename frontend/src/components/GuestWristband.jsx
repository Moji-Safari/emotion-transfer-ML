/**
 * Guest device: a minimal wristband with four pulsing haptic actuators.
 *
 * The pulse rate is based on affective haptics literature:
 *   - baseline    → slow, gentle (calming)
 *   - meditation  → very slow, smooth (parasympathetic)
 *   - amusement   → medium (positive high arousal)
 *   - stress      → fast, strong (sympathetic activation)
 *
 * Hz = cycles per second. Duration = 1/Hz seconds per pulse.
 */

const STATE_PULSE_HZ = {
  baseline:   0.3,
  meditation: 0.5,
  amusement:  2.0,
  stress:     6.0,
};

const STATE_LABEL = {
  baseline:   'slow · gentle',
  meditation: 'very slow · smooth',
  amusement:  'medium · rhythmic',
  stress:     'fast · strong',
};

const STATE_COLOR = {
  baseline:   '#1F3A5F',
  meditation: '#1F3A5F',
  amusement:  '#1F3A5F',
  stress:     '#1F3A5F',
};

export default function GuestWristband({ state }) {
  const hz = STATE_PULSE_HZ[state] ?? 0.3;
  const durationSec = 1 / hz;

  // Animation keyframes for CSS
  const pulseStyle = {
    animation: `pulse-rate ${durationSec}s ease-in-out infinite`,
  };

  return (
    <div className="panel">
      <div className="panel-title">Guest · Wristband</div>

      <div className="wristband-stage">
        <svg
          className="wristband-svg"
          viewBox="0 0 480 200"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Wristband body (blueprint style) */}
          <rect
            x="40" y="80" width="400" height="40"
            rx="20"
            fill="white"
            stroke="#1A1A1A"
            strokeWidth="1.2"
          />

          {/* Blueprint dashed center line */}
          <line
            x1="60" y1="100" x2="420" y2="100"
            stroke="#1A1A1A"
            strokeWidth="0.5"
            strokeDasharray="2 4"
            opacity="0.3"
          />

          {/* Four pulsing actuator circles (navy) */}
          {[130, 210, 290, 370].map((cx, i) => (
            <g key={i}>
              <circle
                cx={cx} cy={100} r="12"
                fill="#1F3A5F"
                opacity="0.15"
              />
              <circle
                cx={cx} cy={100} r="8"
                fill={STATE_COLOR[state]}
                className="pulse-circle"
                style={pulseStyle}
              />
            </g>
          ))}

          {/* Blueprint labels */}
          <text
            x="240" y="50"
            textAnchor="middle"
            fontSize="9"
            fill="#6B6B6B"
            letterSpacing="2"
          >
            4 LRAs · SPACED 80 MM
          </text>
          <text
            x="240" y="160"
            textAnchor="middle"
            fontSize="9"
            fill="#6B6B6B"
            letterSpacing="2"
          >
            BAND · 18 MM WIDTH
          </text>
        </svg>
      </div>

      <div className="state-badge">
        <div className="label">Haptic pattern</div>
        <div className="value">{hz.toFixed(1)} Hz</div>
        <div className="rate">{STATE_LABEL[state]}</div>
      </div>
    </div>
  );
}