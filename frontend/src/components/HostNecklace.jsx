/**
 * Host device: a minimal circular pendant necklace.
 * The yellow waves radiating outward represent wireless transfer
 * of the detected affective state to the guest device.
 */

const STATE_WAVE_SPEED = {
  baseline:   4.0,   // slow
  meditation: 3.5,
  amusement:  1.5,
  stress:     0.8,   // fast
};

const STATE_COLORS = {
  baseline:   '#00704A',
  meditation: '#2E7D8A',
  amusement:  '#E8B84B',
  stress:     '#B53D3D',
};

export default function HostNecklace({ state }) {
  const speed = STATE_WAVE_SPEED[state] ?? 2;
  const stateColor = STATE_COLORS[state] ?? '#00704A';

  // Apply animation speed to wave elements via inline style
  const waveStyle = { animationDuration: `${speed}s` };

  return (
    <div className="panel">
      <div className="panel-title">Host · Necklace</div>

      <div className="necklace-stage">
        <svg
          className="necklace-svg"
          viewBox="0 0 300 400"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Chain */}
          <path
            d="M 80 30 Q 50 120 60 180 Q 70 230 150 250 Q 230 230 240 180 Q 250 120 220 30"
            fill="none"
            stroke="#1A1A1A"
            strokeWidth="1"
            strokeDasharray="2 4"
            opacity="0.4"
          />

          {/* Pendant circle (blueprint style) */}
          <circle
            cx="150"
            cy="280"
            r="42"
            fill="white"
            stroke="#1A1A1A"
            strokeWidth="1.2"
          />
          <circle
            cx="150"
            cy="280"
            r="30"
            fill="white"
            stroke={stateColor}
            strokeWidth="1"
          />
          <circle
            cx="150"
            cy="280"
            r="6"
            fill={stateColor}
          />

          {/* Wi-Fi waves — CSS-animated yellow arcs */}
          <g transform="translate(150, 280)">
            <g className="wave-1" style={waveStyle}>
              <path className="wave-path" d="M -15 -15 Q 0 -30 15 -15" />
            </g>
            <g className="wave-2" style={waveStyle}>
              <path className="wave-path" d="M -25 -25 Q 0 -45 25 -25" />
            </g>
            <g className="wave-3" style={waveStyle}>
              <path className="wave-path" d="M -35 -35 Q 0 -60 35 -35" />
            </g>
            <g className="wave-4" style={waveStyle}>
              <path className="wave-path" d="M -45 -45 Q 0 -75 45 -45" />
            </g>
          </g>

          {/* Blueprint labels */}
          <text
            x="150"
            y="360"
            textAnchor="middle"
            fontSize="9"
            fill="#6B6B6B"
            letterSpacing="2"
          >
            PENDANT · Ø 60 MM
          </text>
        </svg>
      </div>

      <div className="state-badge">
        <div className="label">Detected state</div>
        <div className="value" style={{ color: stateColor }}>{state}</div>
        <div className="rate">transmitting · {speed.toFixed(1)} s cycle</div>
      </div>
    </div>
  );
}