const generateMockConditions = () => {
  const hr = 60 + Math.random() * 40;             // 60-100 bpm
  const rmssd = 20 + Math.random() * 50;          // 20-70 ms
  const sdnn = 40 + Math.random() * 60;           // 40-100 ms
  const rr = 12 + Math.random() * 12;             // 12-24 breaths/min
  const re = 0.005 + Math.random() * 0.08;        // 0.005-0.085
  const rv = 0.1 + Math.random() * 1.2;           // 0.1-1.3
  const temp = 31 + Math.random() * 3.5;          // 31-34.5 °C
  return {
    heart_rate: Math.round(hr),
    hrv_rmssd: Math.round(rmssd * 10) / 10,
    hrv_sdnn: Math.round(sdnn * 10) / 10,
    respiratory_rate: Math.round(rr * 10) / 10,
    respiratory_energy: Math.round(re * 1000) / 1000,
    respiratory_variability: Math.round(rv * 100) / 100,
    skin_temp_c: Math.round(temp * 10) / 10,
  };
};

export default function HumanMock({ onGenerate, onReset }) {
  return (
    <div className="panel mock-panel">
      <div className="panel-title">Mock Subject</div>

      <div className="mock-figure">
        <svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg">
          <circle cx="100" cy="50" r="28"
                  fill="white" stroke="#1A1A1A" strokeWidth="1.2" />
          <path d="M 72 80 Q 100 75 128 80 L 132 190 Q 100 200 68 190 Z"
                fill="white" stroke="#1A1A1A" strokeWidth="1.2" />
          <path d="M 72 90 Q 50 110 42 160"
                fill="none" stroke="#1A1A1A" strokeWidth="1.2" strokeLinecap="round" />
          <path d="M 128 90 Q 150 110 158 160"
                fill="none" stroke="#1A1A1A" strokeWidth="1.2" strokeLinecap="round" />
          <path d="M 85 195 Q 80 240 78 280"
                fill="none" stroke="#1A1A1A" strokeWidth="1.2" strokeLinecap="round" />
          <path d="M 115 195 Q 120 240 122 280"
                fill="none" stroke="#1A1A1A" strokeWidth="1.2" strokeLinecap="round" />
          <line x1="140" y1="50" x2="180" y2="50"
                stroke="#6B6B6B" strokeWidth="0.5" strokeDasharray="2 2" />
          <text x="182" y="52" fontSize="8" fill="#6B6B6B">SUBJECT</text>
          <line x1="132" y1="140" x2="172" y2="140"
                stroke="#6B6B6B" strokeWidth="0.5" strokeDasharray="2 2" />
          <text x="174" y="142" fontSize="8" fill="#6B6B6B">EDA · TEMP</text>
        </svg>
      </div>

      <div className="mock-buttons">
        <button
          className="primary-button"
          onClick={() => onGenerate(generateMockConditions())}
        >
          Generate mock
        </button>
        <button
          className="secondary-button"
          onClick={onReset}
        >
          Reset
        </button>
      </div>

      <div className="mock-note">
        Generates randomized but physiologically plausible body conditions.
      </div>
    </div>
  );
}