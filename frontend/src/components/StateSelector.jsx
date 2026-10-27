const STATES = ['baseline', 'stress', 'amusement', 'meditation'];

export default function StateSelector({ state, setState }) {
  return (
    <div className="selector-panel">
      <div className="selector-label">Select detected state</div>
      <div className="selector-buttons">
        {STATES.map((s) => (
          <button
            key={s}
            className={state === s ? 'active' : ''}
            onClick={() => setState(s)}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}