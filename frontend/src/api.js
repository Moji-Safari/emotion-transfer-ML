// Thin client for the AffectWave Flask API.

const BASE = 'http://localhost:5000'

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'unknown' }))
    throw new Error(err.error || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function health() {
  const res = await fetch(`${BASE}/health`)
  return res.json()
}

export async function calibrate(baselineFeatures) {
  return post('/calibrate', { baseline_features: baselineFeatures })
}

export async function predict(features, calibration) {
  return post('/predict', { features, calibration })
}

export async function generateHaptic(stressProbability) {
  return post('/generate-haptic', { stress_probability: stressProbability })
}