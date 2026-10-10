import { useState, useEffect } from 'react'
import Header from './components/Header'
import HostPanel from './components/HostPanel'
import DecisionPanel from './components/DecisionPanel'
import GuestPanel from './components/GuestPanel'
import Footer from './components/Footer'
import * as api from './api'

export default function App() {
  const [mode, setMode] = useState('baseline')
  const [calibration, setCalibration] = useState(null)
  const [result, setResult] = useState(null)
  const [hapticPattern, setHapticPattern] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [lastFeatures, setLastFeatures] = useState(null)

  // Verify the API is reachable on load
  useEffect(() => {
    api.health().catch((e) => {
      setError(`Cannot reach API: ${e.message}`)
    })
  }, [])

  async function handleCalibrate(currentMode) {
    setError(null)
    try {
      // Build 5 simulated baseline windows
      const baseline = Array.from({ length: 5 }, () => {
        // Slightly jittered around the baseline mean
        return [
          0.30 + (Math.random() - 0.5) * 0.02,
          0.05 + (Math.random() - 0.5) * 0.005,
          0.005 + (Math.random() - 0.5) * 0.0005,
          33.0 + (Math.random() - 0.5) * 0.2,
        ]
      })
      const cal = await api.calibrate(baseline)
      setCalibration(cal)
      setError(null)
    } catch (e) {
      setError(`Calibration failed: ${e.message}`)
    }
  }

  async function handleSendWindow(features) {
    setError(null)
    setLoading(true)
    setLastFeatures(features)
    try {
      const res = await api.predict(features, calibration)
      setResult(res)

      // Chain into haptic generation
      const pattern = await api.generateHaptic(res.stress_probability)
      setHapticPattern(pattern)
    } catch (e) {
      setError(`Prediction failed: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <Header />

      <div className="panels">
        <HostPanel
          mode={mode}
          setMode={setMode}
          onCalibrate={handleCalibrate}
          onSendWindow={handleSendWindow}
          lastFeatures={lastFeatures}
        />
        <DecisionPanel
          result={result}
          error={error}
          loading={loading}
        />
        <GuestPanel pattern={hapticPattern} />
      </div>

      <Footer />
    </div>
  )
}