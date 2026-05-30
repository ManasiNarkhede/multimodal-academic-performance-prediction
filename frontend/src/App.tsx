import { useEffect } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'

import { useUIStore } from '@/store/uiStore'
import LoginPage from '@/pages/LoginPage'
import CohortDashboard from '@/pages/CohortDashboard'
import ModelPerformancePage from '@/pages/ModelPerformancePage'

export default function App() {
  const theme = useUIStore((state) => state.theme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.documentElement.style.colorScheme = theme
  }, [theme])

  return (
    <div data-theme={theme} className="min-h-screen">
      <Routes>
        <Route path="/" element={<Navigate to="/login" replace />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/dashboard" element={<Navigate to="/cohorts/1" replace />} />
        <Route path="/cohorts/:id" element={<CohortDashboard />} />
        <Route path="/models" element={<ModelPerformancePage />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </div>
  )
}
