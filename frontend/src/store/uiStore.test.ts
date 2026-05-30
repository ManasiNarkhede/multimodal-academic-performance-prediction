import { describe, it, expect, beforeEach } from 'vitest'
import { useUIStore } from './uiStore'

describe('uiStore', () => {
  beforeEach(() => {
    useUIStore.setState({
      sidebarOpen: false,
      theme: 'dark',
    })
  })

  it('should have correct initial state', () => {
    const state = useUIStore.getState()
    expect(state.sidebarOpen).toBe(false)
    expect(state.theme).toBe('dark')
  })

  it('should set sidebar open state', () => {
    const state = useUIStore.getState()
    state.setSidebarOpen(true)

    const newState = useUIStore.getState()
    expect(newState.sidebarOpen).toBe(true)
  })

  it('should set sidebar closed state', () => {
    useUIStore.setState({ sidebarOpen: true })

    const state = useUIStore.getState()
    state.setSidebarOpen(false)

    const newState = useUIStore.getState()
    expect(newState.sidebarOpen).toBe(false)
  })

  it('should toggle sidebar from false to true', () => {
    const state = useUIStore.getState()
    state.toggleSidebar()

    const newState = useUIStore.getState()
    expect(newState.sidebarOpen).toBe(true)
  })

  it('should toggle sidebar from true to false', () => {
    useUIStore.setState({ sidebarOpen: true })

    const state = useUIStore.getState()
    state.toggleSidebar()

    const newState = useUIStore.getState()
    expect(newState.sidebarOpen).toBe(false)
  })

  it('should set theme to light', () => {
    const state = useUIStore.getState()
    state.setTheme('light')

    const newState = useUIStore.getState()
    expect(newState.theme).toBe('light')
  })

  it('should set theme to dark', () => {
    useUIStore.setState({ theme: 'light' })

    const state = useUIStore.getState()
    state.setTheme('dark')

    const newState = useUIStore.getState()
    expect(newState.theme).toBe('dark')
  })
})
