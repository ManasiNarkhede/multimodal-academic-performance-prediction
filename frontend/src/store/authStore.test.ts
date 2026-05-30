import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useAuthStore } from './authStore'
import * as api from '@/lib/api'

vi.mock('@/lib/api', () => ({
  login: vi.fn(),
  logout: vi.fn(),
}))

describe('authStore', () => {
  beforeEach(() => {
    useAuthStore.setState({
      user: null,
      isAuthenticated: false,
    })
    vi.clearAllMocks()
  })

  it('should have correct initial state', () => {
    const state = useAuthStore.getState()
    expect(state.user).toBeNull()
    expect(state.isAuthenticated).toBe(false)
  })

  it('should login successfully', async () => {
    const mockResponse = { message: 'Login successful' }
    vi.mocked(api.login).mockResolvedValue(mockResponse)

    const state = useAuthStore.getState()
    const result = await state.login({ email: 'test@example.com', password: 'password123' })

    expect(api.login).toHaveBeenCalledWith({ email: 'test@example.com', password: 'password123' })
    expect(result).toEqual(mockResponse)

    const newState = useAuthStore.getState()
    expect(newState.user).toEqual({ email: 'test@example.com' })
    expect(newState.isAuthenticated).toBe(true)
  })

  it('should logout successfully', async () => {
    vi.mocked(api.logout).mockResolvedValue({ message: 'Logout successful' })

    useAuthStore.setState({
      user: { email: 'test@example.com' },
      isAuthenticated: true,
    })

    const state = useAuthStore.getState()
    await state.logout()

    expect(api.logout).toHaveBeenCalled()

    const newState = useAuthStore.getState()
    expect(newState.user).toBeNull()
    expect(newState.isAuthenticated).toBe(false)
  })

  it('should clear state even if logout request fails', async () => {
    vi.mocked(api.logout).mockRejectedValue(new Error('Network error'))

    useAuthStore.setState({
      user: { email: 'test@example.com' },
      isAuthenticated: true,
    })

    const state = useAuthStore.getState()
    await expect(state.logout()).rejects.toThrow('Network error')

    const newState = useAuthStore.getState()
    expect(newState.user).toBeNull()
    expect(newState.isAuthenticated).toBe(false)
  })

  it('should set user directly', () => {
    const state = useAuthStore.getState()
    state.setUser({ email: 'user@example.com' })

    const newState = useAuthStore.getState()
    expect(newState.user).toEqual({ email: 'user@example.com' })
    expect(newState.isAuthenticated).toBe(true)
  })

  it('should set isAuthenticated to false when setting user to null', () => {
    useAuthStore.setState({
      user: { email: 'test@example.com' },
      isAuthenticated: true,
    })

    const state = useAuthStore.getState()
    state.setUser(null)

    const newState = useAuthStore.getState()
    expect(newState.user).toBeNull()
    expect(newState.isAuthenticated).toBe(false)
  })
})
