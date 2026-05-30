import { create } from 'zustand'

import {
  type LoginCredentials,
  type LoginResponse,
  login as loginRequest,
  logout as logoutRequest,
} from '@/lib/api'

export type AuthUser = {
  email: string
}

type AuthStore = {
  user: AuthUser | null
  isAuthenticated: boolean
  login: (credentials: LoginCredentials) => Promise<LoginResponse>
  logout: () => Promise<void>
  setUser: (user: AuthUser | null) => void
}

export const useAuthStore = create<AuthStore>((set) => ({
  user: null,
  isAuthenticated: false,
  login: async (credentials) => {
    const response = await loginRequest(credentials)

    set({
      user: { email: credentials.email },
      isAuthenticated: true,
    })

    return response
  },
  logout: async () => {
    try {
      await logoutRequest()
    } finally {
      set({
        user: null,
        isAuthenticated: false,
      })
    }
  },
  setUser: (user) => {
    set({
      user,
      isAuthenticated: user !== null,
    })
  },
}))
