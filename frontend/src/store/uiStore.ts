import { create } from 'zustand'

export type Theme = 'dark' | 'light'

type UIStore = {
  sidebarOpen: boolean
  theme: Theme
  setSidebarOpen: (sidebarOpen: boolean) => void
  toggleSidebar: () => void
  setTheme: (theme: Theme) => void
}

export const useUIStore = create<UIStore>((set) => ({
  sidebarOpen: false,
  theme: 'dark',
  setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setTheme: (theme) => set({ theme }),
}))
