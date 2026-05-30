import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  LoaderCircle,
  Lock,
  Mail,
  ShieldCheck,
} from 'lucide-react'
import { type ChangeEvent, type FormEvent, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { ApiError, login as loginRequest } from '@/lib/api'
import { useAuthStore } from '@/store/authStore'

type LoginFormValues = {
  email: string
  password: string
}

type LoginFormErrors = Partial<Record<keyof LoginFormValues, string>>

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

const formPanelClasses =
  'rounded-[2rem] border border-slate-800 bg-slate-900/80 p-6 shadow-2xl shadow-slate-950/50 backdrop-blur sm:p-8 xl:p-10'
const inputClasses =
  'w-full rounded-2xl border border-slate-800 bg-slate-950/80 py-3.5 pl-11 pr-4 text-sm text-slate-100 placeholder:text-slate-500 shadow-inner shadow-slate-950/40 transition focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/20'
const helperCardClasses =
  'rounded-3xl border border-slate-800/80 bg-slate-900/50 p-5 shadow-xl shadow-slate-950/30 backdrop-blur'

function validateForm(values: LoginFormValues): LoginFormErrors {
  const errors: LoginFormErrors = {}

  if (!values.email.trim()) {
    errors.email = 'Email is required.'
  } else if (!EMAIL_PATTERN.test(values.email.trim())) {
    errors.email = 'Enter a valid email address.'
  }

  if (!values.password) {
    errors.password = 'Password is required.'
  }

  return errors
}

function getErrorMessage(error: unknown) {
  if (error instanceof ApiError) {
    return error.message
  }

  if (error instanceof Error) {
    return error.message
  }

  return 'Unable to sign in right now. Please try again.'
}

export default function LoginPage() {
  const navigate = useNavigate()
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated)
  const setUser = useAuthStore((state) => state.setUser)

  const [values, setValues] = useState<LoginFormValues>({
    email: '',
    password: '',
  })
  const [errors, setErrors] = useState<LoginFormErrors>({})
  const [apiError, setApiError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard', { replace: true })
    }
  }, [isAuthenticated, navigate])

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target

    setValues((current) => ({
      ...current,
      [name]: value,
    }))

    if (errors[name as keyof LoginFormValues]) {
      setErrors((current) => ({
        ...current,
        [name]: undefined,
      }))
    }

    if (apiError) {
      setApiError('')
    }
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    const nextErrors = validateForm(values)

    if (Object.keys(nextErrors).length > 0) {
      setErrors(nextErrors)
      return
    }

    setErrors({})
    setApiError('')
    setIsSubmitting(true)

    try {
      await loginRequest({
        email: values.email.trim(),
        password: values.password,
      })

      setUser({ email: values.email.trim() })
      navigate('/dashboard', { replace: true })
    } catch (error) {
      setApiError(getErrorMessage(error))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="relative min-h-screen overflow-hidden bg-slate-950 text-slate-50">
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -left-12 top-0 h-72 w-72 rounded-full bg-cyan-500/10 blur-3xl" />
        <div className="absolute right-0 top-20 h-96 w-96 rounded-full bg-sky-400/10 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-80 w-80 rounded-full bg-slate-700/20 blur-3xl" />
      </div>

      <div className="relative mx-auto grid min-h-screen max-w-6xl items-center gap-10 px-6 py-10 lg:grid-cols-[1.1fr_0.9fr] lg:px-10 lg:py-12">
        <section className="flex flex-col gap-8">
          <div className="inline-flex w-fit items-center gap-3 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-4 py-2 text-xs font-medium uppercase tracking-[0.3em] text-cyan-300">
            <ShieldCheck className="h-4 w-4" />
            Admin access
          </div>

          <div className="max-w-2xl space-y-5">
            <p className="text-sm font-medium uppercase tracking-[0.3em] text-slate-400">
              Academic Performance Prediction System
            </p>
            <h1 className="text-4xl font-semibold leading-tight text-white sm:text-5xl">
              Enter the command center for cohort intelligence.
            </h1>
            <p className="max-w-xl text-base leading-7 text-slate-300 sm:text-lg">
              Review high-risk learners, inspect prediction narratives, and coordinate intervention work from a
              focused slate-dark workspace designed for fast decisions.
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <div className={helperCardClasses}>
              <p className="text-xs font-medium uppercase tracking-[0.3em] text-slate-500">Security</p>
              <p className="mt-3 text-sm leading-6 text-slate-200">Cookie-based session login through the protected API gateway.</p>
            </div>
            <div className={helperCardClasses}>
              <p className="text-xs font-medium uppercase tracking-[0.3em] text-slate-500">Response</p>
              <p className="mt-3 text-sm leading-6 text-slate-200">Direct error feedback and a clear loading state keep the sign-in flow predictable.</p>
            </div>
            <div className={helperCardClasses}>
              <p className="text-xs font-medium uppercase tracking-[0.3em] text-slate-500">Focus</p>
              <p className="mt-3 text-sm leading-6 text-slate-200">A single secure entry point aligned with the existing dashboard dark theme.</p>
            </div>
          </div>

          <div className="flex flex-col gap-3 text-sm text-slate-300">
            {[
              'Monitor cohort-level student risk patterns.',
              'Review model outputs with explainable context.',
              'Move directly into the dashboard after authentication.',
            ].map((item) => (
              <div key={item} className="flex items-start gap-3">
                <CheckCircle2 className="mt-0.5 h-5 w-5 flex-none text-cyan-400" />
                <span className="leading-6">{item}</span>
              </div>
            ))}
          </div>
        </section>

        <section className={formPanelClasses}>
          <div className="mb-8 space-y-3">
            <p className="text-sm font-medium uppercase tracking-[0.3em] text-cyan-400">Secure sign in</p>
            <h2 className="text-3xl font-semibold text-white">Welcome back, administrator.</h2>
            <p className="text-sm leading-6 text-slate-300">
              Sign in with your institutional account to continue to the analytics dashboard.
            </p>
          </div>

          <form className="space-y-5" noValidate onSubmit={handleSubmit}>
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-200" htmlFor="email">
                Email address
              </label>
              <div className="relative">
                <Mail className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <input
                  autoComplete="email"
                  className={inputClasses}
                  id="email"
                  name="email"
                  onChange={handleChange}
                  placeholder="admin@institution.edu"
                  type="email"
                  value={values.email}
                  aria-invalid={Boolean(errors.email)}
                  aria-describedby={errors.email ? 'email-error' : undefined}
                />
              </div>
              {errors.email ? (
                <p id="email-error" className="text-sm text-rose-300">
                  {errors.email}
                </p>
              ) : null}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-200" htmlFor="password">
                Password
              </label>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <input
                  autoComplete="current-password"
                  className={inputClasses}
                  id="password"
                  name="password"
                  onChange={handleChange}
                  placeholder="Enter your password"
                  type="password"
                  value={values.password}
                  aria-invalid={Boolean(errors.password)}
                  aria-describedby={errors.password ? 'password-error' : undefined}
                />
              </div>
              {errors.password ? (
                <p id="password-error" className="text-sm text-rose-300">
                  {errors.password}
                </p>
              ) : null}
            </div>

            {apiError ? (
              <div className="flex items-start gap-3 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                <AlertCircle className="mt-0.5 h-4 w-4 flex-none" />
                <span className="leading-6">{apiError}</span>
              </div>
            ) : null}

            <button
              className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-cyan-400 px-4 py-3.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:cursor-not-allowed disabled:bg-cyan-400/60"
              disabled={isSubmitting}
              type="submit"
            >
              {isSubmitting ? (
                <>
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                  Signing in...
                </>
              ) : (
                <>
                  Continue to dashboard
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </button>
          </form>
        </section>
      </div>
    </main>
  )
}
