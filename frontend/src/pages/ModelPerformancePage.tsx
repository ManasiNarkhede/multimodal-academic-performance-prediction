import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  AlertCircle,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Cpu,
  Gauge,
  RefreshCw,
  ShieldAlert,
  Sparkles,
} from 'lucide-react'
import type { CSSProperties, ReactNode } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { getHealth, getModels, type HealthResponse, type ModelsResponse } from '@/lib/api'

const REFRESH_INTERVAL_MS = 30_000

type DashboardModel = ModelsResponse['models'][number] & {
  accuracy?: number | null
  f1_score?: number | null
}

const pageTheme = {
  '--dashboard-accent': '#22d3ee',
  '--dashboard-accent-soft': 'rgba(34, 211, 238, 0.14)',
  '--dashboard-success': '#22c55e',
  '--dashboard-danger': '#ef4444',
  '--dashboard-slate-line': 'rgba(148, 163, 184, 0.18)',
  '--dashboard-tooltip-bg': 'rgba(2, 6, 23, 0.96)',
} as CSSProperties

const surfaceClass =
  'rounded-3xl border border-slate-800/80 bg-slate-900/75 shadow-[0_24px_80px_rgba(2,6,23,0.45)] backdrop-blur'

const panelClass = `${surfaceClass} p-6`

const badgeClassByStatus: Record<string, string> = {
  active: 'border border-emerald-500/30 bg-emerald-500/12 text-emerald-300',
  deprecated: 'border border-slate-600/50 bg-slate-700/40 text-slate-300',
  experimental: 'border border-sky-500/30 bg-sky-500/12 text-sky-300',
}

function formatPercent(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—'
  }

  return `${(value * 100).toFixed(1)}%`
}

function formatShortPercent(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—'
  }

  return `${Math.round(value * 100)}%`
}

function formatDate(value?: string | null) {
  if (!value) {
    return 'Not available'
  }

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return 'Not available'
  }

  return new Intl.DateTimeFormat('en', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

function formatStatus(value?: string) {
  if (!value) {
    return 'Unknown'
  }

  return value
    .split(/[_-]/g)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function getSystemHealth(health?: HealthResponse) {
  if (!health) {
    return false
  }

  const checks = [health.status, health.db, health.models].filter(Boolean)
  return checks.every((value) => value === 'ok')
}

function getStatusBadgeClass(status: string) {
  return badgeClassByStatus[status] ?? 'border border-slate-700 bg-slate-800/70 text-slate-200'
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.28em] ${getStatusBadgeClass(status)}`}
    >
      {formatStatus(status)}
    </span>
  )
}

function HealthDot({ healthy }: { healthy: boolean }) {
  return (
    <span className="relative flex h-3 w-3 shrink-0">
      <span
        className={`absolute inline-flex h-full w-full rounded-full opacity-75 ${healthy ? 'bg-emerald-400' : 'bg-rose-400'}`}
      />
      <span className={`relative inline-flex h-3 w-3 rounded-full ${healthy ? 'bg-emerald-300' : 'bg-rose-300'}`} />
    </span>
  )
}

function Panel({ title, eyebrow, action, children }: { title: string; eyebrow?: string; action?: ReactNode; children: ReactNode }) {
  return (
    <section className={panelClass}>
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-2">
          {eyebrow ? (
            <p className="text-xs font-semibold uppercase tracking-[0.32em] text-cyan-300/85">{eyebrow}</p>
          ) : null}
          <h2 className="text-lg font-semibold text-white">{title}</h2>
        </div>
        {action}
      </div>
      {children}
    </section>
  )
}

function MetricCard({
  icon,
  label,
  value,
  detail,
}: {
  icon: ReactNode
  label: string
  value: string
  detail: string
}) {
  return (
    <div className={`${surfaceClass} relative overflow-hidden p-5`}>
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-300/35 to-transparent" />
      <div className="mb-5 flex h-11 w-11 items-center justify-center rounded-2xl bg-cyan-400/10 text-cyan-300">
        {icon}
      </div>
      <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">{label}</p>
      <p className="mt-3 text-3xl font-semibold text-white">{value}</p>
      <p className="mt-2 text-sm leading-6 text-slate-400">{detail}</p>
    </div>
  )
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-3 rounded-2xl border border-rose-500/25 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  )
}

function LoadingGrid() {
  const skeletonKeys = ['models', 'active', 'accuracy', 'last-trained']

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {skeletonKeys.map((key) => (
        <div key={key} className={`${surfaceClass} animate-pulse p-5`}>
          <div className="h-11 w-11 rounded-2xl bg-slate-800" />
          <div className="mt-5 h-3 w-28 rounded-full bg-slate-800" />
          <div className="mt-4 h-8 w-24 rounded-full bg-slate-700" />
          <div className="mt-3 h-3 w-full rounded-full bg-slate-800" />
        </div>
      ))}
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex min-h-64 items-center justify-center rounded-2xl border border-dashed border-slate-700/70 bg-slate-950/35 px-6 text-center text-sm text-slate-400">
      {message}
    </div>
  )
}

export default function ModelPerformancePage() {
  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
    refetchInterval: REFRESH_INTERVAL_MS,
    refetchIntervalInBackground: true,
  })

  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: REFRESH_INTERVAL_MS,
    refetchIntervalInBackground: true,
  })

  const models = (modelsQuery.data?.models ?? []) as DashboardModel[]
  const health = healthQuery.data
  const healthy = getSystemHealth(health)

  const activeModels = models.filter((model) => model.status === 'active').length
  const bestAccuracy = models.reduce<number | null>((best, model) => {
    if (model.accuracy === null || model.accuracy === undefined) {
      return best
    }

    return best === null ? model.accuracy : Math.max(best, model.accuracy)
  }, null)

  const latestTrainingDate = models.reduce<string | null>((latest, model) => {
    if (!model.last_trained) {
      return latest
    }

    if (!latest) {
      return model.last_trained
    }

    return new Date(model.last_trained).getTime() > new Date(latest).getTime() ? model.last_trained : latest
  }, null)

  const chartData = models.map((model) => ({
    name: model.name,
    accuracy: Number(((model.accuracy ?? 0) * 100).toFixed(1)),
    version: model.version,
  }))

  const isInitialLoading = (!modelsQuery.data && modelsQuery.isPending) || (!healthQuery.data && healthQuery.isPending)

  return (
    <main
      style={pageTheme}
      className="min-h-screen overflow-hidden bg-slate-950 px-6 py-10 text-slate-50 sm:px-8 lg:px-10"
    >
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute left-0 top-0 h-72 w-72 rounded-full bg-cyan-400/10 blur-3xl" />
        <div className="absolute right-0 top-20 h-80 w-80 rounded-full bg-sky-500/10 blur-3xl" />
      </div>

      <div className="relative mx-auto flex max-w-7xl flex-col gap-8">
        <header className={`${surfaceClass} relative overflow-hidden p-8`}>
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-300/50 to-transparent" />
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl space-y-4">
              <p className="text-xs font-semibold uppercase tracking-[0.34em] text-cyan-300/90">
                Model intelligence desk
              </p>
              <div className="space-y-3">
                <h1 className="text-4xl font-semibold tracking-tight text-white sm:text-5xl">
                  Model Performance Dashboard
                </h1>
                <p className="max-w-2xl text-base leading-7 text-slate-300 sm:text-lg">
                  Monitor deployment readiness, compare accuracy, and verify service health for every model in the
                  prediction stack.
                </p>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-2xl border border-slate-800 bg-slate-950/55 px-4 py-3">
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">
                  <RefreshCw className={`h-3.5 w-3.5 ${modelsQuery.isFetching || healthQuery.isFetching ? 'animate-spin' : ''}`} />
                  Refresh cadence
                </div>
                <p className="mt-2 text-lg font-semibold text-white">30 seconds</p>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-950/55 px-4 py-3">
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">
                  <HealthDot healthy={healthy} />
                  System health
                </div>
                <p className="mt-2 text-lg font-semibold text-white">{healthy ? 'Operational' : 'Needs attention'}</p>
              </div>
            </div>
          </div>
        </header>

        {(modelsQuery.error || healthQuery.error) && (
          <div className="grid gap-3">
            {modelsQuery.error instanceof Error ? <ErrorBanner message={`Models request failed: ${modelsQuery.error.message}`} /> : null}
            {healthQuery.error instanceof Error ? <ErrorBanner message={`Health request failed: ${healthQuery.error.message}`} /> : null}
          </div>
        )}

        {isInitialLoading ? (
          <LoadingGrid />
        ) : (
          <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              icon={<BrainCircuit className="h-5 w-5" />}
              label="Models tracked"
              value={String(models.length)}
              detail="Live inventory of deployed and testable prediction models."
            />
            <MetricCard
              icon={<CheckCircle2 className="h-5 w-5" />}
              label="Active models"
              value={String(activeModels)}
              detail="Currently marked as production-ready and available for serving."
            />
            <MetricCard
              icon={<Gauge className="h-5 w-5" />}
              label="Best accuracy"
              value={formatPercent(bestAccuracy)}
              detail="Top reported accuracy score across the current model lineup."
            />
            <MetricCard
              icon={<Clock3 className="h-5 w-5" />}
              label="Last trained"
              value={latestTrainingDate ? formatDate(latestTrainingDate) : 'No history'}
              detail="Most recent training timestamp reported by the model metadata feed."
            />
          </section>
        )}

        <div className="grid gap-6 xl:grid-cols-[1.35fr_0.95fr]">
          <Panel title="Accuracy comparison" eyebrow="Performance snapshot">
            {models.length > 0 ? (
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 12, right: 12, left: -18, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke="var(--dashboard-slate-line)" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="name"
                      tickLine={false}
                      axisLine={false}
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                    />
                    <YAxis
                      tickLine={false}
                      axisLine={false}
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      domain={[0, 100]}
                      tickFormatter={(value) => `${value}%`}
                    />
                    <Tooltip
                      cursor={{ fill: 'rgba(148, 163, 184, 0.08)' }}
                      contentStyle={{
                        backgroundColor: 'var(--dashboard-tooltip-bg)',
                        border: '1px solid rgba(34, 211, 238, 0.16)',
                        borderRadius: '16px',
                        boxShadow: '0 24px 48px rgba(2, 6, 23, 0.45)',
                      }}
                      labelStyle={{ color: '#e2e8f0', fontWeight: 600 }}
                      labelFormatter={(label, payload) => {
                        const version = payload?.[0]?.payload?.version
                        return version ? `${label} · ${version}` : label
                      }}
                      formatter={(value: unknown) => {
                        if (typeof value === 'number') {
                          return [`${value}%`, 'Accuracy']
                        }

                        if (Array.isArray(value)) {
                          return [value.join(' / '), 'Accuracy']
                        }

                        return [typeof value === 'string' ? value : '—', 'Accuracy']
                      }}
                    />
                    <Bar dataKey="accuracy" radius={[14, 14, 4, 4]} fill="var(--dashboard-accent)" maxBarSize={56} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState message="No model metrics are available yet. Once /models returns accuracy data, the chart will populate automatically." />
            )}
          </Panel>

          <Panel
            title="Service health"
            eyebrow="Operational watch"
            action={
              <div className="inline-flex items-center gap-2 rounded-full border border-slate-700/70 bg-slate-950/55 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.24em] text-slate-300">
                <HealthDot healthy={healthy} />
                {healthy ? 'Healthy' : 'Degraded'}
              </div>
            }
          >
            <div className="grid gap-3">
              <div className="rounded-2xl border border-slate-800/80 bg-slate-950/45 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">API status</p>
                    <p className="mt-2 text-lg font-semibold text-white">{formatStatus(health?.status)}</p>
                  </div>
                  <Activity className={`h-5 w-5 ${healthy ? 'text-emerald-300' : 'text-rose-300'}`} />
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-slate-800/80 bg-slate-950/45 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">Database</p>
                  <p className="mt-2 text-lg font-semibold text-white">{formatStatus(health?.db)}</p>
                </div>
                <div className="rounded-2xl border border-slate-800/80 bg-slate-950/45 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">Model service</p>
                  <p className="mt-2 text-lg font-semibold text-white">{formatStatus(health?.models)}</p>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800/80 bg-slate-950/45 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">Memory usage</p>
                    <p className="mt-2 text-lg font-semibold text-white">
                      {health?.memory_mb !== undefined ? `${health.memory_mb} MB` : 'Not reported'}
                    </p>
                  </div>
                  <Cpu className="h-5 w-5 text-cyan-300" />
                </div>
              </div>
            </div>
          </Panel>
        </div>

        <Panel
          title="Model registry"
          eyebrow="Metadata overview"
          action={
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/20 bg-cyan-400/10 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.24em] text-cyan-200">
              <Sparkles className="h-3.5 w-3.5" />
              Auto-refresh enabled
            </div>
          }
        >
          {models.length > 0 ? (
            <div className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
              {models.map((model) => (
                <article key={`${model.name}-${model.version}`} className="rounded-2xl border border-slate-800/80 bg-slate-950/40 p-5">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="space-y-2">
                      <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">{model.version}</p>
                      <h3 className="text-xl font-semibold text-white">{model.name}</h3>
                    </div>
                    <StatusBadge status={model.status} />
                  </div>

                  <div className="mt-5 grid grid-cols-2 gap-3">
                    <div className="rounded-2xl border border-slate-800 bg-slate-900/65 p-4">
                      <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">Accuracy</p>
                      <p className="mt-2 text-2xl font-semibold text-white">{formatPercent(model.accuracy)}</p>
                    </div>
                    <div className="rounded-2xl border border-slate-800 bg-slate-900/65 p-4">
                      <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">F1 score</p>
                      <p className="mt-2 text-2xl font-semibold text-white">{formatPercent(model.f1_score)}</p>
                    </div>
                  </div>

                  <dl className="mt-5 grid gap-3 text-sm text-slate-300">
                    <div className="flex items-center justify-between gap-4 rounded-2xl border border-slate-800/70 bg-slate-950/45 px-4 py-3">
                      <dt className="text-slate-400">Last trained</dt>
                      <dd className="text-right font-medium text-slate-100">{formatDate(model.last_trained)}</dd>
                    </div>
                    <div className="flex items-center justify-between gap-4 rounded-2xl border border-slate-800/70 bg-slate-950/45 px-4 py-3">
                      <dt className="text-slate-400">Health check</dt>
                      <dd className="inline-flex items-center gap-2 font-medium text-slate-100">
                        <HealthDot healthy={healthy} />
                        {healthy ? 'Passing' : 'Failing'}
                      </dd>
                    </div>
                    <div className="flex items-center justify-between gap-4 rounded-2xl border border-slate-800/70 bg-slate-950/45 px-4 py-3">
                      <dt className="text-slate-400">Confidence snapshot</dt>
                      <dd className="inline-flex items-center gap-2 font-medium text-slate-100">
                        {model.status === 'active' ? (
                          <CheckCircle2 className="h-4 w-4 text-emerald-300" />
                        ) : model.status === 'experimental' ? (
                          <Sparkles className="h-4 w-4 text-sky-300" />
                        ) : (
                          <ShieldAlert className="h-4 w-4 text-slate-300" />
                        )}
                        {model.accuracy !== undefined && model.accuracy !== null
                          ? `${formatShortPercent(model.accuracy)} accuracy / ${formatShortPercent(model.f1_score)} F1`
                          : 'Awaiting evaluation metrics'}
                      </dd>
                    </div>
                  </dl>
                </article>
              ))}
            </div>
          ) : (
            <EmptyState message="No models were returned by the registry endpoint." />
          )}
        </Panel>
      </div>
    </main>
  )
}
