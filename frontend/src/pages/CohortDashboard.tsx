import { keepPreviousData, useQuery } from '@tanstack/react-query'
import {
  AlertTriangle,
  ArrowDownUp,
  BarChart3,
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  Filter,
  RefreshCcw,
  ShieldAlert,
  TrendingUp,
  Users,
  type LucideIcon,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { ApiError, type GetCohortParams, getCohort, type RiskLevel } from '@/lib/api'
import { useAuthStore } from '@/store/authStore'

type CohortDashboardProps = {
  cohortId?: number
}

type SortBy = NonNullable<GetCohortParams['sortBy']>
type SortOrder = NonNullable<GetCohortParams['order']>
type RiskFilter = RiskLevel | 'all'

type MetricCardConfig = {
  title: string
  value: string
  detail: string
  icon: LucideIcon
  accentClassName: string
}

type ModalityCardConfig = {
  key: 'text' | 'tabular' | 'behavioral'
  label: string
  description: string
  accentClassName: string
}

type RiskChartDatum = {
  level: RiskLevel
  label: string
  count: number
  fill: string
}

type RiskTooltipProps = {
  active?: boolean
  payload?: Array<{
    value?: number
    payload?: RiskChartDatum
  }>
}

const PAGE_SIZE = 10
const skeletonMetricKeys = ['students', 'risk', 'percentage'] as const
const skeletonModalityKeys = ['text', 'tabular', 'behavioral'] as const
const skeletonFilterKeys = ['all', 'low', 'medium', 'high'] as const
const skeletonTableKeys = ['header', 'row-1', 'row-2', 'row-3', 'row-4', 'row-5'] as const

const dashboardTokens = {
  shell: 'min-h-screen bg-slate-950 text-slate-50',
  container: 'mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8',
  panel:
    'rounded-3xl border border-slate-800/80 bg-slate-900/75 shadow-2xl shadow-slate-950/50 backdrop-blur',
  panelInset: 'rounded-2xl border border-slate-800/70 bg-slate-950/70',
  pill:
    'inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-950/80 px-3 py-1.5 text-xs font-medium uppercase tracking-widest text-slate-300',
  button:
    'inline-flex items-center justify-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition focus:outline-none focus:ring-2 focus:ring-cyan-400/70 focus:ring-offset-2 focus:ring-offset-slate-950 disabled:cursor-not-allowed disabled:opacity-40',
  buttonNeutral: 'border-slate-700 bg-slate-900 text-slate-200 hover:border-slate-600 hover:bg-slate-800',
  buttonActive: 'border-cyan-400/60 bg-cyan-400/10 text-cyan-200 shadow-lg shadow-cyan-950/40',
  mutedText: 'text-slate-400',
  subtleText: 'text-slate-500',
  headingAccent: 'text-cyan-300',
} as const

const dashboardPalette = {
  chartAxis: '#94a3b8',
  chartGrid: '#1e293b',
  chartTooltipBorder: '#334155',
  chartTooltipSurface: '#020617',
  risk: {
    low: '#22c55e',
    medium: '#f59e0b',
    high: '#f43f5e',
  },
} as const

const riskStyles: Record<
  RiskLevel,
  {
    label: string
    badgeClassName: string
    accentClassName: string
    color: string
  }
> = {
  low: {
    label: 'Low risk',
    badgeClassName: 'border-emerald-400/30 bg-emerald-400/10 text-emerald-200',
    accentClassName: 'text-emerald-300',
    color: dashboardPalette.risk.low,
  },
  medium: {
    label: 'Medium risk',
    badgeClassName: 'border-amber-400/30 bg-amber-400/10 text-amber-200',
    accentClassName: 'text-amber-300',
    color: dashboardPalette.risk.medium,
  },
  high: {
    label: 'High risk',
    badgeClassName: 'border-rose-400/30 bg-rose-400/10 text-rose-200',
    accentClassName: 'text-rose-300',
    color: dashboardPalette.risk.high,
  },
}

const modalityCards: ModalityCardConfig[] = [
  {
    key: 'text',
    label: 'Text signal',
    description: 'Narrative and written context strength',
    accentClassName: 'from-cyan-400/20 to-cyan-500/5 text-cyan-200',
  },
  {
    key: 'tabular',
    label: 'Tabular signal',
    description: 'Structured academic and attendance features',
    accentClassName: 'from-violet-400/20 to-violet-500/5 text-violet-200',
  },
  {
    key: 'behavioral',
    label: 'Behavioral signal',
    description: 'Engagement and activity trend strength',
    accentClassName: 'from-emerald-400/20 to-emerald-500/5 text-emerald-200',
  },
]

function cn(...classNames: Array<string | false | null | undefined>) {
  return classNames.filter(Boolean).join(' ')
}

function formatProbability(value: number | null) {
  if (value === null) {
    return 'No score'
  }

  return `${(value * 100).toFixed(1)}%`
}

function formatCompactPercentage(value: number | null) {
  if (value === null) {
    return '—'
  }

  return `${Math.round(value * 100)}%`
}

function formatDateLabel(value: string | null) {
  if (!value) {
    return 'No recent prediction'
  }

  return new Intl.DateTimeFormat('en', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(new Date(value))
}

function resolveCohortId(explicitId: number | undefined, routeId: string | null, queryId: string | null) {
  if (typeof explicitId === 'number' && Number.isInteger(explicitId) && explicitId > 0) {
    return explicitId
  }

  const candidate = routeId ?? queryId
  if (!candidate) {
    return null
  }

  const parsed = Number(candidate)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null
}

function LoadingSkeleton() {
  return (
    <main className={dashboardTokens.shell}>
      <div className={dashboardTokens.container}>
        <section className={cn(dashboardTokens.panel, 'overflow-hidden p-6 sm:p-8')}>
          <div className="animate-pulse space-y-6">
            <div className="h-4 w-44 rounded-full bg-slate-800" />
            <div className="h-10 w-full max-w-xl rounded-2xl bg-slate-800/90" />
            <div className="h-5 w-full max-w-3xl rounded-full bg-slate-800/70" />
            <div className="grid gap-4 lg:grid-cols-3">
              {skeletonMetricKeys.map((key) => (
                <div key={key} className={cn(dashboardTokens.panelInset, 'h-36 bg-slate-950/70')} />
              ))}
            </div>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <div className={cn(dashboardTokens.panel, 'h-[26rem] animate-pulse p-6 sm:p-8')}>
            <div className="h-full rounded-2xl bg-slate-950/70" />
          </div>

          <div className="grid gap-6">
            {skeletonModalityKeys.map((key) => (
              <div key={key} className={cn(dashboardTokens.panel, 'h-32 animate-pulse p-6')}>
                <div className="h-full rounded-2xl bg-slate-950/70" />
              </div>
            ))}
          </div>
        </section>

        <section className={cn(dashboardTokens.panel, 'animate-pulse p-6 sm:p-8')}>
          <div className="mb-6 flex flex-wrap gap-3">
            {skeletonFilterKeys.map((key) => (
              <div key={key} className="h-10 w-24 rounded-full bg-slate-800" />
            ))}
          </div>
          <div className="overflow-hidden rounded-2xl border border-slate-800/70">
            <div className="grid grid-cols-[1.6fr_1fr_1fr] gap-px bg-slate-800/70">
              {skeletonTableKeys.map((key) => (
                <div key={key} className="h-14 bg-slate-950/70" />
              ))}
            </div>
          </div>
        </section>
      </div>
    </main>
  )
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <main className={dashboardTokens.shell}>
      <div className={dashboardTokens.container}>
        <section className={cn(dashboardTokens.panel, 'mx-auto flex w-full max-w-2xl flex-col items-start gap-5 p-8 sm:p-10')}>
          <span className={dashboardTokens.pill}>
            <AlertTriangle className="h-4 w-4 text-rose-300" />
            Cohort request failed
          </span>
          <div className="space-y-3">
            <h1 className="text-3xl font-semibold text-white">Unable to load this cohort dashboard.</h1>
            <p className="max-w-xl text-base leading-7 text-slate-300">{message}</p>
          </div>
          <button
            type="button"
            onClick={onRetry}
            className={cn(dashboardTokens.button, dashboardTokens.buttonActive)}
          >
            <RefreshCcw className="h-4 w-4" />
            Retry request
          </button>
        </section>
      </div>
    </main>
  )
}

function MissingCohortState() {
  return (
    <main className={dashboardTokens.shell}>
      <div className={dashboardTokens.container}>
        <section className={cn(dashboardTokens.panel, 'mx-auto flex w-full max-w-2xl flex-col items-start gap-5 p-8 sm:p-10')}>
          <span className={dashboardTokens.pill}>
            <BarChart3 className="h-4 w-4 text-cyan-300" />
            Cohort dashboard
          </span>
          <div className="space-y-3">
            <h1 className="text-3xl font-semibold text-white">A cohort id is required to render this view.</h1>
            <p className="max-w-xl text-base leading-7 text-slate-300">
              Wire this page to a route such as <span className="font-medium text-white">/cohorts/:id</span> or pass a
              <span className="font-medium text-white"> cohortId </span>
              prop when mounting the component.
            </p>
          </div>
        </section>
      </div>
    </main>
  )
}

function MetricCard({ title, value, detail, icon: Icon, accentClassName }: MetricCardConfig) {
  return (
    <article className={cn(dashboardTokens.panelInset, 'relative overflow-hidden p-5 sm:p-6')}>
      <div className={cn('absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/30 to-transparent')} />
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-3">
          <p className="text-sm font-medium uppercase tracking-wider text-slate-400">{title}</p>
          <div className="space-y-1">
            <p className="text-3xl font-semibold text-white sm:text-4xl">{value}</p>
            <p className="text-sm leading-6 text-slate-400">{detail}</p>
          </div>
        </div>
        <div className={cn('rounded-2xl border border-white/10 p-3', accentClassName)}>
          <Icon className="h-6 w-6" />
        </div>
      </div>
    </article>
  )
}

function RiskDistributionTooltip({ active, payload }: RiskTooltipProps) {
  if (!active || !payload || payload.length === 0 || !payload[0]?.payload) {
    return null
  }

  const item = payload[0].payload

  return (
    <div
      className="rounded-2xl border px-4 py-3 shadow-2xl"
      style={{
        borderColor: dashboardPalette.chartTooltipBorder,
        backgroundColor: dashboardPalette.chartTooltipSurface,
      }}
    >
      <p className="text-sm font-semibold text-white">{item.label}</p>
      <p className="mt-1 text-sm text-slate-300">{item.count} students in this tier</p>
    </div>
  )
}

export default function CohortDashboard({ cohortId }: CohortDashboardProps) {
  const routeParams = useParams<{ id?: string; cohortId?: string }>()
  const [searchParams] = useSearchParams()
  const user = useAuthStore((state) => state.user)
  const [page, setPage] = useState(1)
  const [sortBy, setSortBy] = useState<SortBy>('risk')
  const [order, setOrder] = useState<SortOrder>('desc')
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('all')

  const resolvedCohortId = resolveCohortId(
    cohortId,
    routeParams.id ?? routeParams.cohortId ?? null,
    searchParams.get('cohortId') ?? searchParams.get('id'),
  )

  const activeRiskLevel = riskFilter === 'all' ? undefined : riskFilter

  const cohortQuery = useQuery({
    enabled: resolvedCohortId !== null,
    placeholderData: keepPreviousData,
    queryKey: ['cohort', resolvedCohortId, page, PAGE_SIZE, sortBy, order, activeRiskLevel],
    queryFn: () =>
      getCohort(resolvedCohortId as number, {
        page,
        limit: PAGE_SIZE,
        sortBy,
        order,
        riskLevel: activeRiskLevel,
      }),
  })

  const chartData = useMemo<RiskChartDatum[]>(() => {
    if (!cohortQuery.data) {
      return [
        { level: 'low', label: 'Low risk', count: 0, fill: dashboardPalette.risk.low },
        { level: 'medium', label: 'Medium risk', count: 0, fill: dashboardPalette.risk.medium },
        { level: 'high', label: 'High risk', count: 0, fill: dashboardPalette.risk.high },
      ]
    }

    return [
      {
        level: 'low',
        label: 'Low risk',
        count: cohortQuery.data.risk_distribution.low,
        fill: dashboardPalette.risk.low,
      },
      {
        level: 'medium',
        label: 'Medium risk',
        count: cohortQuery.data.risk_distribution.medium,
        fill: dashboardPalette.risk.medium,
      },
      {
        level: 'high',
        label: 'High risk',
        count: cohortQuery.data.risk_distribution.high,
        fill: dashboardPalette.risk.high,
      },
    ]
  }, [cohortQuery.data])

  const pageNumbers = useMemo(() => {
    const totalPages = cohortQuery.data?.pagination.total_pages ?? 1
    const start = Math.max(1, page - 2)
    const end = Math.min(totalPages, start + 4)
    const adjustedStart = Math.max(1, end - 4)

    return Array.from({ length: end - adjustedStart + 1 }, (_, index) => adjustedStart + index)
  }, [cohortQuery.data?.pagination.total_pages, page])

  const metricCards = useMemo<MetricCardConfig[]>(() => {
    if (!cohortQuery.data) {
      return []
    }

    return [
      {
        title: 'Total students',
        value: cohortQuery.data.total_students.toLocaleString(),
        detail: 'Full cohort population currently tracked in this view.',
        icon: Users,
        accentClassName: 'bg-cyan-400/10 text-cyan-200',
      },
      {
        title: 'At-risk count',
        value: cohortQuery.data.at_risk_count.toLocaleString(),
        detail: 'Students currently flagged for medium or high intervention urgency.',
        icon: ShieldAlert,
        accentClassName: 'bg-rose-400/10 text-rose-200',
      },
      {
        title: 'At-risk percentage',
        value: `${cohortQuery.data.at_risk_percentage.toFixed(1)}%`,
        detail: 'Share of the cohort requiring the closest academic attention.',
        icon: TrendingUp,
        accentClassName: 'bg-amber-400/10 text-amber-200',
      },
    ]
  }, [cohortQuery.data])

  const errorMessage =
    cohortQuery.error != null && (cohortQuery.error instanceof ApiError || cohortQuery.error instanceof Error)
      ? cohortQuery.error.message
      : 'An unexpected error occurred while loading the cohort overview.'

  if (resolvedCohortId === null) {
    return <MissingCohortState />
  }

  if (cohortQuery.isLoading) {
    return <LoadingSkeleton />
  }

  if (cohortQuery.isError && !cohortQuery.data) {
    return <ErrorState message={errorMessage} onRetry={() => void cohortQuery.refetch()} />
  }

  const data = cohortQuery.data

  if (!data) {
    return <LoadingSkeleton />
  }

  const handleSortChange = (nextSortBy: SortBy) => {
    setPage(1)
    setSortBy(nextSortBy)
  }

  const handleOrderToggle = () => {
    setPage(1)
    setOrder((currentOrder: SortOrder) => (currentOrder === 'asc' ? 'desc' : 'asc'))
  }

  const handleRiskFilterChange = (nextRiskFilter: RiskFilter) => {
    setPage(1)
    setRiskFilter(nextRiskFilter)
  }

  const activeSortLabel = sortBy === 'risk' ? 'Risk probability' : 'Student name'

  return (
    <main className={dashboardTokens.shell}>
      <div className="absolute inset-0 bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900" aria-hidden="true" />
      <div className="absolute inset-x-0 top-0 h-72 bg-gradient-to-r from-cyan-500/10 via-transparent to-emerald-500/10" aria-hidden="true" />

      <div className={cn(dashboardTokens.container, 'relative')}>
        <section className={cn(dashboardTokens.panel, 'overflow-hidden p-6 sm:p-8')}>
          <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
            <div className="max-w-3xl space-y-4">
              <div className="flex flex-wrap items-center gap-3">
                <span className={dashboardTokens.pill}>
                  <BarChart3 className={cn('h-4 w-4', dashboardTokens.headingAccent)} />
                  Cohort overview
                </span>
                {user?.email ? (
                  <span className="inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-950/80 px-3 py-1.5 text-sm text-slate-300">
                    Signed in as <span className="font-medium text-slate-100">{user.email}</span>
                  </span>
                ) : null}
              </div>

              <div className="space-y-3">
                <h1 className="text-3xl font-semibold text-white sm:text-5xl">{data.cohort_name}</h1>
                <p className="max-w-2xl text-base leading-7 text-slate-300 sm:text-lg">
                  A dark command-center view for intervention planning, with cohort health signals, modality balance,
                  and a server-driven roster tuned for rapid triage.
                </p>
              </div>
            </div>

            <div className="grid w-full gap-3 sm:grid-cols-2 lg:w-auto">
              <div className={cn(dashboardTokens.panelInset, 'p-4 sm:min-w-56')}>
                <p className="text-xs font-medium uppercase tracking-widest text-slate-500">Active sort</p>
                <p className="mt-2 text-lg font-semibold text-white">{activeSortLabel}</p>
                <p className="mt-1 text-sm text-slate-400">{order === 'desc' ? 'Highest first' : 'Lowest / A–Z first'}</p>
              </div>
              <div className={cn(dashboardTokens.panelInset, 'p-4 sm:min-w-56')}>
                <p className="text-xs font-medium uppercase tracking-widest text-slate-500">Visible roster</p>
                <p className="mt-2 text-lg font-semibold text-white">{data.pagination.total_students.toLocaleString()}</p>
                <p className="mt-1 text-sm text-slate-400">
                  {riskFilter === 'all' ? 'All risk levels represented' : `${riskStyles[riskFilter].label} filter applied`}
                </p>
              </div>
            </div>
          </div>

          {cohortQuery.isFetching ? (
            <div className="mt-6 flex items-center gap-2 text-sm text-cyan-200">
              <RefreshCcw className="h-4 w-4 animate-spin" />
              Refreshing cohort signals…
            </div>
          ) : null}

          {cohortQuery.isError && cohortQuery.data ? (
            <div className="mt-6 rounded-2xl border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
              {errorMessage}
            </div>
          ) : null}

          <div className="mt-8 grid gap-4 lg:grid-cols-3">
            {metricCards.map((card) => (
              <MetricCard key={card.title} {...card} />
            ))}
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <article className={cn(dashboardTokens.panel, 'p-6 sm:p-8')}>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="text-sm font-medium uppercase tracking-widest text-slate-500">Risk distribution</p>
                <h2 className="mt-2 text-2xl font-semibold text-white">Population by intervention tier</h2>
              </div>
              <p className="max-w-sm text-sm leading-6 text-slate-400">
                Distribution remains cohort-wide so planners can compare the filtered roster against the full risk mix.
              </p>
            </div>

            <div className="mt-6 rounded-3xl border border-slate-800/70 bg-slate-950/70 p-4 sm:p-6">
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={dashboardPalette.chartGrid} horizontal={false} />
                    <XAxis type="number" allowDecimals={false} tick={{ fill: dashboardPalette.chartAxis, fontSize: 12 }} />
                    <YAxis
                      dataKey="label"
                      type="category"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: dashboardPalette.chartAxis, fontSize: 12 }}
                      width={88}
                    />
                    <Tooltip cursor={{ fill: 'rgba(148, 163, 184, 0.08)' }} content={<RiskDistributionTooltip />} />
                    <Bar dataKey="count" radius={[0, 18, 18, 0]}>
                      {chartData.map((entry) => (
                        <Cell key={entry.level} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </article>

          <article className="grid gap-6">
            <section className={cn(dashboardTokens.panel, 'p-6 sm:p-8')}>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm font-medium uppercase tracking-widest text-slate-500">Modality strength</p>
                  <h2 className="mt-2 text-2xl font-semibold text-white">Average signal balance</h2>
                </div>
                <div className="rounded-2xl border border-slate-800/70 bg-slate-950/80 p-3 text-cyan-200">
                  <BrainCircuit className="h-5 w-5" />
                </div>
              </div>

              <div className="mt-6 grid gap-4">
                {modalityCards.map((card) => {
                  const value = data.average_modality_scores[card.key]
                  const width = value === null ? 0 : Math.min(100, Math.max(0, value * 100))

                  return (
                    <div key={card.key} className={cn(dashboardTokens.panelInset, 'overflow-hidden p-4')}>
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <p className="text-sm font-semibold text-white">{card.label}</p>
                          <p className="mt-1 text-sm leading-6 text-slate-400">{card.description}</p>
                        </div>
                        <p className="text-lg font-semibold text-white">{formatCompactPercentage(value)}</p>
                      </div>

                      <div className="mt-4 h-2 rounded-full bg-slate-800">
                        <div
                          className={cn('h-2 rounded-full bg-gradient-to-r', card.accentClassName)}
                          style={{ width: `${width}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </section>

            <section className={cn(dashboardTokens.panel, 'p-6')}>
              <p className="text-sm font-medium uppercase tracking-widest text-slate-500">Risk mix</p>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                {chartData.map((item) => (
                  <div key={item.level} className={cn(dashboardTokens.panelInset, 'p-4')}>
                    <div className="flex items-center justify-between gap-3">
                      <span className={cn('text-sm font-medium', riskStyles[item.level].accentClassName)}>{item.label}</span>
                      <span className="text-lg font-semibold text-white">{item.count}</span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          </article>
        </section>

        <section className={cn(dashboardTokens.panel, 'p-6 sm:p-8')}>
          <div className="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <p className="text-sm font-medium uppercase tracking-widest text-slate-500">Student roster</p>
              <h2 className="mt-2 text-2xl font-semibold text-white">Filtered intervention queue</h2>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
                The roster is server-paginated and can be reordered by name or probability while preserving the cohort
                summary above.
              </p>
            </div>

            <div className="flex flex-col gap-3 xl:items-end">
              <div className="flex flex-wrap gap-2">
                {(['all', 'low', 'medium', 'high'] as const).map((level) => {
                  const isActive = riskFilter === level
                  const count =
                    level === 'all'
                      ? data.total_students
                      : data.risk_distribution[level]

                  return (
                    <button
                      key={level}
                      type="button"
                      onClick={() => handleRiskFilterChange(level)}
                      className={cn(
                        dashboardTokens.button,
                        isActive ? dashboardTokens.buttonActive : dashboardTokens.buttonNeutral,
                      )}
                    >
                      <Filter className="h-4 w-4" />
                      {level === 'all' ? 'All' : riskStyles[level].label}
                      <span className="rounded-full bg-white/10 px-2 py-0.5 text-xs tabular-nums text-slate-100">{count}</span>
                    </button>
                  )
                })}
              </div>

              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => handleSortChange('risk')}
                  className={cn(
                    dashboardTokens.button,
                    sortBy === 'risk' ? dashboardTokens.buttonActive : dashboardTokens.buttonNeutral,
                  )}
                >
                  Risk probability
                </button>
                <button
                  type="button"
                  onClick={() => handleSortChange('name')}
                  className={cn(
                    dashboardTokens.button,
                    sortBy === 'name' ? dashboardTokens.buttonActive : dashboardTokens.buttonNeutral,
                  )}
                >
                  Student name
                </button>
                <button
                  type="button"
                  onClick={handleOrderToggle}
                  className={cn(dashboardTokens.button, dashboardTokens.buttonNeutral)}
                >
                  <ArrowDownUp className="h-4 w-4" />
                  {order === 'desc' ? 'Descending' : 'Ascending'}
                </button>
              </div>
            </div>
          </div>

          <div className="mt-6 overflow-hidden rounded-3xl border border-slate-800/70 bg-slate-950/70">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-800 text-left">
                <thead className="bg-slate-900/70">
                  <tr>
                    <th className="px-5 py-4 text-xs font-semibold uppercase tracking-widest text-slate-400">Name</th>
                    <th className="px-5 py-4 text-xs font-semibold uppercase tracking-widest text-slate-400">Risk probability</th>
                    <th className="px-5 py-4 text-xs font-semibold uppercase tracking-widest text-slate-400">Risk level</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80">
                  {data.students.length === 0 ? (
                    <tr>
                      <td colSpan={3} className="px-5 py-10 text-center text-sm text-slate-400">
                        No students match the current filter.
                      </td>
                    </tr>
                  ) : (
                    data.students.map((student) => {
                      const riskLevel = student.risk_level

                      return (
                        <tr key={student.id} className="transition hover:bg-slate-900/60">
                          <td className="px-5 py-4 align-top">
                            <div className="space-y-1">
                              <p className="font-medium text-white">{student.name}</p>
                              <p className="text-sm text-slate-500">{formatDateLabel(student.last_prediction_date)}</p>
                            </div>
                          </td>
                          <td className="px-5 py-4 align-top text-sm text-slate-200 tabular-nums">
                            {formatProbability(student.risk_probability)}
                          </td>
                          <td className="px-5 py-4 align-top">
                            {riskLevel ? (
                              <span
                                className={cn(
                                  'inline-flex items-center rounded-full border px-3 py-1 text-sm font-medium',
                                  riskStyles[riskLevel].badgeClassName,
                                )}
                              >
                                {riskStyles[riskLevel].label}
                              </span>
                            ) : (
                              <span className="inline-flex items-center rounded-full border border-slate-700 bg-slate-900/80 px-3 py-1 text-sm font-medium text-slate-400">
                                No prediction
                              </span>
                            )}
                          </td>
                        </tr>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="text-sm text-slate-400">
              Page <span className="font-medium text-slate-100">{data.pagination.page}</span> of{' '}
              <span className="font-medium text-slate-100">{data.pagination.total_pages}</span> · Showing up to{' '}
              <span className="font-medium text-slate-100">{PAGE_SIZE}</span> students per page.
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => setPage((currentPage) => Math.max(1, currentPage - 1))}
                disabled={data.pagination.page <= 1}
                className={cn(dashboardTokens.button, dashboardTokens.buttonNeutral)}
              >
                <ChevronLeft className="h-4 w-4" />
                Previous
              </button>

              {pageNumbers.map((pageNumber) => {
                const isActive = data.pagination.page === pageNumber

                return (
                  <button
                    key={pageNumber}
                    type="button"
                    onClick={() => setPage(pageNumber)}
                    className={cn(
                      dashboardTokens.button,
                      'min-w-11 px-0',
                      isActive ? dashboardTokens.buttonActive : dashboardTokens.buttonNeutral,
                    )}
                  >
                    {pageNumber}
                  </button>
                )
              })}

              <button
                type="button"
                onClick={() => setPage((currentPage) => Math.min(data.pagination.total_pages, currentPage + 1))}
                disabled={data.pagination.page >= data.pagination.total_pages}
                className={cn(dashboardTokens.button, dashboardTokens.buttonNeutral)}
              >
                Next
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        </section>
      </div>
    </main>
  )
}
