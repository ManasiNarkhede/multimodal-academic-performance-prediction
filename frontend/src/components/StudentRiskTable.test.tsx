import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import StudentRiskTable from './StudentRiskTable'
import type { StudentRiskRecord } from '@/components/ExplanationModal'

vi.mock('lucide-react', () => ({
  AlertTriangle: () => <svg data-testid="alert-triangle" />,
  ArrowRight: () => <svg data-testid="arrow-right" />,
  CalendarClock: () => <svg data-testid="calendar-clock" />,
  ScanSearch: () => <svg data-testid="scan-search" />,
  BrainCircuit: () => <svg data-testid="brain-circuit" />,
  Sparkles: () => <svg data-testid="sparkles" />,
  TrendingDown: () => <svg data-testid="trending-down" />,
  TrendingUp: () => <svg data-testid="trending-up" />,
  X: () => <svg data-testid="x-icon" />,
}))

const mockStudents: StudentRiskRecord[] = [
  {
    id: 1,
    name: 'Alice Smith',
    risk_probability: 0.85,
    risk_level: 'high',
    last_prediction_date: '2024-06-15T10:00:00Z',
    explanation: {
      risk_level: 'high',
      probability: 0.85,
      top_factors: [
        { feature: 'attendance', shap_value: -0.3, description: 'Low attendance' },
      ],
      modality_contributions: 'Text: 0.5, Tabular: 0.3',
      narrative_summary: 'At risk due to low attendance',
    },
  },
  {
    id: 2,
    name: 'Bob Jones',
    risk_probability: 0.45,
    risk_level: 'medium',
    last_prediction_date: '2024-06-10T10:00:00Z',
    explanation: null,
  },
  {
    id: 3,
    name: 'Carol White',
    risk_probability: 0.15,
    risk_level: 'low',
    last_prediction_date: null,
    explanation: null,
  },
]

describe('StudentRiskTable', () => {
  it('should render with default title and description', () => {
    render(<StudentRiskTable students={mockStudents} />)

    expect(screen.getByText('At-risk students')).toBeInTheDocument()
    expect(
      screen.getByText(
        'Prioritized learners who need attention, with explanation overlays for fast intervention planning.',
      ),
    ).toBeInTheDocument()
  })

  it('should render with custom title and description', () => {
    render(
      <StudentRiskTable
        students={mockStudents}
        title="Custom Title"
        description="Custom description"
      />,
    )

    expect(screen.getByText('Custom Title')).toBeInTheDocument()
    expect(screen.getByText('Custom description')).toBeInTheDocument()
  })

  it('should render distribution counts', () => {
    render(<StudentRiskTable students={mockStudents} />)

    expect(screen.getAllByText('Low risk').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Medium risk').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('High risk').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1').length).toBeGreaterThanOrEqual(3)
  })

  it('should render student names', () => {
    render(<StudentRiskTable students={mockStudents} />)

    expect(screen.getAllByText('Alice Smith').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Bob Jones').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Carol White').length).toBeGreaterThanOrEqual(1)
  })

  it('should render empty message when no students', () => {
    render(
      <StudentRiskTable
        students={[]}
        emptyMessage="No students found in this cohort."
      />,
    )

    expect(screen.getByText('Nothing to escalate')).toBeInTheDocument()
    expect(screen.getByText('No students found in this cohort.')).toBeInTheDocument()
  })

  it('should call onStudentSelect when a row is clicked', () => {
    const onStudentSelect = vi.fn()
    render(<StudentRiskTable students={mockStudents} onStudentSelect={onStudentSelect} />)

    const desktopTable = screen.getByRole('table')
    const aliceCell = within(desktopTable).getByText('Alice Smith')
    const aliceButton = aliceCell.closest('button')
    expect(aliceButton).toBeInTheDocument()
    fireEvent.click(aliceButton!)

    expect(onStudentSelect).toHaveBeenCalledTimes(1)
    expect(onStudentSelect).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 1,
        name: 'Alice Smith',
      }),
    )
  })

  it('should open modal when Open explanation button is clicked', async () => {
    render(<StudentRiskTable students={mockStudents} />)

    const openButtons = screen.getAllByText('Open explanation')
    expect(openButtons.length).toBeGreaterThan(0)

    fireEvent.click(openButtons[0])

    await waitFor(() => {
      expect(screen.getByText('Explanation dossier')).toBeInTheDocument()
    })
  })

  it('should close modal when close button is clicked', async () => {
    render(<StudentRiskTable students={mockStudents} />)

    const openButtons = screen.getAllByText('Open explanation')
    fireEvent.click(openButtons[0])

    await waitFor(() => {
      expect(screen.getByText('Explanation dossier')).toBeInTheDocument()
    })

    const closeButton = screen.getByLabelText('Close explanation')
    fireEvent.click(closeButton)

    await waitFor(() => {
      expect(screen.queryByText('Explanation dossier')).not.toBeInTheDocument()
    })
  })

  it('should render risk probabilities formatted', () => {
    render(<StudentRiskTable students={mockStudents} />)

    expect(screen.getAllByText('85.0%').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('45.0%').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('15.0%').length).toBeGreaterThanOrEqual(1)
  })

  it('should render formatted prediction dates', () => {
    render(<StudentRiskTable students={mockStudents} />)

    expect(screen.getAllByText('Jun 15, 2024').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Jun 10, 2024').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('No recent prediction').length).toBeGreaterThanOrEqual(1)
  })
})
