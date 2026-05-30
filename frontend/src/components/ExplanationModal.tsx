import {
	BrainCircuit,
	CalendarClock,
	Sparkles,
	TrendingDown,
	TrendingUp,
	X,
} from "lucide-react";
import { useEffect } from "react";

import type {
	CohortStudent,
	PredictionExplanation,
	PredictionFactor,
	RiskLevel,
} from "@/lib/api";

export type { RiskLevel };

export type StudentRiskFactor = PredictionFactor & {
	value?: number;
};

export type StudentRiskExplanation = PredictionExplanation & {
	explanation?: string;
	shap_values?: StudentRiskFactor[];
};

export type StudentRiskRecord = CohortStudent & {
	riskLevel?: RiskLevel;
	riskProbability?: number;
	probability?: number;
	lastPredictionDate?: string | null;
	explanation?: StudentRiskExplanation | null;
};

type ExplanationModalProps = {
	open: boolean;
	student: StudentRiskRecord | null;
	explanation: StudentRiskExplanation | null;
	onClose: () => void;
};

type RiskTone = {
	badge: string;
	accent: string;
	panel: string;
	label: string;
};

export const riskTones: Record<RiskLevel, RiskTone> = {
	low: {
		badge: "border border-emerald-500/20 bg-emerald-500/10 text-emerald-300",
		accent: "text-emerald-300",
		panel: "from-emerald-500/12 via-emerald-500/0 to-slate-950",
		label: "Low risk",
	},
	medium: {
		badge: "border border-amber-500/20 bg-amber-500/10 text-amber-200",
		accent: "text-amber-200",
		panel: "from-amber-500/12 via-amber-500/0 to-slate-950",
		label: "Medium risk",
	},
	high: {
		badge: "border border-rose-500/20 bg-rose-500/10 text-rose-200",
		accent: "text-rose-200",
		panel: "from-rose-500/14 via-rose-500/0 to-slate-950",
		label: "High risk",
	},
};

export function normalizeRiskLevel(level?: string | null): RiskLevel {
	if (level === "low" || level === "medium" || level === "high") {
		return level;
	}

	return "medium";
}

export function getStudentProbability(
	student?: StudentRiskRecord | null,
): number {
	if (!student) {
		return 0;
	}

	return (
		student.risk_probability ??
		student.riskProbability ??
		student.probability ??
		0
	);
}

export function getLastPredictionDate(
	student?: StudentRiskRecord | null,
): string | null {
	if (!student) {
		return null;
	}

	return student.last_prediction_date ?? student.lastPredictionDate ?? null;
}

export function formatProbability(probability: number): string {
	return `${(probability * 100).toFixed(1)}%`;
}

export function formatPredictionDate(dateValue?: string | null): string {
	if (!dateValue) {
		return "No recent prediction";
	}

	const parsed = new Date(dateValue);

	if (Number.isNaN(parsed.getTime())) {
		return dateValue;
	}

	return new Intl.DateTimeFormat("en-US", {
		month: "short",
		day: "numeric",
		year: "numeric",
	}).format(parsed);
}

function getExplanationFactors(
	explanation?: StudentRiskExplanation | null,
): StudentRiskFactor[] {
	return explanation?.top_factors ?? explanation?.shap_values ?? [];
}

function getNarrativeSummary(
	explanation?: StudentRiskExplanation | null,
): string {
	return (
		explanation?.narrative_summary ??
		explanation?.explanation ??
		"A detailed narrative summary has not been generated for this prediction yet."
	);
}

export default function ExplanationModal({
	open,
	student,
	explanation,
	onClose,
}: ExplanationModalProps) {
	useEffect(() => {
		if (!open) {
			return undefined;
		}

		const previousOverflow = document.body.style.overflow;

		document.body.style.overflow = "hidden";

		const handleKeyDown = (event: KeyboardEvent) => {
			if (event.key === "Escape") {
				onClose();
			}
		};

		window.addEventListener("keydown", handleKeyDown);

		return () => {
			document.body.style.overflow = previousOverflow;
			window.removeEventListener("keydown", handleKeyDown);
		};
	}, [onClose, open]);

	if (!open || !student) {
		return null;
	}

	const level = normalizeRiskLevel(
		explanation?.risk_level ?? student.risk_level ?? student.riskLevel,
	);
	const tone = riskTones[level];
	const probability =
		explanation?.probability ?? getStudentProbability(student);
	const topFactors = getExplanationFactors(explanation);
	const lastPrediction = formatPredictionDate(getLastPredictionDate(student));

	return (
		<div
			aria-modal="true"
			className="fixed inset-0 z-50 flex items-center justify-center px-4 py-6"
			role="dialog"
		>
			<button
				aria-label="Close explanation modal"
				className="absolute inset-0 bg-slate-950/80 backdrop-blur-md"
				type="button"
				onClick={onClose}
			/>
			<div
				className={`relative w-full max-w-4xl overflow-hidden rounded-3xl border border-slate-800/80 bg-gradient-to-br ${tone.panel} shadow-2xl shadow-black/40`}
			>
				<div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-transparent to-transparent" />
				<div className="relative flex items-start justify-between gap-4 border-b border-slate-800/80 px-5 py-5 sm:px-7 sm:py-6">
					<div className="space-y-3">
						<div className="flex flex-wrap items-center gap-3">
							<span className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-widest text-cyan-200">
								Explanation dossier
							</span>
							<span
								className={`rounded-full px-3 py-1 text-xs font-semibold ${tone.badge}`}
							>
								{tone.label}
							</span>
						</div>
						<div>
							<h2 className="font-serif text-2xl text-white sm:text-3xl">
								{student.name}
							</h2>
							<p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300 sm:text-base">
								{getNarrativeSummary(explanation)}
							</p>
						</div>
					</div>
					<button
						aria-label="Close explanation"
						className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-slate-700 bg-slate-950/70 text-slate-300 transition hover:border-cyan-400/60 hover:text-cyan-200"
						type="button"
						onClick={onClose}
					>
						<X className="h-5 w-5" />
					</button>
				</div>

				<div className="relative grid gap-4 px-5 py-5 sm:px-7 sm:py-6 xl:grid-cols-5">
					<section className="space-y-4 xl:col-span-3">
						<div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
							<article className="rounded-3xl border border-slate-800 bg-slate-950/70 p-4">
								<div className="flex items-center gap-3 text-slate-400">
									<Sparkles className="h-4 w-4 text-cyan-300" />
									<span className="text-xs font-semibold uppercase tracking-widest">
										Risk probability
									</span>
								</div>
								<div className="mt-3 flex items-end gap-3">
									<span className={`text-3xl font-semibold ${tone.accent}`}>
										{formatProbability(probability)}
									</span>
									<span className="pb-1 text-sm text-slate-400">
										current prediction
									</span>
								</div>
							</article>

							<article className="rounded-3xl border border-slate-800 bg-slate-950/70 p-4">
								<div className="flex items-center gap-3 text-slate-400">
									<CalendarClock className="h-4 w-4 text-cyan-300" />
									<span className="text-xs font-semibold uppercase tracking-widest">
										Last prediction
									</span>
								</div>
								<p className="mt-3 text-lg font-medium text-white">
									{lastPrediction}
								</p>
							</article>

							<article className="rounded-3xl border border-slate-800 bg-slate-950/70 p-4 sm:col-span-2 xl:col-span-1">
								<div className="flex items-center gap-3 text-slate-400">
									<BrainCircuit className="h-4 w-4 text-cyan-300" />
									<span className="text-xs font-semibold uppercase tracking-widest">
										Model readout
									</span>
								</div>
								<p className="mt-3 text-sm leading-6 text-slate-300">
									{explanation?.modality_contributions ??
										"Modality-specific contributions are not available for this prediction yet."}
								</p>
							</article>
						</div>

						<article className="rounded-3xl border border-slate-800 bg-slate-950/70 p-5">
							<div className="flex items-center gap-3">
								<div className="rounded-2xl border border-cyan-500/20 bg-cyan-500/10 p-2 text-cyan-200">
									<BrainCircuit className="h-4 w-4" />
								</div>
								<div>
									<h3 className="text-lg font-semibold text-white">
										Top drivers
									</h3>
									<p className="text-sm text-slate-400">
										The highest-signal factors influencing this prediction.
									</p>
								</div>
							</div>

							<div className="mt-4 space-y-3">
								{topFactors.length > 0 ? (
									topFactors.map((factor) => {
										const contribution = factor.shap_value ?? factor.value ?? 0;
										const raisesRisk = contribution < 0;

										return (
											<div
												key={`${factor.feature}-${factor.description}`}
												className="rounded-3xl border border-slate-800/90 bg-slate-900/70 p-4"
											>
												<div className="flex flex-wrap items-start justify-between gap-3">
													<div>
														<p className="text-sm font-semibold uppercase tracking-wide text-slate-200">
															{factor.feature.replace(/_/g, " ")}
														</p>
														<p className="mt-2 text-sm leading-6 text-slate-400">
															{factor.description}
														</p>
													</div>
													<div
														className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${
															raisesRisk
																? "border border-rose-500/20 bg-rose-500/10 text-rose-200"
																: "border border-cyan-500/20 bg-cyan-500/10 text-cyan-200"
														}`}
													>
														{raisesRisk ? (
															<TrendingDown className="h-3.5 w-3.5" />
														) : (
															<TrendingUp className="h-3.5 w-3.5" />
														)}
														{contribution >= 0 ? "+" : ""}
														{contribution.toFixed(2)}
													</div>
												</div>
											</div>
										);
									})
								) : (
									<div className="rounded-3xl border border-dashed border-slate-700 bg-slate-950/40 p-4 text-sm leading-6 text-slate-400">
										No factor-level explanation is available for this student
										yet.
									</div>
								)}
							</div>
						</article>
					</section>

					<aside className="space-y-4 xl:col-span-2">
						<article className="rounded-3xl border border-slate-800 bg-slate-950/75 p-5">
							<h3 className="text-lg font-semibold text-white">
								Narrative summary
							</h3>
							<p className="mt-3 text-sm leading-7 text-slate-300">
								{getNarrativeSummary(explanation)}
							</p>
						</article>

						<article className="rounded-3xl border border-slate-800 bg-slate-950/75 p-5">
							<h3 className="text-lg font-semibold text-white">
								Modality contributions
							</h3>
							<p className="mt-3 text-sm leading-7 text-slate-300">
								{explanation?.modality_contributions ??
									"The system has not yet provided a modality breakdown for this prediction."}
							</p>
						</article>
					</aside>
				</div>
			</div>
		</div>
	);
}
