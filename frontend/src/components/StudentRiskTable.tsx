import {
	AlertTriangle,
	ArrowRight,
	CalendarClock,
	ScanSearch,
} from "lucide-react";
import { useMemo, useState } from "react";

import ExplanationModal, {
	formatPredictionDate,
	formatProbability,
	getLastPredictionDate,
	getStudentProbability,
	normalizeRiskLevel,
	riskTones,
	type StudentRiskRecord,
} from "@/components/ExplanationModal";

type StudentRiskTableProps = {
	students: StudentRiskRecord[];
	title?: string;
	description?: string;
	emptyMessage?: string;
	className?: string;
	onStudentSelect?: (student: StudentRiskRecord) => void;
};

function joinClasses(...values: Array<string | undefined>) {
	return values.filter(Boolean).join(" ");
}

const rowButtonClassName =
	"w-full appearance-none bg-transparent p-0 text-left font-inherit text-inherit focus:outline-none";

export default function StudentRiskTable({
	students,
	title = "At-risk students",
	description = "Prioritized learners who need attention, with explanation overlays for fast intervention planning.",
	emptyMessage = "No at-risk students are available right now.",
	className,
	onStudentSelect,
}: StudentRiskTableProps) {
	const [selectedStudent, setSelectedStudent] =
		useState<StudentRiskRecord | null>(null);

	const distribution = useMemo(() => {
		return students.reduce(
			(summary, student) => {
				const level = normalizeRiskLevel(
					student.risk_level ?? student.riskLevel,
				);

				summary[level] += 1;

				return summary;
			},
			{ low: 0, medium: 0, high: 0 },
		);
	}, [students]);

	const openStudent = (student: StudentRiskRecord) => {
		setSelectedStudent(student);
		onStudentSelect?.(student);
	};

	return (
		<>
			<section
				className={joinClasses(
					"overflow-hidden rounded-3xl border border-slate-800/80 bg-slate-950/75 shadow-2xl shadow-black/20",
					className,
				)}
			>
				<div className="border-b border-slate-800/80 px-5 py-5 sm:px-7 sm:py-6">
					<div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
						<div className="space-y-3">
							<span className="inline-flex w-fit items-center gap-2 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-widest text-cyan-200">
								<AlertTriangle className="h-3.5 w-3.5" />
								Intervention queue
							</span>
							<div>
								<h2 className="font-serif text-2xl text-white sm:text-3xl">
									{title}
								</h2>
								<p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400 sm:text-base">
									{description}
								</p>
							</div>
						</div>

						<div className="grid grid-cols-3 gap-3 sm:w-auto">
							{(["low", "medium", "high"] as const).map((level) => {
								const tone = riskTones[level];

								return (
									<div
										key={level}
										className="rounded-2xl border border-slate-800 bg-slate-900/70 px-3 py-3 text-center"
									>
										<p className="text-xs font-semibold uppercase tracking-widest text-slate-500">
											{tone.label}
										</p>
										<p className={`mt-2 text-xl font-semibold ${tone.accent}`}>
											{distribution[level]}
										</p>
									</div>
								);
							})}
						</div>
					</div>
				</div>

				{students.length === 0 ? (
					<div className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
						<div className="rounded-full border border-slate-800 bg-slate-900/70 p-4 text-cyan-200">
							<ScanSearch className="h-6 w-6" />
						</div>
						<div className="space-y-2">
							<p className="text-lg font-medium text-white">
								Nothing to escalate
							</p>
							<p className="max-w-md text-sm leading-6 text-slate-400">
								{emptyMessage}
							</p>
						</div>
					</div>
				) : (
					<>
						<div className="hidden overflow-x-auto lg:block">
							<table className="min-w-full border-collapse text-left text-sm text-slate-200">
								<thead className="bg-slate-950/70 text-xs uppercase tracking-widest text-slate-500">
									<tr>
										<th className="px-7 py-4 font-medium">Student</th>
										<th className="px-7 py-4 font-medium">Risk probability</th>
										<th className="px-7 py-4 font-medium">Risk level</th>
										<th className="px-7 py-4 font-medium">Last prediction</th>
										<th className="px-7 py-4 font-medium text-right">
											Details
										</th>
									</tr>
								</thead>
								<tbody>
									{students.map((student) => {
										const level = normalizeRiskLevel(
											student.risk_level ?? student.riskLevel,
										);
										const tone = riskTones[level];

										return (
											<tr
												key={student.id}
												className="border-t border-slate-800/70 transition hover:bg-slate-900/70 focus-within:bg-slate-900/70 hover:text-white"
											>
												<td className="px-7 py-4">
													<button
														className={rowButtonClassName}
														type="button"
														onClick={() => openStudent(student)}
													>
														<p className="font-semibold text-white">
															{student.name}
														</p>
														<p className="mt-1 text-xs uppercase tracking-wide text-slate-500">
															Student #{student.id}
														</p>
													</button>
												</td>
												<td className="px-7 py-4">
													<button
														className={rowButtonClassName}
														type="button"
														onClick={() => openStudent(student)}
													>
														<span
															className={`inline-flex rounded-full px-3 py-1.5 text-sm font-semibold ${tone.badge}`}
														>
															{formatProbability(
																getStudentProbability(student),
															)}
														</span>
													</button>
												</td>
												<td className="px-7 py-4">
													<button
														className={rowButtonClassName}
														type="button"
														onClick={() => openStudent(student)}
													>
														<span
															className={`inline-flex rounded-full px-3 py-1.5 text-sm font-semibold ${tone.badge}`}
														>
															{tone.label}
														</span>
													</button>
												</td>
												<td className="px-7 py-4 text-slate-300">
													<button
														className={rowButtonClassName}
														type="button"
														onClick={() => openStudent(student)}
													>
														{formatPredictionDate(
															getLastPredictionDate(student),
														)}
													</button>
												</td>
												<td className="px-7 py-4 text-right">
													<button
														className="inline-flex items-center gap-2 text-sm font-medium text-cyan-200"
														type="button"
														onClick={() => openStudent(student)}
													>
														Open explanation
														<ArrowRight className="h-4 w-4" />
													</button>
												</td>
											</tr>
										);
									})}
								</tbody>
							</table>
						</div>

						<div className="grid gap-4 p-4 lg:hidden sm:p-6">
							{students.map((student) => {
								const level = normalizeRiskLevel(
									student.risk_level ?? student.riskLevel,
								);
								const tone = riskTones[level];

								return (
									<button
										key={student.id}
										className="rounded-3xl border border-slate-800 bg-slate-900/75 p-4 text-left transition hover:border-cyan-500/40 hover:bg-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-400/50"
										type="button"
										onClick={() => openStudent(student)}
									>
										<div className="flex items-start justify-between gap-3">
											<div>
												<p className="text-lg font-semibold text-white">
													{student.name}
												</p>
												<p className="mt-1 text-xs uppercase tracking-wide text-slate-500">
													Student #{student.id}
												</p>
											</div>
											<span
												className={`rounded-full px-3 py-1 text-xs font-semibold ${tone.badge}`}
											>
												{tone.label}
											</span>
										</div>

										<div className="mt-4 grid gap-3 sm:grid-cols-2">
											<div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-3">
												<p className="text-xs font-semibold uppercase tracking-widest text-slate-500">
													Risk probability
												</p>
												<p
													className={`mt-2 text-xl font-semibold ${tone.accent}`}
												>
													{formatProbability(getStudentProbability(student))}
												</p>
											</div>
											<div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-3">
												<p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-slate-500">
													<CalendarClock className="h-3.5 w-3.5" />
													Last prediction
												</p>
												<p className="mt-2 text-sm font-medium text-slate-200">
													{formatPredictionDate(getLastPredictionDate(student))}
												</p>
											</div>
										</div>

										<div className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-cyan-200">
											Tap for full explanation
											<ArrowRight className="h-4 w-4" />
										</div>
									</button>
								);
							})}
						</div>
					</>
				)}
			</section>

			<ExplanationModal
				explanation={selectedStudent?.explanation ?? null}
				open={selectedStudent !== null}
				student={selectedStudent}
				onClose={() => setSelectedStudent(null)}
			/>
		</>
	);
}
