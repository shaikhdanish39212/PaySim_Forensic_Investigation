import { useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

const navigationItems = [
  ["dashboard", "⌂", "Dashboard"],
  ["methodology", "◈", "Methodology"],
  ["detection", "⌁", "Fraud Detection"],
  ["suspicious", "◉", "Suspicious Transactions"],
  ["cases", "▣", "Forensic Cases"],
  ["evidence", "↗", "Evidence Traceability"],
  ["sensitivity", "⌁", "Window Sensitivity"],
  ["results", "▤", "Results"],
];

const workflowSteps = [
  ["01", "PaySim Dataset", "Financial transaction records are preserved as the source evidence base."],
  ["02", "Audited Detection", "The Random Forest model flags high-risk transactions as investigation seeds."],
  ["03", "Case Reconstruction", "Each seed is reconstructed into a candidate forensic case."],
  ["04", "Evidence Mapping", "Cases are linked back to transaction records, findings and traceability results."],
];

function useApi(url, enabled = true) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    setLoading(true);
    setError("");

    fetch(`${API_BASE}${url}`)
      .then((response) => {
        if (!response.ok) throw new Error(`API request failed: ${response.status}`);
        return response.json();
      })
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [url, enabled]);

  return { data, loading, error };
}

function numberValue(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function percent(value, digits = 2) {
  return `${(numberValue(value) * 100).toFixed(digits)}%`;
}

function fixed(value, digits = 4) {
  return numberValue(value).toFixed(digits);
}

function intFormat(value) {
  return Math.round(numberValue(value)).toLocaleString();
}

function amountFormat(value) {
  return numberValue(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function metricFormat(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return value ?? "-";
  if (Number.isInteger(numeric)) return numeric.toLocaleString();
  return numeric.toFixed(4);
}

function firstExisting(row, keys, fallback = "-") {
  for (const key of keys) {
    if (row?.[key] !== undefined && row?.[key] !== null && row?.[key] !== "") {
      return row[key];
    }
  }
  return fallback;
}

function PageHeader({ eyebrow, title, description, badge }) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        {description && <p className="header-description">{description}</p>}
      </div>
      {badge && <span className="page-badge">{badge}</span>}
    </header>
  );
}

function Panel({ children, className = "" }) {
  return <section className={`panel ${className}`}>{children}</section>;
}

function Loading({ text = "Loading..." }) {
  return (
    <div className="page-state">
      <div className="loader" />
      <p>{text}</p>
    </div>
  );
}

function ErrorState({ error }) {
  return (
    <div className="page-state error-state">
      <h2>Unable to load data</h2>
      <p>{error}</p>
    </div>
  );
}

function EmptyState({ text }) {
  return <div className="empty-state">{text}</div>;
}

function StatCard({ icon, label, value, subtitle }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>
      <div className="stat-card-content">
        <p className="stat-label">{label}</p>
        <h2 className="stat-value">{value}</h2>
        {subtitle && <p className="stat-subtitle">{subtitle}</p>}
      </div>
    </div>
  );
}

function MetricRow({ label, value }) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SectionTitle({ label, title, badge }) {
  return (
    <div className="section-heading">
      <div>
        <p className="section-label">{label}</p>
        <h2>{title}</h2>
      </div>
      {badge && <span className="dataset-badge">{badge}</span>}
    </div>
  );
}

function Workflow() {
  return (
    <div className="workflow">
      {workflowSteps.map(([number, title, text], index) => (
        <div className="workflow-group" key={number}>
          <div className="workflow-step">
            <div className="workflow-number">{number}</div>
            <h3>{title}</h3>
            <p>{text}</p>
          </div>
          {index < workflowSteps.length - 1 && <div className="workflow-arrow">→</div>}
        </div>
      ))}
    </div>
  );
}

function ResultCard({ number, value, text }) {
  return (
    <div className="result-card">
      <span>{number}</span>
      <strong>{value}</strong>
      <p>{text}</p>
    </div>
  );
}

function ProgressPill({ label, value }) {
  const numeric = Math.max(0, Math.min(100, numberValue(value)));

  return (
    <div className="progress-item">
      <div className="progress-header">
        <span>{label}</span>
        <strong>{numeric.toFixed(0)}%</strong>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${numeric}%` }} />
      </div>
    </div>
  );
}

function MiniDonut({ value, label, sublabel }) {
  const numeric = Math.max(0, Math.min(100, numberValue(value)));
  const circumference = 2 * Math.PI * 42;
  const dash = (numeric / 100) * circumference;

  return (
    <div className="donut-card">
      <svg viewBox="0 0 110 110" role="img" aria-label={`${label}: ${numeric}%`}>
        <circle className="donut-track" cx="55" cy="55" r="42" />
        <circle
          className="donut-value"
          cx="55"
          cy="55"
          r="42"
          strokeDasharray={`${dash} ${circumference - dash}`}
        />
      </svg>
      <strong>{numeric.toFixed(0)}%</strong>
      <span>{label}</span>
      {sublabel && <p>{sublabel}</p>}
    </div>
  );
}

function PerformanceChart({ rows }) {
  const metrics = [
    ["precision", "Precision"],
    ["recall", "Recall"],
    ["f1", "F1"],
    ["roc_auc", "ROC-AUC"],
    ["pr_auc", "PR-AUC"],
  ];

  if (!rows.length) return <EmptyState text="No model comparison records available." />;

  return (
    <div className="performance-chart">
      {rows.map((row) => (
        <div className="performance-group" key={row.model}>
          <div className="performance-title">
            <strong>{row.model}</strong>
            {row.model === "Random Forest" && <span className="success-badge">Selected</span>}
          </div>
          <div className="performance-bars">
            {metrics.map(([key, label]) => {
              const value = numberValue(row[key]);
              return (
                <div className="performance-bar-row" key={key}>
                  <span>{label}</span>
                  <div className="metric-bar-track">
                    <div className="metric-bar-fill" style={{ width: `${Math.min(100, value * 100)}%` }} />
                  </div>
                  <strong>{key.includes("auc") ? fixed(value) : percent(value)}</strong>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function FeatureImportanceChart({ features }) {
  const normalized = (features ?? []).slice(0, 10).map((feature, index) => {
    const entries = Object.entries(feature);
    const nameEntry = entries.find(([key]) => {
      const lower = key.toLowerCase();
      return lower.includes("feature") || lower.includes("name");
    });
    const valueEntry = entries.find(([key]) => key.toLowerCase().includes("importance"));

    return {
      name: nameEntry?.[1] ?? entries[0]?.[1] ?? `Feature ${index + 1}`,
      value: numberValue(valueEntry?.[1] ?? entries[1]?.[1]),
    };
  });

  const maxValue = Math.max(...normalized.map((item) => item.value), 1);

  if (!normalized.length) return <EmptyState text="No feature-importance records available." />;

  return (
    <div className="feature-list">
      {normalized.map((feature) => (
        <div className="feature-row" key={feature.name}>
          <div className="feature-label">
            <span>{feature.name}</span>
            <strong>{feature.value.toFixed(5)}</strong>
          </div>
          <div className="metric-bar-track">
            <div className="metric-bar-fill alt" style={{ width: `${(feature.value / maxValue) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function WindowSensitivityChart({ rows }) {
  if (!rows.length) return <EmptyState text="No temporal window records available." />;

  const maxContext = Math.max(...rows.map((row) => numberValue(row.cases_with_context)), 1);
  const points = rows.map((row, index) => {
    const x = rows.length === 1 ? 50 : 8 + (index * 84) / (rows.length - 1);
    const y = 88 - (numberValue(row.cases_with_context) / maxContext) * 72;
    return { x, y, row };
  });
  const polyline = points.map((point) => `${point.x},${point.y}`).join(" ");

  return (
    <div className="window-visual">
      <svg className="line-chart" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Window sensitivity trend">
        <line className="axis-line" x1="8" y1="88" x2="94" y2="88" />
        <line className="axis-line" x1="8" y1="12" x2="8" y2="88" />
        <polyline className="trend-line" points={polyline} />
        {points.map((point) => (
          <circle className="trend-point" key={point.row.window_steps} cx={point.x} cy={point.y} r="1.8" />
        ))}
      </svg>

      <div className="sensitivity-chart">
        {rows.map((row) => {
          const cases = numberValue(row.cases_with_context);
          const height = (cases / maxContext) * 100;

          return (
            <div className="chart-column" key={row.window_steps}>
              <div className="chart-value">{cases}</div>
              <div className="chart-bar-area">
                <div
                  className={numberValue(row.window_steps) === 24 ? "chart-bar selected" : "chart-bar"}
                  style={{ height: `${Math.max(height, 4)}%` }}
                />
              </div>
              <strong>±{row.window_steps}</strong>
              <span>{row.window_hours_approx}h</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CaseReconstructionVisual({ caseData, evidence }) {
  if (!caseData) return null;

  const seedEvidence = evidence.filter((row) => row.evidence_role === "SUSPICIOUS_SEED");
  const contextualEvidence = evidence.filter((row) => row.evidence_role !== "SUSPICIOUS_SEED");

  return (
    <div className="case-reconstruction">
      <div className="reconstruction-node source-node">
        <span>Seed</span>
        <strong>{caseData.seed_transaction_ids}</strong>
        <p>Score {fixed(caseData.maximum_seed_suspicion_score, 4)}</p>
      </div>

      <div className="reconstruction-link" />

      <div className="reconstruction-node case-node">
        <span>Case</span>
        <strong>{caseData.case_id}</strong>
        <p>{intFormat(caseData.evidence_transaction_count)} evidence record(s)</p>
      </div>

      <div className="reconstruction-link" />

      <div className="reconstruction-node evidence-node">
        <span>Evidence</span>
        <strong>{seedEvidence.length} seed + {contextualEvidence.length} context</strong>
        <p>{intFormat(caseData.unique_account_count)} linked account(s)</p>
      </div>
    </div>
  );
}

function Dashboard({ dashboard }) {
  return (
    <>
      <PageHeader
        eyebrow="RESEARCH IMPLEMENTATION"
        title="Banking & Financial Cybercrime Investigation"
        description="Digital forensics research demonstration using the PaySim financial transaction dataset."
        badge="System Active"
      />

      <div className="research-notice">
        <div className="notice-icon">i</div>
        <div>
          <h3>Research Demonstration</h3>
          <p>
            This interface visualizes the implemented fraud detection and forensic reconstruction workflow.
            Suspicious transactions are investigation seeds and do not constitute proof of criminal activity.
          </p>
        </div>
      </div>

      <section>
        <SectionTitle label="DATASET OVERVIEW" title="Investigation Summary" badge="PaySim" />
        <div className="stats-grid">
          <StatCard icon="◈" label="Total Transactions" value={dashboard.total_transactions.toLocaleString()} subtitle="Complete PaySim dataset" />
          <StatCard icon="!" label="Fraud Transactions" value={dashboard.fraud_transactions.toLocaleString()} subtitle="Known fraudulent records" />
          <StatCard icon="⌁" label="Suspicious Seeds" value={dashboard.suspicious_seeds} subtitle={`Threshold: ${dashboard.forensic_threshold}`} />
          <StatCard icon="▣" label="Candidate Cases" value={dashboard.candidate_cases} subtitle="Structured investigations" />
          <StatCard icon="◆" label="Evidence Records" value={dashboard.evidence_records} subtitle="Seed + contextual evidence" />
          <StatCard icon="◎" label="Contextual Cases" value={dashboard.contextual_cases} subtitle={`${dashboard.contextual_enrichment.toFixed(2)}% enrichment`} />
        </div>
      </section>

      <section className="two-column-section">
        <Panel>
          <div className="panel-header">
            <div>
              <p className="section-label">MACHINE LEARNING</p>
              <h2>Fraud Detection Model</h2>
            </div>
            <span className="model-badge">{dashboard.selected_model}</span>
          </div>

          <div className="model-highlight">
            <div>
              <span>Selected Model</span>
              <strong>{dashboard.selected_model}</strong>
            </div>
            <div>
              <span>Forensic Threshold</span>
              <strong>{dashboard.forensic_threshold}</strong>
            </div>
          </div>

          <div className="metric-list">
            <MetricRow label="ROC-AUC" value={dashboard.roc_auc.toFixed(4)} />
            <MetricRow label="PR-AUC" value={dashboard.pr_auc.toFixed(4)} />
            <MetricRow label="Suspicious Seeds" value={dashboard.suspicious_seeds} />
          </div>
        </Panel>

        <Panel>
          <div className="panel-header">
            <div>
              <p className="section-label">FORENSIC ANALYSIS</p>
              <h2>Evidence Integrity</h2>
            </div>
            <span className="success-badge">Evaluated</span>
          </div>

          <div className="donut-grid compact">
            <MiniDonut value={dashboard.seed_traceability} label="Seed" />
            <MiniDonut value={dashboard.evidence_traceability} label="Evidence" />
            <MiniDonut value={dashboard.finding_traceability} label="Finding" />
          </div>
        </Panel>
      </section>

      <Panel>
        <div className="panel-header">
          <div>
            <p className="section-label">RESEARCH WORKFLOW</p>
            <h2>Detection to Evidence Chain</h2>
          </div>
        </div>
        <Workflow />
      </Panel>

      <Panel>
        <div className="panel-header">
          <div>
            <p className="section-label">KEY FINDINGS</p>
            <h2>Research Results</h2>
          </div>
        </div>

        <div className="results-grid">
          <ResultCard number="01" value="480" text="Suspicious seeds generated" />
          <ResultCard number="02" value="480" text="Candidate forensic cases" />
          <ResultCard number="03" value="491" text="Traceable evidence records" />
          <ResultCard number="04" value="2.29%" text="Contextual enrichment" />
        </div>
      </Panel>
    </>
  );
}

function Methodology() {
  const stages = [
    ["01", "Dataset Inspection", "PaySim transaction data was inspected for structure, missing values, duplicates, transaction types and fraud distribution."],
    ["02", "Preprocessing", "Transactions were prepared using the implemented preprocessing pipeline while preserving temporal ordering."],
    ["03", "Feature Engineering", "Historical and transaction-level features were generated using a leakage-audited feature engineering process."],
    ["04", "Model Training", "Logistic Regression and Random Forest models were evaluated using temporal train, validation and test partitions."],
    ["05", "Suspicion Generation", "The audited Random Forest generated investigation seeds using the frozen forensic threshold of 0.95."],
    ["06", "Case Reconstruction", "Suspicious seeds were reconstructed into candidate forensic cases using temporal and account relationships."],
    ["07", "Evidence Evaluation", "Cases were evaluated for structural completeness, contextual enrichment and traceability."],
    ["08", "Window Sensitivity", "Temporal reconstruction was evaluated using ±1, ±6, ±12 and ±24 PaySim-step windows."],
  ];

  return (
    <>
      <PageHeader
        eyebrow="RESEARCH METHODOLOGY"
        title="From Detection to Digital Evidence"
        description="The implemented workflow transforms analytical fraud outputs into structured investigative cases with explicit evidence traceability."
        badge="8 Stages"
      />

      <Panel className="wide-panel">
        <div className="panel-header">
          <div>
            <p className="section-label">END-TO-END PIPELINE</p>
            <h2>Research Architecture</h2>
          </div>
        </div>
        <Workflow />
      </Panel>

      <section>
        <SectionTitle label="IMPLEMENTATION STAGES" title="Methodology Pipeline" />
        <div className="methodology-grid">
          {stages.map(([number, title, text]) => (
            <div className="methodology-card" key={number}>
              <div className="methodology-number">{number}</div>
              <div>
                <h3>{title}</h3>
                <p>{text}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="two-column-section">
        <Panel>
          <p className="section-label">MODEL CONFIGURATION</p>
          <h2>Audited Random Forest</h2>
          <div className="metric-list methodology-metrics">
            <MetricRow label="ROC-AUC" value="0.9703" />
            <MetricRow label="PR-AUC" value="0.3153" />
            <MetricRow label="Forensic Threshold" value="0.95" />
            <MetricRow label="Suspicious Seeds" value="480" />
          </div>
        </Panel>

        <Panel>
          <p className="section-label">FORENSIC CONFIGURATION</p>
          <h2>Case Reconstruction</h2>
          <div className="metric-list methodology-metrics">
            <MetricRow label="Candidate Cases" value="480" />
            <MetricRow label="Evidence Records" value="491" />
            <MetricRow label="Contextual Cases" value="11" />
            <MetricRow label="Selected Window" value="±24 steps" />
          </div>
        </Panel>
      </section>
    </>
  );
}

function FraudDetection() {
  const { data, loading, error } = useApi("/api/model-comparison");
  const { data: features } = useApi("/api/feature-importance");

  if (loading) return <Loading text="Loading model comparison..." />;
  if (error) return <ErrorState error={error} />;

  const rows = Array.isArray(data) ? data : [];
  const selected = rows.find((row) => row.model === "Random Forest") ?? rows[0] ?? {};

  return (
    <>
      <PageHeader
        eyebrow="MACHINE LEARNING"
        title="Fraud Detection Results"
        description="Audited model comparison used to identify suspicious financial transactions for forensic investigation."
        badge="Audited Models"
      />

      <div className="stats-grid">
        <StatCard icon="◎" label="Random Forest ROC-AUC" value={fixed(selected.roc_auc)} subtitle="Selected model" />
        <StatCard icon="◆" label="Random Forest PR-AUC" value={fixed(selected.pr_auc)} subtitle="Selected model" />
        <StatCard icon="⌁" label="Forensic Threshold" value="0.95" subtitle="Frozen after validation" />
      </div>

      <section className="two-column-section asymmetric">
        <Panel>
          <div className="panel-header">
            <div>
              <p className="section-label">MODEL PERFORMANCE</p>
              <h2>Comparative Metrics</h2>
            </div>
            <span className="dataset-badge">{rows.length} models</span>
          </div>
          <PerformanceChart rows={rows} />
        </Panel>

        <Panel>
          <div className="panel-header">
            <div>
              <p className="section-label">FEATURE ANALYSIS</p>
              <h2>Top Feature Importance</h2>
            </div>
          </div>
          <FeatureImportanceChart features={Array.isArray(features) ? features : []} />
        </Panel>
      </section>

      <Panel>
        <div className="panel-header">
          <div>
            <p className="section-label">MODEL COMPARISON</p>
            <h2>Audited Model Performance Table</h2>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="research-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Threshold</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>ROC-AUC</th>
                <th>PR-AUC</th>
                <th>TP</th>
                <th>FP</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.model} className={row.model === "Random Forest" ? "selected-row" : ""}>
                  <td>{row.model}</td>
                  <td>{row.threshold}</td>
                  <td>{percent(row.accuracy)}</td>
                  <td>{percent(row.precision)}</td>
                  <td>{percent(row.recall)}</td>
                  <td>{percent(row.f1)}</td>
                  <td>{fixed(row.roc_auc)}</td>
                  <td>{fixed(row.pr_auc)}</td>
                  <td>{intFormat(row.true_positive)}</td>
                  <td>{intFormat(row.false_positive)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="research-notice">
        <div className="notice-icon">!</div>
        <div>
          <h3>Evaluation Meaning</h3>
          <p>
            Model-comparison metrics use threshold 0.5. The forensic workflow separately freezes
            threshold 0.95 for high-confidence suspicious seed generation, producing 480 investigation seeds.
          </p>
        </div>
      </div>
    </>
  );
}

function SuspiciousTransactions() {
  const { data, loading, error } = useApi("/api/case-seeds");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const pageSize = 12;

  const rows = useMemo(() => {
    if (!Array.isArray(data)) return [];
    const query = search.trim().toLowerCase();
    if (!query) return data;
    return data.filter((row) => Object.values(row).some((value) => String(value).toLowerCase().includes(query)));
  }, [data, search]);

  useEffect(() => setPage(1), [search]);

  if (loading) return <Loading text="Loading suspicious seed transactions..." />;
  if (error) return <ErrorState error={error} />;

  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  const visibleRows = rows.slice((page - 1) * pageSize, page * pageSize);

  return (
    <>
      <PageHeader
        eyebrow="SUSPICIOUS TRANSACTIONS"
        title="Suspicious Investigation Seeds"
        description="High-confidence Random Forest outputs selected using the frozen forensic threshold. These are starting points for investigation."
        badge={`${rows.length} seeds`}
      />

      <Panel>
        <div className="toolbar">
          <div>
            <p className="section-label">SEED RECORDS</p>
            <h2>Case Seed Transactions</h2>
          </div>
          <input className="search-input" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search transaction, account or case..." />
        </div>

        <div className="table-wrapper">
          <table className="research-table">
            <thead>
              <tr>
                <th>Case</th>
                <th>Transaction</th>
                <th>Step</th>
                <th>Type</th>
                <th>Amount</th>
                <th>Origin</th>
                <th>Destination</th>
                <th>Score</th>
                <th>Known Fraud</th>
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row, index) => {
                const key = `${firstExisting(row, ["case_id"], "case")}-${firstExisting(row, ["transaction_id", "seed_transaction_id"], index)}`;
                return (
                  <tr key={key}>
                    <td><span className="case-pill">{firstExisting(row, ["case_id"])}</span></td>
                    <td className="mono">{firstExisting(row, ["transaction_id", "seed_transaction_id"])}</td>
                    <td>{firstExisting(row, ["step"])}</td>
                    <td>{firstExisting(row, ["type"])}</td>
                    <td>{amountFormat(firstExisting(row, ["amount"]))}</td>
                    <td className="mono">{firstExisting(row, ["nameOrig", "origin"])}</td>
                    <td className="mono">{firstExisting(row, ["nameDest", "destination"])}</td>
                    <td><span className="score-pill">{fixed(firstExisting(row, ["suspicion_score", "fraud_probability", "score", "maximum_seed_suspicion_score"]))}</span></td>
                    <td>{numberValue(firstExisting(row, ["isFraud"])) === 1 ? <span className="fraud-pill">Fraud</span> : <span className="genuine-pill">Not flagged</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <Pagination page={page} totalPages={totalPages} total={rows.length} onChange={setPage} />
      </Panel>
    </>
  );
}

function ForensicCases() {
  const { data, loading, error } = useApi("/api/cases");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [selectedCase, setSelectedCase] = useState(null);
  const pageSize = 9;

  const rows = useMemo(() => {
    if (!Array.isArray(data)) return [];
    const query = search.trim().toLowerCase();
    if (!query) return data;
    return data.filter((row) => Object.values(row).some((value) => String(value).toLowerCase().includes(query)));
  }, [data, search]);

  useEffect(() => setPage(1), [search]);

  if (loading) return <Loading text="Loading forensic cases..." />;
  if (error) return <ErrorState error={error} />;

  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  const visibleCases = rows.slice((page - 1) * pageSize, page * pageSize);

  return (
    <>
      <PageHeader
        eyebrow="FORENSIC CASES"
        title="Candidate Case Reconstruction"
        description="Suspicious transaction seeds reconstructed into structured candidate cases with linked accounts, evidence and findings."
        badge={`${rows.length} cases`}
      />

      <Panel>
        <div className="toolbar">
          <div>
            <p className="section-label">CASE INDEX</p>
            <h2>Forensic Candidate Cases</h2>
          </div>
          <input className="search-input" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search case, transaction or account..." />
        </div>

        <div className="case-grid">
          {visibleCases.map((row) => (
            <button className="case-card" key={row.case_id} onClick={() => setSelectedCase(row.case_id)}>
              <div className="case-card-top">
                <span className="case-pill">{row.case_id}</span>
                <span className={numberValue(row.context_transaction_count) > 0 ? "success-badge" : "dataset-badge"}>
                  {numberValue(row.context_transaction_count) > 0 ? "Context" : "Seed only"}
                </span>
              </div>
              <h3>{amountFormat(row.total_transaction_amount)}</h3>
              <div className="case-card-metrics">
                <span>Seeds<strong>{intFormat(row.seed_count)}</strong></span>
                <span>Evidence<strong>{intFormat(row.evidence_transaction_count)}</strong></span>
                <span>Accounts<strong>{intFormat(row.unique_account_count)}</strong></span>
              </div>
              <div className="case-card-footer">
                Step {row.start_step} to {row.end_step} · Score {fixed(row.maximum_seed_suspicion_score)}
              </div>
            </button>
          ))}
        </div>

        <Pagination page={page} totalPages={totalPages} total={rows.length} onChange={setPage} />
      </Panel>

      {selectedCase && <CaseDetail caseId={selectedCase} onClose={() => setSelectedCase(null)} />}
    </>
  );
}

function CaseDetail({ caseId, onClose }) {
  const { data: caseData, loading, error } = useApi(`/api/cases/${caseId}`);
  const { data: evidenceData, loading: evidenceLoading } = useApi(`/api/cases/${caseId}/evidence`);
  const evidence = Array.isArray(evidenceData) ? evidenceData : [];

  return (
    <div className="case-detail-overlay" role="dialog" aria-modal="true">
      <div className="case-detail-panel">
        <div className="case-detail-header">
          <div>
            <h2>{caseId}</h2>
            <p>Forensic case reconstruction and source evidence chain</p>
          </div>
          <button className="close-button" onClick={onClose} aria-label="Close case details">×</button>
        </div>

        {(loading || evidenceLoading) && <Loading text="Loading case evidence..." />}
        {error && <ErrorState error={error} />}

        {!loading && !error && caseData && (
          <>
            <CaseReconstructionVisual caseData={caseData} evidence={evidence} />

            <div className="case-detail-stats">
              <MetricRow label="Seed Count" value={caseData.seed_count} />
              <MetricRow label="Evidence Transactions" value={caseData.evidence_transaction_count} />
              <MetricRow label="Context Transactions" value={caseData.context_transaction_count} />
              <MetricRow label="Unique Accounts" value={caseData.unique_account_count} />
              <MetricRow label="Step Range" value={`${caseData.start_step} → ${caseData.end_step}`} />
              <MetricRow label="Time Span" value={`${caseData.time_span_steps} steps`} />
            </div>

            <div className="finding-box">
              <p className="section-label">FINDING</p>
              <p>{caseData.finding}</p>
            </div>

            <div className="panel-inner">
              <div className="panel-header">
                <div>
                  <p className="section-label">EVIDENCE</p>
                  <h3>Source Transaction Records</h3>
                </div>
              </div>

              <div className="table-wrapper">
                <table className="research-table">
                  <thead>
                    <tr>
                      <th>Transaction</th>
                      <th>Role</th>
                      <th>Step</th>
                      <th>Type</th>
                      <th>Amount</th>
                      <th>Origin</th>
                      <th>Destination</th>
                      <th>Known Fraud</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evidence.map((row) => (
                      <tr key={row.transaction_id}>
                        <td className="mono">{row.transaction_id}</td>
                        <td><span className="evidence-role">{row.evidence_role}</span></td>
                        <td>{row.step}</td>
                        <td>{row.type}</td>
                        <td>{amountFormat(row.amount)}</td>
                        <td className="mono">{row.nameOrig}</td>
                        <td className="mono">{row.nameDest}</td>
                        <td>{numberValue(row.isFraud) === 1 ? <span className="fraud-pill">Fraud</span> : <span className="genuine-pill">Not flagged</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function EvidenceTraceability() {
  const { data, loading, error } = useApi("/api/traceability");
  const [search, setSearch] = useState("");

  const allRows = Array.isArray(data) ? data : [];
  const summary = useMemo(() => {
    const traceable = allRows.filter((row) => row.case_fully_traceable === true || row.case_fully_traceable === "True").length;
    const contextual = allRows.filter((row) => row.has_contextual_evidence === true || row.has_contextual_evidence === "True").length;
    const evidenceRecords = allRows.reduce((sum, row) => sum + numberValue(row.total_evidence_records), 0);
    const traceableRecords = allRows.reduce((sum, row) => sum + numberValue(row.traceable_evidence_records), 0);
    return { traceable, contextual, evidenceRecords, traceableRecords };
  }, [allRows]);

  const rows = useMemo(() => {
    const query = search.trim().toLowerCase();
    const filtered = query
      ? allRows.filter((row) => Object.values(row).some((value) => String(value).toLowerCase().includes(query)))
      : allRows;
    return filtered.slice(0, 100);
  }, [allRows, search]);

  if (loading) return <Loading text="Loading traceability records..." />;
  if (error) return <ErrorState error={error} />;

  return (
    <>
      <PageHeader
        eyebrow="EVIDENCE TRACEABILITY"
        title="Evidence Chain Traceability"
        description="Verification that case evidence and findings remain linked to their source transaction records."
        badge="100% Traceability"
      />

      <div className="stats-grid">
        <StatCard icon="✓" label="Fully Traceable Cases" value={`${summary.traceable}/${allRows.length}`} subtitle="Case-level traceability" />
        <StatCard icon="◆" label="Traceable Records" value={`${summary.traceableRecords}/${summary.evidenceRecords}`} subtitle="Evidence-level linkage" />
        <StatCard icon="◎" label="Contextual Cases" value={summary.contextual} subtitle="Cases with added context" />
      </div>

      <section className="two-column-section">
        <Panel>
          <div className="panel-header">
            <div>
              <p className="section-label">TRACEABILITY SUMMARY</p>
              <h2>Evidence Chain Integrity</h2>
            </div>
          </div>
          <div className="donut-grid">
            <MiniDonut value={100} label="Seed" sublabel="Every case has a seed reference" />
            <MiniDonut value={100} label="Evidence" sublabel="Every evidence record is linked" />
            <MiniDonut value={100} label="Finding" sublabel="Every case preserves a finding" />
          </div>
        </Panel>

        <Panel>
          <div className="panel-header">
            <div>
              <p className="section-label">CASE MIX</p>
              <h2>Seed and Context Evidence</h2>
            </div>
          </div>
          <div className="trace-stack">
            <ProgressPill label="Fully traceable cases" value={(summary.traceable / Math.max(allRows.length, 1)) * 100} />
            <ProgressPill label="Cases with contextual evidence" value={(summary.contextual / Math.max(allRows.length, 1)) * 100} />
            <ProgressPill label="Traceable evidence records" value={(summary.traceableRecords / Math.max(summary.evidenceRecords, 1)) * 100} />
          </div>
        </Panel>
      </section>

      <Panel>
        <div className="toolbar">
          <div>
            <p className="section-label">CASE TRACEABILITY</p>
            <h2>Traceability Records</h2>
          </div>
          <input className="search-input" type="text" placeholder="Search case ID..." value={search} onChange={(event) => setSearch(event.target.value)} />
        </div>

        <div className="table-wrapper">
          <table className="research-table">
            <thead>
              <tr>
                <th>Case</th>
                <th>Evidence Records</th>
                <th>Traceable Records</th>
                <th>Evidence Rate</th>
                <th>Seed</th>
                <th>Context</th>
                <th>Finding</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.case_id}>
                  <td><span className="case-pill">{row.case_id}</span></td>
                  <td>{row.total_evidence_records}</td>
                  <td>{row.traceable_evidence_records}</td>
                  <td>{percent(row.evidence_traceability_rate, 0)}</td>
                  <td>{row.has_suspicious_seed ? "Yes" : "No"}</td>
                  <td>{row.has_contextual_evidence ? "Yes" : "No"}</td>
                  <td>{row.finding_exists ? "Yes" : "No"}</td>
                  <td>{row.case_fully_traceable ? <span className="success-badge">Traceable</span> : <span className="warning-badge">Review</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="table-note">Showing up to 100 records for interface performance. The underlying research output contains all 480 cases.</p>
      </Panel>
    </>
  );
}

function WindowSensitivity() {
  const { data, loading, error } = useApi("/api/window-sensitivity");

  if (loading) return <Loading text="Loading window sensitivity..." />;
  if (error) return <ErrorState error={error} />;

  const rows = Array.isArray(data) ? data : [];
  const selected = rows.find((row) => numberValue(row.window_steps) === 24) ?? rows.at(-1) ?? {};

  return (
    <>
      <PageHeader
        eyebrow="TEMPORAL SENSITIVITY"
        title="Window Sensitivity Analysis"
        description="Comparison of contextual evidence enrichment across tested temporal reconstruction windows."
        badge="±24 Selected"
      />

      <div className="stats-grid">
        <StatCard icon="⌁" label="Selected Window" value={`±${selected.window_steps ?? 24}`} subtitle="PaySim steps" />
        <StatCard icon="▣" label="Contextual Cases" value={intFormat(selected.cases_with_context)} subtitle={`${percent(selected.context_enrichment_rate)} enrichment`} />
        <StatCard icon="◆" label="Total Evidence" value={intFormat(selected.total_case_transactions)} subtitle={`${fixed(selected.average_evidence_per_case)} records/case`} />
      </div>

      <Panel>
        <div className="panel-header">
          <div>
            <p className="section-label">CONTEXTUAL ENRICHMENT</p>
            <h2>Temporal Window Comparison</h2>
          </div>
        </div>
        <WindowSensitivityChart rows={rows} />
      </Panel>

      <Panel>
        <div className="panel-header">
          <div>
            <p className="section-label">EXPERIMENTAL RESULTS</p>
            <h2>Window Sensitivity Table</h2>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="research-table">
            <thead>
              <tr>
                <th>Window</th>
                <th>Approx. Hours</th>
                <th>Candidate Cases</th>
                <th>Cases with Context</th>
                <th>Without Context</th>
                <th>Enrichment</th>
                <th>Avg. Evidence / Case</th>
                <th>Total Evidence</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.window_steps} className={numberValue(row.window_steps) === 24 ? "selected-row" : ""}>
                  <td>±{row.window_steps}</td>
                  <td>{row.window_hours_approx}</td>
                  <td>{row.candidate_cases}</td>
                  <td>{row.cases_with_context}</td>
                  <td>{row.cases_without_context}</td>
                  <td>{percent(row.context_enrichment_rate)}</td>
                  <td>{fixed(row.average_evidence_per_case)}</td>
                  <td>{row.total_case_transactions}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="research-notice sensitivity-note">
          <div className="notice-icon">✓</div>
          <div>
            <h3>Selected Configuration</h3>
            <p>
              The ±24 PaySim-step window produced the highest contextual enrichment among the tested windows:
              <strong> 11 cases (2.29%)</strong>. This is an empirical result for the tested configuration, not a universal optimal window.
            </p>
          </div>
        </div>
      </Panel>
    </>
  );
}

function Results() {
  const { data, loading, error } = useApi("/api/forensic-evaluation");
  const { data: modelRows } = useApi("/api/model-comparison");
  const { data: windowRows } = useApi("/api/window-sensitivity");

  if (loading) return <Loading text="Loading final research results..." />;
  if (error) return <ErrorState error={error} />;

  const rows = Array.isArray(data) ? data : [];
  const metrics = Object.fromEntries(rows.map((row) => [row.metric, row.value]));
  const selectedModel = Array.isArray(modelRows) ? modelRows.find((row) => row.model === "Random Forest") : null;
  const selectedWindow = Array.isArray(windowRows) ? windowRows.find((row) => numberValue(row.window_steps) === 24) : null;

  return (
    <>
      <PageHeader
        eyebrow="FINAL EVALUATION"
        title="Results & Analysis"
        description="Consolidated forensic evaluation results from the completed PaySim research implementation."
        badge="Final Results"
      />

      <div className="stats-grid">
        <StatCard icon="▣" label="Candidate Cases" value={metricFormat(metrics["Candidate forensic cases"])} subtitle="Structurally complete" />
        <StatCard icon="◆" label="Evidence Records" value={metricFormat(metrics["Total evidence records"])} subtitle="Seed + contextual records" />
        <StatCard icon="◎" label="Contextual Enrichment" value={`${metricFormat(metrics["Context enrichment rate"])}%`} subtitle="11 of 480 cases" />
        <StatCard icon="✓" label="Traceability" value="100%" subtitle="Seed, evidence and finding" />
      </div>

      <section className="two-column-section">
        <Panel>
          <div className="panel-header">
            <div>
              <p className="section-label">RESULTS LAYER</p>
              <h2>Research Performance Snapshot</h2>
            </div>
          </div>
          <div className="summary-matrix">
            <div>
              <span>ROC-AUC</span>
              <strong>{selectedModel ? fixed(selectedModel.roc_auc) : "0.9703"}</strong>
            </div>
            <div>
              <span>PR-AUC</span>
              <strong>{selectedModel ? fixed(selectedModel.pr_auc) : "0.3153"}</strong>
            </div>
            <div>
              <span>Final Window</span>
              <strong>±{selectedWindow?.window_steps ?? 24}</strong>
            </div>
            <div>
              <span>Threshold</span>
              <strong>{metricFormat(metrics["Final suspicion threshold"])}</strong>
            </div>
          </div>
        </Panel>

        <Panel>
          <div className="panel-header">
            <div>
              <p className="section-label">TRACEABILITY</p>
              <h2>Evidence Chain Integrity</h2>
            </div>
          </div>
          <div className="traceability-large">
            <ProgressPill label="Seed Traceability" value={metrics["Seed traceability"]} />
            <ProgressPill label="Evidence Traceability" value={metrics["Evidence traceability"]} />
            <ProgressPill label="Finding Traceability" value={metrics["Finding traceability"]} />
          </div>
        </Panel>
      </section>

      <Panel>
        <div className="panel-header">
          <div>
            <p className="section-label">COMPLETE RESULTS TABLE</p>
            <h2>Final Forensic Evaluation</h2>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="research-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Unit</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.metric}>
                  <td>{row.metric}</td>
                  <td>{metricFormat(row.value)}</td>
                  <td>{row.unit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel>
        <div className="final-interpretation">
          <p className="section-label">RESEARCH INTERPRETATION</p>
          <h2>What the Results Demonstrate</h2>
          <p>
            The implemented workflow successfully transformed analytically suspicious transactions into structured candidate forensic cases with explicit source references.
          </p>
          <p>
            All 480 generated cases were structurally complete and achieved 100% seed, evidence and finding traceability. Only 11 cases received additional contextual evidence, producing 2.29% contextual enrichment.
          </p>
          <p>
            The results demonstrate strong structural traceability within the available PaySim transactional evidence, while showing the limitation of relying only on simulated transaction records for richer contextual reconstruction.
          </p>
        </div>
      </Panel>
    </>
  );
}

function Pagination({ page, totalPages, onChange, total }) {
  if (totalPages <= 1) return <div className="pagination-summary">{total} records</div>;

  return (
    <div className="pagination">
      <span>Page {page} of {totalPages} · {total} records</span>
      <div className="pagination-buttons">
        <button disabled={page === 1} onClick={() => onChange(Math.max(1, page - 1))}>Previous</button>
        <button disabled={page === totalPages} onClick={() => onChange(Math.min(totalPages, page + 1))}>Next</button>
      </div>
    </div>
  );
}

function App() {
  const [activePage, setActivePage] = useState("dashboard");

  const {
    data: dashboard,
    loading: dashboardLoading,
    error: dashboardError,
  } = useApi("/api/dashboard");

  function renderPage() {
    if (activePage === "dashboard") {
      if (dashboardLoading) return <Loading text="Loading forensic research dashboard..." />;
      if (dashboardError) return <ErrorState error={dashboardError} />;
      return <Dashboard dashboard={dashboard} />;
    }

    if (activePage === "methodology") return <Methodology />;
    if (activePage === "detection") return <FraudDetection />;
    if (activePage === "suspicious") return <SuspiciousTransactions />;
    if (activePage === "cases") return <ForensicCases />;
    if (activePage === "evidence") return <EvidenceTraceability />;
    if (activePage === "sensitivity") return <WindowSensitivity />;
    if (activePage === "results") return <Results />;

    return null;
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">DF</div>
          <div>
            <h1>Digital Forensics</h1>
            <p>Research Demo</p>
          </div>
        </div>

        <nav className="navigation" aria-label="Research demo navigation">
          {navigationItems.map(([id, icon, label]) => (
            <button key={id} className={`nav-item ${activePage === id ? "active" : ""}`} onClick={() => setActivePage(id)}>
              <span>{icon}</span>
              {label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <p>PaySim Dataset</p>
          <span>Forensic Investigation System</span>
        </div>
      </aside>

      <main className="main-content">
        <div className="mobile-titlebar">
          <div className="mobile-brand">
            <div className="brand-logo">DF</div>
            <div>
              <strong>Digital Forensics</strong>
              <span>Research Demo</span>
            </div>
          </div>
        </div>

        {renderPage()}

        <footer className="footer">
          <div>
            <strong>Role of Digital Forensics in Banking and Financial Cybercrime Investigation</strong>
            <p>M.Sc. Computer Science · Semester I · Thakur College of Science & Commerce</p>
          </div>
          <span>PaySim Research Demo</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
