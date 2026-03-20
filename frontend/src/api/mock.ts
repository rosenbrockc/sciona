import type {
  BountyResponse,
  BountySummaryResponse,
  AtomDetailResponse,
  AtomSummaryResponse,
  AtomVersionResponse,
  LeaderboardEntry,
  OriginatorImpact,
  ComputePreserved,
  BenchmarkRecord,
  SubmissionLeaderboardEntry,
  SettlementInfo,
  PaginatedResponse,
} from "./types";

const bounties: BountyResponse[] = [
  {
    bounty_id: "b-001",
    title: "Protein folding stability predictor",
    escrow_amount: 5000,
    deadline: "2026-06-01",
    tier: "premium",
    status: "active",
    domain_tags: ["bioinformatics", "structural-biology"],
    verification_budget_used: 2,
    verification_budget_total: 10,
    created_at: "2026-01-15",
  },
  {
    bounty_id: "b-002",
    title: "Time-series anomaly detection for IoT sensors",
    escrow_amount: 2500,
    deadline: "2026-05-01",
    tier: "standard",
    status: "active",
    domain_tags: ["time-series", "iot"],
    verification_budget_used: 1,
    verification_budget_total: 5,
    created_at: "2026-02-01",
  },
  {
    bounty_id: "b-003",
    title: "Multilingual NER for legal documents",
    escrow_amount: 8000,
    deadline: "2026-04-15",
    tier: "premium",
    status: "verification",
    domain_tags: ["nlp", "legal"],
    verification_budget_used: 8,
    verification_budget_total: 10,
    created_at: "2025-12-01",
  },
  {
    bounty_id: "b-004",
    title: "Graph-based recommendation engine",
    escrow_amount: 3000,
    deadline: "2026-03-01",
    tier: "standard",
    status: "settled",
    domain_tags: ["graph-ml", "recommendations"],
    verification_budget_used: 5,
    verification_budget_total: 5,
    created_at: "2025-11-01",
  },
  {
    bounty_id: "b-005",
    title: "Climate model downscaling with diffusion",
    escrow_amount: 12000,
    deadline: "2026-08-01",
    tier: "premium",
    status: "open",
    domain_tags: ["climate", "diffusion-models"],
    verification_budget_used: 0,
    verification_budget_total: 10,
    created_at: "2026-03-10",
  },
];

const atoms: AtomSummaryResponse[] = [
  { fqdn: "bio.fold.stability-v2", description: "Protein stability prediction using SE(3) equivariant networks", domain_tags: ["bioinformatics"], latest_version: "2.1.0" },
  { fqdn: "ts.anomaly.spectral", description: "Spectral residual anomaly detector for multivariate time series", domain_tags: ["time-series"], latest_version: "1.3.0" },
  { fqdn: "nlp.ner.multilingual", description: "Multilingual named entity recognition with cross-lingual transfer", domain_tags: ["nlp"], latest_version: "3.0.1" },
  { fqdn: "graph.recsys.lightgcn", description: "LightGCN-based collaborative filtering", domain_tags: ["graph-ml", "recommendations"], latest_version: "1.0.0" },
  { fqdn: "climate.downscale.diffusion", description: "Diffusion-based climate model downscaling", domain_tags: ["climate", "diffusion-models"], latest_version: "0.9.0" },
  { fqdn: "cv.segment.sam-lite", description: "Lightweight SAM variant for edge deployment", domain_tags: ["computer-vision"], latest_version: "1.2.0" },
  { fqdn: "opt.hyperband.async", description: "Asynchronous Hyperband for distributed HPO", domain_tags: ["optimization"], latest_version: "2.0.0" },
  { fqdn: "nlp.summarize.led", description: "Longformer Encoder-Decoder for long document summarization", domain_tags: ["nlp"], latest_version: "1.1.0" },
  { fqdn: "audio.denoise.conv-tasnet", description: "Conv-TasNet for single-channel speech separation", domain_tags: ["audio"], latest_version: "1.0.2" },
  { fqdn: "tabular.boost.catboost-tuned", description: "Auto-tuned CatBoost wrapper with Optuna integration", domain_tags: ["tabular"], latest_version: "3.2.1" },
];

const atomDetail: AtomDetailResponse = {
  atom_id: "a-001",
  fqdn: "bio.fold.stability-v2",
  description: "Protein stability prediction using SE(3) equivariant networks. Predicts ΔΔG values for single-point mutations with state-of-the-art accuracy on the S669 benchmark.",
  domain_tags: ["bioinformatics", "structural-biology"],
  latest_version: "2.1.0",
  created_at: "2025-09-15",
  authors: ["alice@research.org", "bob@university.edu"],
};

const atomVersions: AtomVersionResponse[] = [
  { version_id: "v-003", version: "2.1.0", fingerprint: "sha256:a1b2c3d4e5f6", is_latest: true, published_at: "2026-02-10" },
  { version_id: "v-002", version: "2.0.0", fingerprint: "sha256:f6e5d4c3b2a1", is_latest: false, published_at: "2025-12-01" },
  { version_id: "v-001", version: "1.0.0", fingerprint: "sha256:1a2b3c4d5e6f", is_latest: false, published_at: "2025-09-15" },
];

const leaderboard: LeaderboardEntry[] = [
  { rank: 1, username: "alice_ml", h_index: 12, bounty_count: 8, total_value: 45000, atom_count: 15 },
  { rank: 2, username: "bob_research", h_index: 9, bounty_count: 6, total_value: 32000, atom_count: 11 },
  { rank: 3, username: "carol_ai", h_index: 7, bounty_count: 5, total_value: 28000, atom_count: 9 },
  { rank: 4, username: "dave_bio", h_index: 6, bounty_count: 4, total_value: 19000, atom_count: 7 },
  { rank: 5, username: "eve_data", h_index: 5, bounty_count: 3, total_value: 14000, atom_count: 6 },
  { rank: 6, username: "frank_cv", h_index: 4, bounty_count: 3, total_value: 11000, atom_count: 5 },
  { rank: 7, username: "grace_nlp", h_index: 3, bounty_count: 2, total_value: 8000, atom_count: 4 },
  { rank: 8, username: "hank_opt", h_index: 2, bounty_count: 2, total_value: 5500, atom_count: 3 },
];

const computePreserved: ComputePreserved = {
  estimated_tokens_saved: 2_400_000_000,
  estimated_cost_saved_usd: 48_000,
  total_bounties_settled: 34,
  total_escrow_distributed: 185_000,
  cross_discipline_atoms: 23,
};

const benchmarks: BenchmarkRecord[] = [
  { atom_fqdn: "bio.fold.stability-v2", metric_name: "spearman_rho", metric_value: 0.847, dataset_tag: "S669", recorded_at: "2026-02-10" },
  { atom_fqdn: "bio.fold.stability-v2", metric_name: "rmse", metric_value: 1.23, dataset_tag: "S669", recorded_at: "2026-02-10" },
  { atom_fqdn: "bio.fold.stability-v2", metric_name: "spearman_rho", metric_value: 0.812, dataset_tag: "Ssym", recorded_at: "2026-02-10" },
];

const submissionLeaderboard: SubmissionLeaderboardEntry[] = [
  { rank: 1, submission_id: "s-101", architect_id: "alice_ml", metric_values: { spearman_rho: 0.847, rmse: 1.23 }, verified_at: "2026-02-15" },
  { rank: 2, submission_id: "s-102", architect_id: "bob_research", metric_values: { spearman_rho: 0.821, rmse: 1.45 }, verified_at: "2026-02-14" },
  { rank: 3, submission_id: "s-103", architect_id: "carol_ai", metric_values: { spearman_rho: 0.798, rmse: 1.67 }, verified_at: "2026-02-13" },
];

const settlement: SettlementInfo = {
  bounty_id: "b-004",
  status: "settled",
  winning_submission_id: "s-201",
  payouts: [
    { recipient_id: "alice_ml", role: "architect", amount: 2100, stripe_account: "acct_xxx1" },
    { recipient_id: "bob_research", role: "originator", amount: 600, stripe_account: "acct_xxx2" },
    { recipient_id: "platform", role: "platform", amount: 300, stripe_account: "acct_platform" },
  ],
};

const originatorImpact: OriginatorImpact = {
  originator_id: "alice_ml",
  username: "alice_ml",
  h_index: 12,
  bounty_count: 8,
  total_value: 45000,
  atom_count: 15,
  atoms: atoms.slice(0, 4),
};

// --- Mock API functions ---

export const mockApi = {
  getBounties(params?: { status?: string; domain_tag?: string; limit?: number; offset?: number }): PaginatedResponse<BountySummaryResponse> {
    let filtered = bounties.map(({ bounty_id, title, escrow_amount, status, domain_tags, deadline }) => ({
      bounty_id, title, escrow_amount, status, domain_tags, deadline,
    }));
    if (params?.status) filtered = filtered.filter((b) => b.status === params.status);
    if (params?.domain_tag) filtered = filtered.filter((b) => b.domain_tags.includes(params.domain_tag!));
    const offset = params?.offset ?? 0;
    const limit = params?.limit ?? 20;
    return { items: filtered.slice(offset, offset + limit), total: filtered.length, limit, offset };
  },

  getBounty(id: string): BountyResponse | undefined {
    return bounties.find((b) => b.bounty_id === id);
  },

  getBountyLeaderboard(_id: string): SubmissionLeaderboardEntry[] {
    return submissionLeaderboard;
  },

  getBountySettlement(_id: string): SettlementInfo {
    return settlement;
  },

  getAtoms(params?: { search?: string; domain_tag?: string; limit?: number; offset?: number }): PaginatedResponse<AtomSummaryResponse> {
    let filtered = [...atoms];
    if (params?.search) {
      const q = params.search.toLowerCase();
      filtered = filtered.filter((a) => a.fqdn.includes(q) || a.description.toLowerCase().includes(q));
    }
    if (params?.domain_tag) filtered = filtered.filter((a) => a.domain_tags.includes(params.domain_tag!));
    const offset = params?.offset ?? 0;
    const limit = params?.limit ?? 20;
    return { items: filtered.slice(offset, offset + limit), total: filtered.length, limit, offset };
  },

  getAtom(_fqdn: string): AtomDetailResponse {
    return atomDetail;
  },

  getAtomVersions(_fqdn: string): AtomVersionResponse[] {
    return atomVersions;
  },

  getAtomBenchmarks(_fqdn: string): BenchmarkRecord[] {
    return benchmarks;
  },

  getAtomBibtex(fqdn: string): string {
    return `@software{${fqdn.replace(/\./g, "_")},\n  title = {${atomDetail.description.split(".")[0]}},\n  author = {${atomDetail.authors.join(" and ")}},\n  version = {${atomDetail.latest_version}},\n  year = {2026}\n}`;
  },

  getLeaderboard(_limit?: number): LeaderboardEntry[] {
    return leaderboard;
  },

  getComputePreserved(): ComputePreserved {
    return computePreserved;
  },

  getOriginatorImpact(_id: string): OriginatorImpact {
    return originatorImpact;
  },
};
