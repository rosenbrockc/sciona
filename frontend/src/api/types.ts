// Mirrors Pydantic models from ageom/api/models.py and ageom/ecosystem/models.py

export interface BountyResponse {
  bounty_id: string;
  title: string;
  escrow_amount: number;
  deadline: string;
  tier: string;
  status: string;
  domain_tags: string[];
  verification_budget_used: number;
  verification_budget_total: number;
  created_at: string;
}

export interface BountySummaryResponse {
  bounty_id: string;
  title: string;
  escrow_amount: number;
  status: string;
  domain_tags: string[];
  deadline: string;
}

export interface AtomDetailResponse {
  atom_id: string;
  fqdn: string;
  description: string;
  domain_tags: string[];
  latest_version: string;
  created_at: string;
  authors: string[];
}

export interface AtomSummaryResponse {
  fqdn: string;
  description: string;
  domain_tags: string[];
  latest_version: string;
}

export interface AtomVersionResponse {
  version_id: string;
  version: string;
  fingerprint: string;
  is_latest: boolean;
  published_at: string;
}

export interface LeaderboardEntry {
  rank: number;
  username: string;
  h_index: number;
  bounty_count: number;
  total_value: number;
  atom_count: number;
}

export interface OriginatorImpact {
  originator_id: string;
  username: string;
  h_index: number;
  bounty_count: number;
  total_value: number;
  atom_count: number;
  atoms: AtomSummaryResponse[];
}

export interface ComputePreserved {
  estimated_tokens_saved: number;
  estimated_cost_saved_usd: number;
  total_bounties_settled: number;
  total_escrow_distributed: number;
  cross_discipline_atoms: number;
}

export interface BenchmarkRecord {
  atom_fqdn: string;
  metric_name: string;
  metric_value: number;
  dataset_tag: string;
  recorded_at: string;
}

export interface SubmissionLeaderboardEntry {
  rank: number;
  submission_id: string;
  architect_id: string;
  metric_values: Record<string, number>;
  verified_at: string;
}

export interface SettlementInfo {
  bounty_id: string;
  status: string;
  winning_submission_id: string;
  payouts: PayoutRecipient[];
}

export interface PayoutRecipient {
  recipient_id: string;
  role: string;
  amount: number;
  stripe_account: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}
