import { mockApi } from "./mock";
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

const USE_MOCK = import.meta.env.VITE_USE_MOCK !== "false";
const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

export const api = {
  async getBounties(params?: {
    status?: string;
    domain_tag?: string;
    limit?: number;
    offset?: number;
  }): Promise<PaginatedResponse<BountySummaryResponse>> {
    if (USE_MOCK) return mockApi.getBounties(params);
    const qs = new URLSearchParams();
    if (params?.status) qs.set("status", params.status);
    if (params?.domain_tag) qs.set("domain_tag", params.domain_tag);
    if (params?.limit) qs.set("limit", String(params.limit));
    if (params?.offset) qs.set("offset", String(params.offset));
    return get(`/bounties?${qs}`);
  },

  async getBounty(id: string): Promise<BountyResponse> {
    if (USE_MOCK) return mockApi.getBounty(id)!;
    return get(`/bounties/${id}`);
  },

  async getBountyLeaderboard(id: string): Promise<SubmissionLeaderboardEntry[]> {
    if (USE_MOCK) return mockApi.getBountyLeaderboard(id);
    return get(`/bounties/${id}/leaderboard`);
  },

  async getBountySettlement(id: string): Promise<SettlementInfo> {
    if (USE_MOCK) return mockApi.getBountySettlement(id);
    return get(`/bounties/${id}/settlement`);
  },

  async getAtoms(params?: {
    search?: string;
    domain_tag?: string;
    limit?: number;
    offset?: number;
  }): Promise<PaginatedResponse<AtomSummaryResponse>> {
    if (USE_MOCK) return mockApi.getAtoms(params);
    const qs = new URLSearchParams();
    if (params?.search) qs.set("q", params.search);
    if (params?.domain_tag) qs.set("domain_tag", params.domain_tag);
    if (params?.limit) qs.set("limit", String(params.limit));
    if (params?.offset) qs.set("offset", String(params.offset));
    return get(`/atoms?${qs}`);
  },

  async getAtom(fqdn: string): Promise<AtomDetailResponse> {
    if (USE_MOCK) return mockApi.getAtom(fqdn);
    return get(`/atoms/${fqdn}`);
  },

  async getAtomVersions(fqdn: string): Promise<AtomVersionResponse[]> {
    if (USE_MOCK) return mockApi.getAtomVersions(fqdn);
    return get(`/atoms/${fqdn}/versions`);
  },

  async getAtomBenchmarks(fqdn: string): Promise<BenchmarkRecord[]> {
    if (USE_MOCK) return mockApi.getAtomBenchmarks(fqdn);
    return get(`/dashboard/atom/${fqdn}/benchmarks`);
  },

  async getAtomBibtex(fqdn: string): Promise<string> {
    if (USE_MOCK) return mockApi.getAtomBibtex(fqdn);
    const res = await fetch(`${API_BASE}/dashboard/atom/${fqdn}/bibtex`);
    return res.text();
  },

  async getLeaderboard(limit?: number): Promise<LeaderboardEntry[]> {
    if (USE_MOCK) return mockApi.getLeaderboard(limit);
    const qs = limit ? `?limit=${limit}` : "";
    return get(`/dashboard/leaderboard${qs}`);
  },

  async getComputePreserved(): Promise<ComputePreserved> {
    if (USE_MOCK) return mockApi.getComputePreserved();
    return get("/dashboard/compute-preserved");
  },

  async getOriginatorImpact(id: string): Promise<OriginatorImpact> {
    if (USE_MOCK) return mockApi.getOriginatorImpact(id);
    return get(`/dashboard/originator/${id}/impact`);
  },
};
