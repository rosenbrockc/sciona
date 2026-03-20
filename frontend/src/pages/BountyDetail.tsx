import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api/client";
import type { BountyResponse, SubmissionLeaderboardEntry, SettlementInfo } from "../api/types";
import StatusBadge from "../components/StatusBadge";
import StatCard from "../components/StatCard";

export default function BountyDetail() {
  const { id } = useParams<{ id: string }>();
  const [bounty, setBounty] = useState<BountyResponse | null>(null);
  const [leaderboard, setLeaderboard] = useState<SubmissionLeaderboardEntry[]>([]);
  const [settlement, setSettlement] = useState<SettlementInfo | null>(null);

  useEffect(() => {
    if (!id) return;
    api.getBounty(id).then(setBounty);
    api.getBountyLeaderboard(id).then(setLeaderboard);
    api.getBountySettlement(id).then(setSettlement).catch(() => {});
  }, [id]);

  if (!bounty) return <p className="text-muted">Loading...</p>;

  const budgetPct = bounty.verification_budget_total
    ? Math.round((bounty.verification_budget_used / bounty.verification_budget_total) * 100)
    : 0;

  return (
    <div className="space-y-8">
      <div>
        <div className="flex items-center gap-3 mb-2">
          <h2 className="text-xl font-bold">{bounty.title}</h2>
          <StatusBadge status={bounty.status} />
        </div>
        <p className="text-muted text-sm font-mono">{bounty.bounty_id}</p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Escrow" value={`$${bounty.escrow_amount.toLocaleString()}`} />
        <StatCard label="Tier" value={bounty.tier} />
        <StatCard label="Deadline" value={bounty.deadline} />
        <StatCard label="Created" value={bounty.created_at} />
      </div>

      {/* Tags */}
      <div className="flex gap-2">
        {bounty.domain_tags.map((t) => (
          <span key={t} className="px-3 py-1 bg-panel-soft rounded-full text-xs text-muted border border-border">
            {t}
          </span>
        ))}
      </div>

      {/* Verification budget */}
      <div className="bg-panel border border-border rounded-lg p-5">
        <h3 className="text-sm font-semibold text-muted uppercase tracking-wide mb-3">Verification Budget</h3>
        <div className="flex items-center gap-4">
          <div className="flex-1 h-2 bg-panel-soft rounded-full overflow-hidden">
            <div
              className="h-full bg-accent rounded-full transition-all"
              style={{ width: `${budgetPct}%` }}
            />
          </div>
          <span className="text-sm text-muted font-mono">
            {bounty.verification_budget_used}/{bounty.verification_budget_total}
          </span>
        </div>
      </div>

      {/* Submission leaderboard */}
      {leaderboard.length > 0 && (
        <div className="bg-panel border border-border rounded-lg p-5">
          <h3 className="text-sm font-semibold text-muted uppercase tracking-wide mb-4">Submission Leaderboard</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-muted border-b border-border">
                <th className="pb-2 pr-4">#</th>
                <th className="pb-2 pr-4">Architect</th>
                <th className="pb-2 pr-4">Metrics</th>
                <th className="pb-2">Verified</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.map((s) => (
                <tr key={s.submission_id} className="border-b border-border/50">
                  <td className="py-2 pr-4 text-muted">{s.rank}</td>
                  <td className="py-2 pr-4 text-accent">{s.architect_id}</td>
                  <td className="py-2 pr-4 font-mono text-xs">
                    {Object.entries(s.metric_values)
                      .map(([k, v]) => `${k}: ${v}`)
                      .join(", ")}
                  </td>
                  <td className="py-2 text-muted">{s.verified_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Settlement */}
      {settlement && settlement.status === "settled" && (
        <div className="bg-panel border border-border rounded-lg p-5">
          <h3 className="text-sm font-semibold text-muted uppercase tracking-wide mb-4">Settlement Breakdown</h3>
          <p className="text-sm text-muted mb-3">
            Winning submission: <span className="font-mono text-accent">{settlement.winning_submission_id}</span>
          </p>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-muted border-b border-border">
                <th className="pb-2 pr-4">Recipient</th>
                <th className="pb-2 pr-4">Role</th>
                <th className="pb-2">Amount</th>
              </tr>
            </thead>
            <tbody>
              {settlement.payouts.map((p) => (
                <tr key={p.recipient_id} className="border-b border-border/50">
                  <td className="py-2 pr-4 font-mono">{p.recipient_id}</td>
                  <td className="py-2 pr-4">{p.role}</td>
                  <td className="py-2 font-mono">${p.amount.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
