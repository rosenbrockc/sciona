import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { api } from "../api/client";
import type { OriginatorImpact } from "../api/types";
import StatCard from "../components/StatCard";

export default function OriginatorProfile() {
  const { id } = useParams<{ id: string }>();
  const [impact, setImpact] = useState<OriginatorImpact | null>(null);

  useEffect(() => {
    if (!id) return;
    api.getOriginatorImpact(id).then(setImpact);
  }, [id]);

  if (!impact) return <p className="text-muted">Loading...</p>;

  return (
    <div className="space-y-8">
      <h2 className="text-xl font-bold">{impact.username}</h2>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="h-index" value={impact.h_index} />
        <StatCard label="Bounties" value={impact.bounty_count} />
        <StatCard label="Total Value" value={`$${impact.total_value.toLocaleString()}`} />
        <StatCard label="Atoms" value={impact.atom_count} />
      </div>

      <div className="bg-panel border border-border rounded-lg p-5">
        <h3 className="text-sm font-semibold text-muted uppercase tracking-wide mb-4">Published Atoms</h3>
        <div className="grid sm:grid-cols-2 gap-3">
          {impact.atoms.map((a) => (
            <Link
              key={a.fqdn}
              to={`/atoms/${a.fqdn}`}
              className="bg-panel-soft border border-border rounded p-3 hover:border-accent/50 transition-colors"
            >
              <p className="font-mono text-accent text-sm">{a.fqdn}</p>
              <p className="text-xs text-muted mt-1 line-clamp-1">{a.description}</p>
              <span className="text-xs font-mono text-muted mt-1 inline-block">v{a.latest_version}</span>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
