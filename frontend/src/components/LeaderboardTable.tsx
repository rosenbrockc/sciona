import { Link } from "react-router-dom";
import type { LeaderboardEntry } from "../api/types";

export default function LeaderboardTable({ entries, compact }: { entries: LeaderboardEntry[]; compact?: boolean }) {
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-muted border-b border-border">
          <th className="pb-2 pr-4">#</th>
          <th className="pb-2 pr-4">Originator</th>
          <th className="pb-2 pr-4">h-index</th>
          {!compact && <th className="pb-2 pr-4">Bounties</th>}
          <th className="pb-2 pr-4">Total Value</th>
          {!compact && <th className="pb-2">Atoms</th>}
        </tr>
      </thead>
      <tbody>
        {entries.map((e) => (
          <tr key={e.rank} className="border-b border-border/50">
            <td className="py-2 pr-4 text-muted">{e.rank}</td>
            <td className="py-2 pr-4">
              <Link to={`/originator/${e.username}`} className="text-accent hover:underline">
                {e.username}
              </Link>
            </td>
            <td className="py-2 pr-4 font-mono">{e.h_index}</td>
            {!compact && <td className="py-2 pr-4">{e.bounty_count}</td>}
            <td className="py-2 pr-4 font-mono">${e.total_value.toLocaleString()}</td>
            {!compact && <td className="py-2">{e.atom_count}</td>}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
