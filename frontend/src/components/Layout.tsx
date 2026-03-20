import { NavLink, Outlet } from "react-router-dom";

const links = [
  { to: "/", label: "Home" },
  { to: "/bounties", label: "Bounties" },
  { to: "/atoms", label: "Atoms" },
  { to: "/leaderboard", label: "Leaderboard" },
  { to: "/esg", label: "ESG Dashboard" },
];

export default function Layout() {
  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <nav className="w-60 shrink-0 bg-panel border-r border-border flex flex-col">
        <div className="p-5 border-b border-border">
          <h1 className="text-accent font-bold text-lg tracking-tight">
            Algorithmic Commons
          </h1>
        </div>
        <ul className="flex-1 py-3">
          {links.map((l) => (
            <li key={l.to}>
              <NavLink
                to={l.to}
                end={l.to === "/"}
                className={({ isActive }) =>
                  `block px-5 py-2.5 text-sm transition-colors ${
                    isActive
                      ? "text-accent bg-panel-soft border-r-2 border-accent"
                      : "text-muted hover:text-gray-200 hover:bg-panel-soft"
                  }`
                }
              >
                {l.label}
              </NavLink>
            </li>
          ))}
        </ul>
        <div className="p-4 border-t border-border text-xs text-muted">
          v0.1.0
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 p-8 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
