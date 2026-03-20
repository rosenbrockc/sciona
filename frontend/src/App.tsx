import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import BountyList from "./pages/BountyList";
import BountyDetail from "./pages/BountyDetail";
import AtomList from "./pages/AtomList";
import AtomDetail from "./pages/AtomDetail";
import Leaderboard from "./pages/Leaderboard";
import ESGDashboard from "./pages/ESGDashboard";
import OriginatorProfile from "./pages/OriginatorProfile";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="bounties" element={<BountyList />} />
        <Route path="bounties/:id" element={<BountyDetail />} />
        <Route path="atoms" element={<AtomList />} />
        <Route path="atoms/:fqdn" element={<AtomDetail />} />
        <Route path="leaderboard" element={<Leaderboard />} />
        <Route path="esg" element={<ESGDashboard />} />
        <Route path="originator/:id" element={<OriginatorProfile />} />
      </Route>
    </Routes>
  );
}
