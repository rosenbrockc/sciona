-- Phase D: Ecosystem (Community Flywheel) schema additions.
-- Adds fuzzing results, benchmark suites, behavioral equivalence,
-- discipline repo tracking, and soft deprecation support.

-- Benchmark suites (curated test suites)
CREATE TABLE IF NOT EXISTS benchmark_suites (
    benchmark_id    TEXT PRIMARY KEY,
    domain_tags     TEXT[] NOT NULL DEFAULT '{}',
    description     TEXT,
    dataset_s3_key  TEXT NOT NULL DEFAULT '',
    metric_names    TEXT[] NOT NULL DEFAULT '{}',
    curation_source TEXT NOT NULL DEFAULT 'foundation'
                    CHECK (curation_source IN ('foundation', 'community', 'bounty_derived')),
    proposer_id     UUID REFERENCES users(user_id),
    vote_count      INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'active'
                    CHECK (status IN ('active', 'retired', 'proposed')),
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Benchmark votes
CREATE TABLE IF NOT EXISTS benchmark_votes (
    benchmark_id    TEXT NOT NULL REFERENCES benchmark_suites(benchmark_id),
    voter_id        UUID NOT NULL REFERENCES users(user_id),
    vote            TEXT NOT NULL CHECK (vote IN ('approve', 'reject')),
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (benchmark_id, voter_id)
);

-- Fuzz results
CREATE TABLE IF NOT EXISTS fuzz_results (
    fuzz_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    atom_fqdn       TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    strategy        TEXT NOT NULL
                    CHECK (strategy IN ('property_based', 'boundary_value',
                                        'param_smoothing', 'behavioral_equiv')),
    passed          BOOLEAN NOT NULL,
    failures        JSONB DEFAULT '[]',
    inputs_tested   INTEGER NOT NULL DEFAULT 0,
    runtime_ms      INTEGER,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fuzz_results_atom
    ON fuzz_results (atom_fqdn, content_hash);

-- Behavioral equivalence flags
CREATE TABLE IF NOT EXISTS behavioral_equivalence_flags (
    flag_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    atom_a_fqdn     TEXT NOT NULL,
    atom_a_hash     TEXT NOT NULL,
    atom_b_fqdn     TEXT NOT NULL,
    atom_b_hash     TEXT NOT NULL,
    match_ratio     FLOAT NOT NULL,
    sample_size     INTEGER NOT NULL,
    reviewed        BOOLEAN DEFAULT FALSE,
    reviewer_id     UUID REFERENCES users(user_id),
    disposition     TEXT CHECK (disposition IS NULL OR
                    disposition IN ('plagiarism', 'coincidence', 'common_algorithm')),
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Soft deprecation: add superseded_by to atoms
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'atoms' AND column_name = 'superseded_by'
    ) THEN
        ALTER TABLE atoms ADD COLUMN superseded_by TEXT;
    END IF;
END $$;

-- Discipline repository registration
CREATE TABLE IF NOT EXISTS discipline_repos (
    repo_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_url        TEXT UNIQUE NOT NULL,
    webhook_secret  TEXT NOT NULL DEFAULT '',
    domain_tags     TEXT[] NOT NULL DEFAULT '{}',
    maintainer_ids  UUID[] NOT NULL DEFAULT '{}',
    last_synced_commit TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'paused', 'removed')),
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- Dashboard views

CREATE OR REPLACE VIEW originator_impact AS
SELECT
    aa.user_id AS originator_id,
    u.github_login,
    COUNT(DISTINCT b.bounty_id) AS bounty_count,
    COALESCE(SUM(b.escrow_amount), 0) AS total_bounty_value,
    COUNT(DISTINCT aa.atom_id) AS atom_count
FROM atom_authors aa
JOIN users u ON u.user_id = aa.user_id
JOIN atoms a ON a.atom_id = aa.atom_id
LEFT JOIN submissions s ON s.atom_versions ? a.fqdn AND s.is_winner = true
LEFT JOIN bounties b ON b.bounty_id = s.bounty_id AND b.status = 'settled'
GROUP BY aa.user_id, u.github_login;

CREATE OR REPLACE VIEW compute_preserved AS
SELECT
    b.bounty_id,
    b.escrow_amount,
    COALESCE(jsonb_array_length(s.atom_versions), 0) AS cdg_node_count,
    COALESCE(jsonb_array_length(s.atom_versions), 0) * 2000 * 5 AS estimated_tokens_saved,
    COALESCE(jsonb_array_length(s.atom_versions), 0) * 2000 * 5 * 0.003 AS estimated_cost_saved
FROM bounties b
JOIN submissions s ON s.bounty_id = b.bounty_id AND s.is_winner = true
WHERE b.status = 'settled';
