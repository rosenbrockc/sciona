-- Phase C: Verification & Bounty Settlement schema additions.
-- Extends the Phase B bounty schema with verification tracking,
-- settlement, and execution receipt storage.

-- Verification budget and slot tracking
CREATE TABLE IF NOT EXISTS verification_budgets (
    bounty_id       UUID PRIMARY KEY REFERENCES bounties(bounty_id),
    tier            TEXT NOT NULL CHECK (tier IN ('standard', 'heavy', 'gpu')),
    total_slots     INT NOT NULL,
    used_slots      INT NOT NULL DEFAULT 0,
    cost_per_extra  NUMERIC(10,2) NOT NULL,
    overhead_deposit NUMERIC(10,2) NOT NULL DEFAULT 0,
    overhead_used   NUMERIC(10,2) NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Individual verification runs
CREATE TABLE IF NOT EXISTS verification_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bounty_id       UUID NOT NULL REFERENCES bounties(bounty_id),
    submission_id   UUID NOT NULL REFERENCES submissions(submission_id),
    split_type      TEXT NOT NULL CHECK (split_type IN ('public', 'blind')),
    status          TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    metric_values   JSONB,
    output_hash     TEXT,
    execution_time_s FLOAT,
    peak_memory_bytes BIGINT,
    is_deterministic BOOLEAN,
    sandbox_job_id  TEXT,
    slot_consumed   BOOLEAN NOT NULL DEFAULT false,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_verification_runs_bounty
    ON verification_runs (bounty_id);
CREATE INDEX IF NOT EXISTS idx_verification_runs_submission
    ON verification_runs (submission_id);

-- Best-to-date tracking per bounty
CREATE TABLE IF NOT EXISTS bounty_best_scores (
    bounty_id       UUID NOT NULL REFERENCES bounties(bounty_id),
    metric_name     TEXT NOT NULL,
    best_value      FLOAT NOT NULL,
    best_submission_id UUID REFERENCES submissions(submission_id),
    is_baseline     BOOLEAN NOT NULL DEFAULT false,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (bounty_id, metric_name)
);

-- Principal target adjustments (between verifications)
CREATE TABLE IF NOT EXISTS principal_targets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bounty_id       UUID NOT NULL REFERENCES bounties(bounty_id),
    metric_name     TEXT NOT NULL,
    target_value    FLOAT NOT NULL,
    set_by          UUID NOT NULL REFERENCES users(user_id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_principal_targets_bounty
    ON principal_targets (bounty_id);

-- Execution receipts (from Architects)
CREATE TABLE IF NOT EXISTS execution_receipts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id   UUID NOT NULL REFERENCES submissions(submission_id),
    bounty_id       UUID NOT NULL,
    cdg_hash        TEXT NOT NULL,
    atom_versions   JSONB NOT NULL,
    split_hash      TEXT NOT NULL,
    output_hash     TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    metric_value    FLOAT NOT NULL,
    ageom_version   TEXT NOT NULL,
    ssh_signature   TEXT NOT NULL,
    ssh_public_key  TEXT NOT NULL,
    verified        BOOLEAN DEFAULT false,
    receipt_timestamp TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_execution_receipts_submission
    ON execution_receipts (submission_id);
CREATE INDEX IF NOT EXISTS idx_execution_receipts_bounty
    ON execution_receipts (bounty_id);

-- Dataset split assignments
CREATE TABLE IF NOT EXISTS dataset_splits (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bounty_id       UUID NOT NULL REFERENCES bounties(bounty_id),
    unit_key        TEXT NOT NULL,
    partition       TEXT NOT NULL CHECK (partition IN ('public', 'blind')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (bounty_id, unit_key)
);

CREATE INDEX IF NOT EXISTS idx_dataset_splits_bounty
    ON dataset_splits (bounty_id);

-- Settlement payouts
CREATE TABLE IF NOT EXISTS settlement_payouts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bounty_id       UUID NOT NULL REFERENCES bounties(bounty_id),
    recipient_id    TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('platform', 'architect', 'originator')),
    amount          NUMERIC(12,2) NOT NULL,
    stripe_transfer_id TEXT,
    atom_fqdn       TEXT,
    cdg_hash        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_settlement_payouts_bounty
    ON settlement_payouts (bounty_id);
