#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
infra_root="$(cd "${repo_root}/../sciona-infra" 2>/dev/null && pwd || true)"

if [ -n "${SCIONA_SUPABASE_PROJECT_ROOT:-}" ]; then
  supabase_project_root="$(cd "${SCIONA_SUPABASE_PROJECT_ROOT}" && pwd)"
elif [ -n "${infra_root}" ] && [ -f "${infra_root}/supabase/config.toml" ]; then
  supabase_project_root="${infra_root}"
else
  supabase_project_root="${repo_root}"
fi

seed_sql="${supabase_project_root}/supabase/seed.sql"
validation_sql="${supabase_project_root}/supabase/sql/phase0_validation.sql"

echo "=== Phase 0 validation ==="
echo "Using Supabase project root: ${supabase_project_root}"

: "${SUPABASE_URL:?SUPABASE_URL is not set}"
: "${SUPABASE_ANON_KEY:?SUPABASE_ANON_KEY is not set}"
: "${SUPABASE_SERVICE_ROLE_KEY:?SUPABASE_SERVICE_ROLE_KEY is not set}"
: "${SUPABASE_DATABASE_URL:?SUPABASE_DATABASE_URL is not set}"

echo "[1/5] Checking Supabase CLI..."
supabase --version

echo "[2/5] Checking REST API connectivity..."
http_status="$(curl -s -o /dev/null -w "%{http_code}" \
  -H "apikey: ${SUPABASE_ANON_KEY}" \
  -H "Authorization: Bearer ${SUPABASE_ANON_KEY}" \
  "${SUPABASE_URL}/rest/v1/")"
if [ "${http_status}" != "200" ]; then
  echo "REST API connectivity failed: ${http_status}" >&2
  exit 1
fi

echo "[3/5] Applying seed data..."
psql "${SUPABASE_DATABASE_URL}" -v ON_ERROR_STOP=1 -f "${seed_sql}" >/dev/null

echo "[4/5] Running validation SQL..."
psql "${SUPABASE_DATABASE_URL}" -v ON_ERROR_STOP=1 -f "${validation_sql}"

echo "[5/5] Listing migration history..."
(
  cd "${supabase_project_root}"
  supabase migration list --local
)

echo "=== Phase 0 validation complete ==="
