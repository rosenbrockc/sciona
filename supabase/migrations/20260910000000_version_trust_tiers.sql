-- Association trust tiers are version-specific and independent of access tiers.
-- A Community label is not publication approval. Existing publication predicates
-- remain unchanged. No existing human review is inferred to be certification.
ALTER TABLE public.artifact_versions
    ADD COLUMN IF NOT EXISTS trust_tier smallint NOT NULL DEFAULT 3;
ALTER TABLE public.atom_versions
    ADD COLUMN IF NOT EXISTS trust_tier smallint NOT NULL DEFAULT 3;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='public.artifact_versions'::regclass AND conname='artifact_versions_trust_tier_check') THEN
        ALTER TABLE public.artifact_versions ADD CONSTRAINT artifact_versions_trust_tier_check CHECK (trust_tier IN (1,2,3));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='public.atom_versions'::regclass AND conname='atom_versions_trust_tier_check') THEN
        ALTER TABLE public.atom_versions ADD CONSTRAINT atom_versions_trust_tier_check CHECK (trust_tier IN (1,2,3));
    END IF;
END $$;
COMMENT ON COLUMN public.artifact_versions.trust_tier IS
    '3 Community; 2 Verified; 1 Certified. Version-bound assurance, not publication permission or visibility. New versions default to Community; promotion requires separate evidence.';
COMMENT ON COLUMN public.atom_versions.trust_tier IS
    '3 Community; 2 Verified; 1 Certified. Version-bound assurance, not publication permission or visibility. New versions default to Community; promotion requires separate evidence.';
