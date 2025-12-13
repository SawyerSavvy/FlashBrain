-- ============================================================
-- Skill Verification Migration
-- ============================================================
-- Adds 'github_verified' and 'github_partial' to evidence_strength enum
-- for GitHub-based skill verification during onboarding.
--
-- Author: FlashBrain Onboarding Agent
-- Date: 2025-01-15
-- ============================================================

-- ============================================================
-- 1. UPDATE EVIDENCE_STRENGTH CONSTRAINT
-- ============================================================
ALTER TABLE freelancer_skills
DROP CONSTRAINT IF EXISTS check_evidence;

ALTER TABLE freelancer_skills
ADD CONSTRAINT check_evidence CHECK (
    evidence_strength::text = ANY (ARRAY[
        'self_reported',
        'endorsed',
        'tested',
        'verified',
        'certified',
        'demonstrated',
        'github_verified',
        'github_partial'
    ]::text[])
);

COMMENT ON COLUMN freelancer_skills.evidence_strength IS
'Evidence strength: github_verified = strong GitHub evidence, github_partial = some evidence found';


-- ============================================================
-- 2. ADD GITHUB VERIFICATION TRACKING COLUMNS
-- ============================================================
ALTER TABLE freelancer_skills
ADD COLUMN IF NOT EXISTS github_verified_at TIMESTAMPTZ DEFAULT NULL;

ALTER TABLE freelancer_skills
ADD COLUMN IF NOT EXISTS github_repos_checked INTEGER DEFAULT 0;

COMMENT ON COLUMN freelancer_skills.github_verified_at IS
'Timestamp when GitHub verification was last run for this skill';

COMMENT ON COLUMN freelancer_skills.github_repos_checked IS
'Number of GitHub repositories checked during verification';


-- ============================================================
-- 3. INDEXES FOR VERIFICATION QUERIES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_freelancer_skills_evidence
ON freelancer_skills(evidence_strength, confidence_score DESC)
WHERE evidence_strength IN ('github_verified', 'github_partial');

CREATE INDEX IF NOT EXISTS idx_freelancer_skills_needs_verification
ON freelancer_skills(freelancer_id, skill_id)
WHERE evidence_strength = 'self_reported'
  AND github_verified_at IS NULL;


-- ============================================================
-- 4. HELPER FUNCTION: Get Skills Needing Verification
-- ============================================================
CREATE OR REPLACE FUNCTION get_skills_needing_github_verification(
    p_freelancer_id UUID,
    p_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    skill_id UUID,
    canonical_name TEXT,
    proficiency_level VARCHAR(20),
    evidence_strength VARCHAR(20),
    last_verified TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        fs.skill_id,
        s.canonical_name,
        fs.proficiency_level,
        fs.evidence_strength,
        fs.github_verified_at as last_verified
    FROM freelancer_skills fs
    JOIN skills s ON s.id = fs.skill_id
    WHERE fs.freelancer_id = p_freelancer_id
      AND (
          fs.evidence_strength = 'self_reported'
          OR fs.github_verified_at IS NULL
          OR fs.github_verified_at < NOW() - INTERVAL '30 days'
      )
    ORDER BY fs.confidence_score ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;


-- ============================================================
-- 5. HELPER FUNCTION: Update Skill with GitHub Evidence
-- ============================================================
CREATE OR REPLACE FUNCTION update_skill_github_evidence(
    p_freelancer_id UUID,
    p_skill_id UUID,
    p_confidence NUMERIC(3,2),
    p_evidence JSONB,
    p_repos_checked INTEGER
)
RETURNS VOID AS $$
DECLARE
    v_evidence_strength VARCHAR(20);
BEGIN
    IF p_confidence >= 0.7 THEN
        v_evidence_strength := 'github_verified';
    ELSIF p_confidence >= 0.4 THEN
        v_evidence_strength := 'github_partial';
    ELSE
        v_evidence_strength := 'self_reported';
    END IF;

    UPDATE freelancer_skills
    SET
        evidence_strength = v_evidence_strength,
        confidence_score = GREATEST(confidence_score, p_confidence),
        github_verified_at = NOW(),
        github_repos_checked = p_repos_checked,
        metadata = jsonb_set(
            COALESCE(metadata, '{}'::jsonb),
            '{github_evidence}',
            p_evidence
        ),
        updated_at = NOW()
    WHERE freelancer_id = p_freelancer_id
      AND skill_id = p_skill_id;
END;
$$ LANGUAGE plpgsql;
