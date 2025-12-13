-- ============================================================
-- GitHub Onboarding - Freelancer Profiles Database Schema
-- ============================================================
--
-- This migration creates 3 tables for storing GitHub profile analysis:
-- 1. freelancer_github_profiles - Main profile data with calculated scores
-- 2. freelancer_repositories - Repository metadata per developer
-- 3. freelancer_commits - Sample commits for deep LLM analysis
--
-- Author: FlashBrain Onboarding Agent
-- Date: 2025-01-06
-- ============================================================

-- ============================================================
-- 1. FREELANCER GITHUB PROFILES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS freelancer_github_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- GitHub Identity
    github_username TEXT NOT NULL UNIQUE,
    github_user_id TEXT NOT NULL UNIQUE,

    -- Profile Metadata
    name TEXT,
    bio TEXT,
    location TEXT,
    company TEXT,
    email TEXT,
    blog_url TEXT,
    twitter_username TEXT,
    avatar_url TEXT,

    -- GitHub Stats
    public_repos INTEGER DEFAULT 0,
    followers INTEGER DEFAULT 0,
    following INTEGER DEFAULT 0,
    total_stars INTEGER DEFAULT 0,

    -- Technology Stack (JSONB array)
    -- Example: [{"name": "Python", "count": 45, "percentage": 0.65}, ...]
    tech_stack JSONB DEFAULT '[]'::jsonb,

    -- Domain Expertise (JSONB object with percentages)
    -- Example: {"frontend": 0.3, "backend": 0.6, "ml": 0.1, "devops": 0.0}
    domain_expertise JSONB DEFAULT '{}'::jsonb,

    -- Calculated Skill Scores (0-100)
    commit_consistency_score FLOAT CHECK (commit_consistency_score IS NULL OR (commit_consistency_score BETWEEN 0 AND 100)),
    code_quality_score FLOAT CHECK (code_quality_score IS NULL OR (code_quality_score BETWEEN 0 AND 100)),
    impact_score FLOAT CHECK (impact_score IS NULL OR (impact_score BETWEEN 0 AND 100)),

    -- LLM-Generated Insights
    developer_story TEXT,  -- Narrative summary created by LLM
    top_skills TEXT[],     -- ["Python", "React", "Docker", "AWS"]

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_analyzed_at TIMESTAMPTZ,

    -- Multi-tenant support
    client_id UUID,
    organization_id UUID
);

-- Indexes for freelancer_github_profiles
CREATE INDEX IF NOT EXISTS idx_freelancer_github_profiles_github_username
    ON freelancer_github_profiles(github_username);

CREATE INDEX IF NOT EXISTS idx_freelancer_github_profiles_github_user_id
    ON freelancer_github_profiles(github_user_id);

CREATE INDEX IF NOT EXISTS idx_freelancer_github_profiles_client_id
    ON freelancer_github_profiles(client_id)
    WHERE client_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_freelancer_github_profiles_organization_id
    ON freelancer_github_profiles(organization_id)
    WHERE organization_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_freelancer_github_profiles_created_at
    ON freelancer_github_profiles(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_freelancer_github_profiles_last_analyzed_at
    ON freelancer_github_profiles(last_analyzed_at DESC)
    WHERE last_analyzed_at IS NOT NULL;

-- GIN indexes for JSONB queries (allows searching by tech stack or domain)
CREATE INDEX IF NOT EXISTS idx_freelancer_github_profiles_tech_stack
    ON freelancer_github_profiles USING GIN (tech_stack);

CREATE INDEX IF NOT EXISTS idx_freelancer_github_profiles_domain_expertise
    ON freelancer_github_profiles USING GIN (domain_expertise);

-- GIN index for array queries (allows searching by skills)
CREATE INDEX IF NOT EXISTS idx_freelancer_github_profiles_top_skills
    ON freelancer_github_profiles USING GIN (top_skills);

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_freelancer_github_profiles_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER freelancer_github_profiles_updated_at_trigger
    BEFORE UPDATE ON freelancer_github_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_freelancer_github_profiles_updated_at();


-- ============================================================
-- 2. FREELANCER REPOSITORIES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS freelancer_repositories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID NOT NULL REFERENCES freelancer_github_profiles(id) ON DELETE CASCADE,

    -- Repository Metadata
    repo_name TEXT NOT NULL,
    repo_full_name TEXT NOT NULL,  -- e.g., "octocat/Hello-World"
    repo_url TEXT NOT NULL,
    description TEXT,

    -- Repository Stats
    stars INTEGER DEFAULT 0,
    forks INTEGER DEFAULT 0,
    watchers INTEGER DEFAULT 0,
    open_issues INTEGER DEFAULT 0,
    is_fork BOOLEAN DEFAULT FALSE,

    -- Language Breakdown (JSONB object)
    -- Example: {"Python": 65.5, "JavaScript": 30.2, "HTML": 4.3}
    primary_language TEXT,
    languages JSONB DEFAULT '{}'::jsonb,

    -- Activity Metrics
    total_commits INTEGER DEFAULT 0,
    user_commits INTEGER DEFAULT 0,  -- Commits by this specific user
    last_commit_date TIMESTAMPTZ,

    -- GitHub Timestamps
    created_at_github TIMESTAMPTZ,
    updated_at_github TIMESTAMPTZ,
    pushed_at_github TIMESTAMPTZ,

    -- Technologies Detected (arrays for easy searching)
    technologies TEXT[],  -- ["React", "Node.js", "MongoDB"]
    frameworks TEXT[],    -- ["Express", "Next.js"]

    -- Domain Classification
    domain_tags TEXT[],  -- ["frontend", "api", "data-science"]

    -- Our Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Ensure unique repo per profile
    CONSTRAINT unique_profile_repo UNIQUE(profile_id, repo_full_name)
);

-- Indexes for freelancer_repositories
CREATE INDEX IF NOT EXISTS idx_freelancer_repositories_profile_id
    ON freelancer_repositories(profile_id);

CREATE INDEX IF NOT EXISTS idx_freelancer_repositories_repo_name
    ON freelancer_repositories(repo_name);

CREATE INDEX IF NOT EXISTS idx_freelancer_repositories_repo_full_name
    ON freelancer_repositories(repo_full_name);

CREATE INDEX IF NOT EXISTS idx_freelancer_repositories_primary_language
    ON freelancer_repositories(primary_language)
    WHERE primary_language IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_freelancer_repositories_stars
    ON freelancer_repositories(stars DESC);

CREATE INDEX IF NOT EXISTS idx_freelancer_repositories_user_commits
    ON freelancer_repositories(user_commits DESC);

CREATE INDEX IF NOT EXISTS idx_freelancer_repositories_last_commit_date
    ON freelancer_repositories(last_commit_date DESC)
    WHERE last_commit_date IS NOT NULL;

-- GIN indexes for array searches
CREATE INDEX IF NOT EXISTS idx_freelancer_repositories_technologies
    ON freelancer_repositories USING GIN (technologies);

CREATE INDEX IF NOT EXISTS idx_freelancer_repositories_frameworks
    ON freelancer_repositories USING GIN (frameworks);

CREATE INDEX IF NOT EXISTS idx_freelancer_repositories_domain_tags
    ON freelancer_repositories USING GIN (domain_tags);

-- GIN index for JSONB languages
CREATE INDEX IF NOT EXISTS idx_freelancer_repositories_languages
    ON freelancer_repositories USING GIN (languages);


-- ============================================================
-- 3. FREELANCER COMMITS TABLE (Sample Commits for Deep Analysis)
-- ============================================================
CREATE TABLE IF NOT EXISTS freelancer_commits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID NOT NULL REFERENCES freelancer_github_profiles(id) ON DELETE CASCADE,
    repository_id UUID NOT NULL REFERENCES freelancer_repositories(id) ON DELETE CASCADE,

    -- Commit Metadata
    commit_sha TEXT NOT NULL,
    commit_message TEXT,
    commit_date TIMESTAMPTZ,
    author_name TEXT,
    author_email TEXT,

    -- Commit Stats
    files_changed INTEGER DEFAULT 0,
    additions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,

    -- File-Level Details (JSONB array)
    -- Example: [{"filename": "src/app.py", "additions": 45, "deletions": 12, "status": "modified"}]
    changed_files JSONB DEFAULT '[]'::jsonb,

    -- Extracted File Metadata
    file_extensions TEXT[],  -- [".py", ".js", ".tsx"]
    directories TEXT[],      -- ["src/components", "tests"]

    -- LLM Analysis Results
    commit_type TEXT CHECK (commit_type IS NULL OR commit_type IN (
        'feature', 'bugfix', 'refactor', 'docs', 'test', 'chore', 'performance', 'style', 'unknown'
    )),
    complexity_score FLOAT CHECK (complexity_score IS NULL OR (complexity_score BETWEEN 0 AND 100)),
    impact_score FLOAT CHECK (impact_score IS NULL OR (impact_score BETWEEN 0 AND 100)),
    llm_summary TEXT,  -- LLM-generated description of what changed
    technologies_used TEXT[],  -- Technologies detected in this specific commit

    -- Sampling Metadata
    is_sampled BOOLEAN DEFAULT FALSE,  -- TRUE if this commit was selected for deep LLM analysis
    sample_reason TEXT,  -- Why this commit was sampled (e.g., "high impact", "recent", "diverse files")

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Ensure unique commit per repository
    CONSTRAINT unique_repo_commit UNIQUE(repository_id, commit_sha)
);

-- Indexes for freelancer_commits
CREATE INDEX IF NOT EXISTS idx_freelancer_commits_profile_id
    ON freelancer_commits(profile_id);

CREATE INDEX IF NOT EXISTS idx_freelancer_commits_repository_id
    ON freelancer_commits(repository_id);

CREATE INDEX IF NOT EXISTS idx_freelancer_commits_commit_sha
    ON freelancer_commits(commit_sha);

CREATE INDEX IF NOT EXISTS idx_freelancer_commits_commit_date
    ON freelancer_commits(commit_date DESC)
    WHERE commit_date IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_freelancer_commits_commit_type
    ON freelancer_commits(commit_type)
    WHERE commit_type IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_freelancer_commits_is_sampled
    ON freelancer_commits(is_sampled)
    WHERE is_sampled = TRUE;

-- GIN indexes for arrays
CREATE INDEX IF NOT EXISTS idx_freelancer_commits_file_extensions
    ON freelancer_commits USING GIN (file_extensions);

CREATE INDEX IF NOT EXISTS idx_freelancer_commits_directories
    ON freelancer_commits USING GIN (directories);

CREATE INDEX IF NOT EXISTS idx_freelancer_commits_technologies_used
    ON freelancer_commits USING GIN (technologies_used);

-- GIN index for JSONB changed_files
CREATE INDEX IF NOT EXISTS idx_freelancer_commits_changed_files
    ON freelancer_commits USING GIN (changed_files);


-- ============================================================
-- 4. HELPER FUNCTIONS
-- ============================================================

-- Function to get comprehensive profile summary
CREATE OR REPLACE FUNCTION get_freelancer_profile_summary(profile_uuid UUID)
RETURNS TABLE (
    profile_id UUID,
    github_username TEXT,
    name TEXT,
    bio TEXT,
    total_repos BIGINT,
    total_commits BIGINT,
    total_stars BIGINT,
    tech_stack JSONB,
    domain_expertise JSONB,
    top_languages JSONB,
    commit_consistency_score FLOAT,
    code_quality_score FLOAT,
    impact_score FLOAT,
    top_repositories JSONB,
    developer_story TEXT,
    last_analyzed_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        fp.id,
        fp.github_username,
        fp.name,
        fp.bio,
        COUNT(DISTINCT fr.id) as total_repos,
        COALESCE(SUM(fr.user_commits), 0) as total_commits,
        fp.total_stars,
        fp.tech_stack,
        fp.domain_expertise,
        -- Top 5 languages by repository count
        (
            SELECT jsonb_object_agg(primary_language, lang_count)
            FROM (
                SELECT primary_language, COUNT(*) as lang_count
                FROM freelancer_repositories
                WHERE profile_id = profile_uuid
                  AND primary_language IS NOT NULL
                GROUP BY primary_language
                ORDER BY lang_count DESC
                LIMIT 5
            ) langs
        ) as top_languages,
        fp.commit_consistency_score,
        fp.code_quality_score,
        fp.impact_score,
        -- Top 5 repositories by stars
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'name', repo_name,
                    'stars', stars,
                    'language', primary_language,
                    'commits', user_commits
                )
            )
            FROM (
                SELECT repo_name, stars, primary_language, user_commits
                FROM freelancer_repositories
                WHERE profile_id = profile_uuid
                ORDER BY stars DESC
                LIMIT 5
            ) top_repos
        ) as top_repositories,
        fp.developer_story,
        fp.last_analyzed_at
    FROM freelancer_github_profiles fp
    LEFT JOIN freelancer_repositories fr ON fr.profile_id = fp.id
    WHERE fp.id = profile_uuid
    GROUP BY fp.id, fp.github_username, fp.name, fp.bio, fp.total_stars,
             fp.tech_stack, fp.domain_expertise, fp.commit_consistency_score,
             fp.code_quality_score, fp.impact_score, fp.developer_story,
             fp.last_analyzed_at;
END;
$$ LANGUAGE plpgsql;


-- Function to get commit statistics for analysis
CREATE OR REPLACE FUNCTION get_commit_statistics(profile_uuid UUID)
RETURNS TABLE (
    total_commits BIGINT,
    sampled_commits BIGINT,
    avg_files_per_commit FLOAT,
    avg_additions FLOAT,
    avg_deletions FLOAT,
    commit_types JSONB,
    top_file_extensions JSONB,
    avg_complexity_score FLOAT,
    avg_impact_score FLOAT,
    first_commit_date TIMESTAMPTZ,
    last_commit_date TIMESTAMPTZ,
    days_active FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) as total_commits,
        COUNT(*) FILTER (WHERE is_sampled = TRUE) as sampled_commits,
        AVG(files_changed) as avg_files_per_commit,
        AVG(additions) as avg_additions,
        AVG(deletions) as avg_deletions,
        -- Commit type distribution
        (
            SELECT jsonb_object_agg(commit_type, type_count)
            FROM (
                SELECT
                    COALESCE(commit_type, 'unknown') as commit_type,
                    COUNT(*) as type_count
                FROM freelancer_commits
                WHERE profile_id = profile_uuid
                GROUP BY commit_type
            ) types
        ) as commit_types,
        -- Top 10 file extensions
        (
            SELECT jsonb_object_agg(ext, ext_count)
            FROM (
                SELECT
                    UNNEST(file_extensions) as ext,
                    COUNT(*) as ext_count
                FROM freelancer_commits
                WHERE profile_id = profile_uuid
                GROUP BY ext
                ORDER BY ext_count DESC
                LIMIT 10
            ) exts
        ) as top_file_extensions,
        AVG(complexity_score) FILTER (WHERE complexity_score IS NOT NULL) as avg_complexity_score,
        AVG(impact_score) FILTER (WHERE impact_score IS NOT NULL) as avg_impact_score,
        MIN(commit_date) as first_commit_date,
        MAX(commit_date) as last_commit_date,
        EXTRACT(EPOCH FROM (MAX(commit_date) - MIN(commit_date))) / 86400.0 as days_active
    FROM freelancer_commits
    WHERE profile_id = profile_uuid;
END;
$$ LANGUAGE plpgsql;


-- Function to search profiles by skills
CREATE OR REPLACE FUNCTION search_profiles_by_skills(
    skill_keywords TEXT[],
    min_consistency_score FLOAT DEFAULT 0,
    min_impact_score FLOAT DEFAULT 0,
    result_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    profile_id UUID,
    github_username TEXT,
    name TEXT,
    matching_skills TEXT[],
    tech_stack JSONB,
    commit_consistency_score FLOAT,
    impact_score FLOAT,
    total_repos BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        fp.id,
        fp.github_username,
        fp.name,
        fp.top_skills,
        fp.tech_stack,
        fp.commit_consistency_score,
        fp.impact_score,
        COUNT(DISTINCT fr.id) as total_repos
    FROM freelancer_github_profiles fp
    LEFT JOIN freelancer_repositories fr ON fr.profile_id = fp.id
    WHERE
        -- Match any skill in top_skills array
        fp.top_skills && skill_keywords
        -- Score thresholds
        AND (fp.commit_consistency_score IS NULL OR fp.commit_consistency_score >= min_consistency_score)
        AND (fp.impact_score IS NULL OR fp.impact_score >= min_impact_score)
    GROUP BY fp.id, fp.github_username, fp.name, fp.top_skills, fp.tech_stack,
             fp.commit_consistency_score, fp.impact_score
    ORDER BY
        -- Prioritize by number of matching skills
        cardinality(
            ARRAY(SELECT UNNEST(fp.top_skills) INTERSECT SELECT UNNEST(skill_keywords))
        ) DESC,
        fp.impact_score DESC NULLS LAST
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;


-- ============================================================
-- 5. COMMENTS & DOCUMENTATION
-- ============================================================

COMMENT ON TABLE freelancer_github_profiles IS 'Main table for freelancer GitHub profiles with calculated scores and LLM-generated insights';
COMMENT ON TABLE freelancer_repositories IS 'Repository metadata for each freelancer with language breakdown and activity metrics';
COMMENT ON TABLE freelancer_commits IS 'Sample commits selected for deep LLM analysis with classification and scoring';

COMMENT ON COLUMN freelancer_github_profiles.tech_stack IS 'JSONB array of technologies with usage counts: [{"name": "Python", "count": 45, "percentage": 0.65}]';
COMMENT ON COLUMN freelancer_github_profiles.domain_expertise IS 'JSONB object with domain percentages: {"frontend": 0.3, "backend": 0.6}';
COMMENT ON COLUMN freelancer_github_profiles.developer_story IS 'LLM-generated narrative summary of the developers work and expertise';

COMMENT ON COLUMN freelancer_repositories.languages IS 'JSONB object with language percentages: {"Python": 65.5, "JavaScript": 30.2}';
COMMENT ON COLUMN freelancer_repositories.user_commits IS 'Number of commits made by this specific user (vs total_commits in repo)';

COMMENT ON COLUMN freelancer_commits.changed_files IS 'JSONB array of file changes: [{"filename": "src/app.py", "additions": 45, "deletions": 12}]';
COMMENT ON COLUMN freelancer_commits.is_sampled IS 'TRUE if this commit was selected for deep LLM analysis (expensive operation)';
COMMENT ON COLUMN freelancer_commits.commit_type IS 'LLM-classified commit type: feature, bugfix, refactor, docs, test, chore, performance, style';

COMMENT ON FUNCTION get_freelancer_profile_summary IS 'Returns comprehensive profile summary with aggregated stats from all tables';
COMMENT ON FUNCTION get_commit_statistics IS 'Returns commit statistics for a profile including type distribution and averages';
COMMENT ON FUNCTION search_profiles_by_skills IS 'Search for profiles matching skill keywords with score thresholds';
