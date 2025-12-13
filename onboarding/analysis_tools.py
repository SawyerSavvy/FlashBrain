"""
Analysis Tools

MCP-style tools for analyzing stored GitHub data using LLM.
These tools READ from Supabase and use LLM to generate insights.
"""

import os
import json
import logging
import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from collections import Counter
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# RAG Service configuration
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8080")

# Mapping of skills to relevant file extensions
SKILL_TO_EXTENSIONS = {
    # Languages
    "python": ["py", "pyx", "pyw", "ipynb"],
    "javascript": ["js", "jsx", "mjs", "cjs"],
    "typescript": ["ts", "tsx"],
    "java": ["java", "jar"],
    "go": ["go"],
    "rust": ["rs"],
    "c++": ["cpp", "cc", "cxx", "hpp", "h"],
    "c#": ["cs"],
    "ruby": ["rb", "erb"],
    "php": ["php", "phtml"],
    "swift": ["swift"],
    "kotlin": ["kt", "kts"],
    "scala": ["scala", "sc"],
    "r": ["r", "rmd"],
    "julia": ["jl"],
    # Frameworks
    "react": ["jsx", "tsx"],
    "vue": ["vue"],
    "angular": ["ts", "component.ts"],
    "django": ["py"],
    "flask": ["py"],
    "fastapi": ["py"],
    "express": ["js", "ts"],
    "spring": ["java"],
    "rails": ["rb"],
    "nextjs": ["jsx", "tsx", "js", "ts"],
    # Data/ML
    "tensorflow": ["py", "ipynb"],
    "pytorch": ["py", "ipynb"],
    "pandas": ["py", "ipynb"],
    "numpy": ["py", "ipynb"],
    "scikit-learn": ["py", "ipynb"],
    # DevOps
    "docker": ["dockerfile", "yml", "yaml"],
    "kubernetes": ["yml", "yaml"],
    "terraform": ["tf", "tfvars"],
    "ansible": ["yml", "yaml"],
    "ci/cd": ["yml", "yaml"],
    # Database
    "sql": ["sql", "psql"],
    "postgresql": ["sql", "psql"],
    "mongodb": ["js", "ts"],
    "graphql": ["graphql", "gql"],
}


def create_analysis_tools(supabase_client, llm: Optional[ChatGoogleGenerativeAI] = None):
    """
    Factory function to create analysis tools with dependencies.

    Args:
        supabase_client: Initialized Supabase client
        llm: Optional LLM instance (defaults to gemini-2.0-flash-exp)

    Returns:
        List of analysis tools
    """
    # Initialize LLM if not provided
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
        )

    @tool
    async def analyze_commit_patterns(profile_id: str) -> str:
        """
        Analyze commit patterns and calculate consistency score.

        This tool:
        1. Queries sampled commits from Supabase
        2. Uses LLM to classify commit types (feature, bugfix, refactor, etc.)
        3. Calculates commit consistency score based on frequency and distribution
        4. Updates profile with results

        Args:
            profile_id: Freelancer profile UUID

        Returns:
            JSON string with:
            - consistency_score: Float 0-100
            - commit_type_distribution: Dict of commit types and percentages
            - total_commits_analyzed: Number of commits
        """
        try:
            print(f"[DEBUG] analyze_commit_patterns START for profile {profile_id}")
            logger.info(f"Analyzing commit patterns for profile {profile_id}")

            # ============================================================
            # STEP 1: Fetch Sampled Commits
            # ============================================================
            print("[DEBUG] STEP 1: Fetching sampled commits...")
            sampled_commits = supabase_client.table("freelancer_commits").select(
                "id, commit_sha, commit_message, commit_date, commit_type"
            ).eq("profile_id", profile_id).eq("is_sampled", True).execute()

            print(f"[DEBUG] STEP 1: Query returned {len(sampled_commits.data) if sampled_commits.data else 0} commits")

            if not sampled_commits.data:
                print("[DEBUG] STEP 1: No sampled commits found - returning error")
                return json.dumps({
                    "status": "error",
                    "message": "No sampled commits found for this profile"
                })

            commits = sampled_commits.data
            logger.info(f"Found {len(commits)} sampled commits")
            print(f"[DEBUG] STEP 1: Sample commit data: {commits[0] if commits else 'None'}")

            # ============================================================
            # STEP 2: LLM Classification of Commit Types
            # ============================================================
            print("[DEBUG] STEP 2: Starting LLM classification...")
            unclassified = [c for c in commits if not c.get("commit_type")]
            print(f"[DEBUG] STEP 2: Found {len(unclassified)} unclassified commits")

            if unclassified:
                logger.info(f"Classifying {len(unclassified)} commits with LLM")

                # Prepare batch prompt (handle None messages)
                commit_messages = "\n".join([
                    f"{i+1}. {(c.get('commit_message') or 'No message')[:200]}"
                    for i, c in enumerate(unclassified)
                ])
                print(f"[DEBUG] STEP 2: Commit messages prepared, length={len(commit_messages)}")

                classification_prompt = f"""Analyze these Git commit messages and classify each as ONE of these types:
- feature: New functionality or capabilities
- bugfix: Bug fixes or error corrections
- refactor: Code restructuring without changing behavior
- docs: Documentation changes
- test: Test additions or modifications
- style: Code formatting, whitespace, naming
- chore: Build scripts, dependencies, tooling
- performance: Performance improvements

Commit messages:
{commit_messages}

Respond with ONLY a JSON array of classifications in this exact format:
[{{"index": 1, "type": "feature"}}, {{"index": 2, "type": "bugfix"}}, ...]"""

                print("[DEBUG] STEP 2: Calling LLM...")
                response = await llm.ainvoke(classification_prompt)
                response_text = response.content.strip()
                print(f"[DEBUG] STEP 2: LLM response length={len(response_text)}, preview={response_text[:200]}")

                # Parse LLM response
                try:
                    # Remove markdown code blocks if present
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()

                    print(f"[DEBUG] STEP 2: Parsing JSON: {response_text[:200]}")
                    classifications = json.loads(response_text)
                    print(f"[DEBUG] STEP 2: Parsed {len(classifications)} classifications")

                    # Update commits with classifications
                    for item in classifications:
                        idx = item["index"] - 1  # Convert to 0-indexed
                        if 0 <= idx < len(unclassified):
                            commit_id = unclassified[idx]["id"]
                            commit_type = item["type"]

                            # Update in database
                            supabase_client.table("freelancer_commits").update({
                                "commit_type": commit_type
                            }).eq("id", commit_id).execute()

                            # Update local copy
                            unclassified[idx]["commit_type"] = commit_type

                    print(f"[DEBUG] STEP 2: Updated {len(classifications)} commits in database")

                except json.JSONDecodeError as e:
                    print(f"[DEBUG] STEP 2: JSON parse error: {e}")
                    logger.warning(f"Failed to parse LLM classifications: {e}")
                    # Fallback: classify as 'other'
                    for c in unclassified:
                        supabase_client.table("freelancer_commits").update({
                            "commit_type": "other"
                        }).eq("id", c["id"]).execute()
            else:
                print("[DEBUG] STEP 2: All commits already classified, skipping LLM")

            # ============================================================
            # STEP 3: Calculate Commit Type Distribution
            # ============================================================
            print("[DEBUG] STEP 3: Calculating commit type distribution...")
            commits = supabase_client.table("freelancer_commits").select(
                "commit_type, commit_date"
            ).eq("profile_id", profile_id).eq("is_sampled", True).execute().data

            print(f"[DEBUG] STEP 3: Re-fetched {len(commits)} commits")

            type_counts = Counter(c.get("commit_type", "other") for c in commits)
            total = len(commits)
            print(f"[DEBUG] STEP 3: Type counts: {dict(type_counts)}")

            type_distribution = {
                commit_type: round((count / total) * 100, 2)
                for commit_type, count in type_counts.items()
            }
            print(f"[DEBUG] STEP 3: Type distribution: {type_distribution}")

            # ============================================================
            # STEP 4: Calculate Consistency Score
            # ============================================================
            print("[DEBUG] STEP 4: Fetching all commits for consistency calculation...")
            all_commits = supabase_client.table("freelancer_commits").select(
                "commit_date"
            ).eq("profile_id", profile_id).execute().data

            print(f"[DEBUG] STEP 4: Fetched {len(all_commits)} total commits")

            # Parse dates and calculate consistency
            dates = []
            parse_errors = 0
            for c in all_commits:
                if c.get("commit_date"):
                    try:
                        date_str = c["commit_date"]
                        # Handle various ISO formats
                        if date_str.endswith("Z"):
                            date_str = date_str[:-1] + "+00:00"
                        dates.append(datetime.fromisoformat(date_str))
                    except (ValueError, TypeError) as e:
                        parse_errors += 1
                        logger.warning(f"Could not parse date '{c.get('commit_date')}': {e}")

            print(f"[DEBUG] STEP 4: Parsed {len(dates)} dates, {parse_errors} errors")

            # Initialize variables
            active_days = 0
            consistency_score = 50.0  # Default

            if len(dates) < 2:
                print(f"[DEBUG] STEP 4: Not enough dates ({len(dates)}), using default score")
                consistency_score = 50.0  # Not enough data
                active_days = len(dates)
            else:
                dates.sort()

                # Calculate metrics
                total_days = (dates[-1] - dates[0]).days + 1
                active_days = len(set(d.date() for d in dates))
                activity_ratio = active_days / total_days if total_days > 0 else 0

                # Calculate commit frequency (commits per active day)
                commits_per_day = len(dates) / active_days if active_days > 0 else 0

                print(f"[DEBUG] STEP 4: total_days={total_days}, active_days={active_days}, activity_ratio={activity_ratio}, commits_per_day={commits_per_day}")

                # Score based on activity ratio and frequency
                # High consistency = high activity ratio + regular commits
                consistency_score = min(100, (activity_ratio * 50) + (min(commits_per_day, 5) * 10))
                consistency_score = round(consistency_score, 2)

            print(f"[DEBUG] STEP 4: Final consistency_score={consistency_score}")

            # ============================================================
            # STEP 5: Update Profile
            # ============================================================
            print("[DEBUG] STEP 5: Updating profile in database...")
            supabase_client.table("freelancer_github_profiles").update({
                "commit_consistency_score": consistency_score,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", profile_id).execute()

            logger.info(f"Consistency score: {consistency_score}")
            print(f"[DEBUG] STEP 5: Profile updated successfully")

            result = {
                "status": "success",
                "consistency_score": consistency_score,
                "commit_type_distribution": type_distribution,
                "total_commits_analyzed": total,
                "active_days": active_days,
                "message": f"✅ Analyzed {total} commits, consistency: {consistency_score}/100"
            }
            print(f"[DEBUG] analyze_commit_patterns COMPLETE: {result}")
            return json.dumps(result)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[DEBUG] analyze_commit_patterns FAILED: {e}")
            print(f"[DEBUG] Full traceback:\n{error_trace}")
            logger.error(f"Failed to analyze commit patterns: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"❌ Failed to analyze commit patterns: {str(e)}",
                "traceback": error_trace
            })

    @tool
    async def calculate_domain_expertise(profile_id: str) -> str:
        """
        Calculate domain expertise percentages from file changes.

        This tool:
        1. Analyzes file extensions from commit data
        2. Maps extensions to domains (frontend, backend, ML, DevOps, etc.)
        3. Calculates percentage distribution
        4. Updates profile with domain_expertise JSONB

        Args:
            profile_id: Freelancer profile UUID

        Returns:
            JSON string with domain percentages
        """
        try:
            logger.info(f"Calculating domain expertise for profile {profile_id}")

            # ============================================================
            # STEP 1: Fetch File Extensions from Commits
            # ============================================================
            commits = supabase_client.table("freelancer_commits").select(
                "file_extensions"
            ).eq("profile_id", profile_id).execute().data

            # Aggregate all file extensions
            extension_counts = Counter()
            for commit in commits:
                extensions = commit.get("file_extensions", [])
                if extensions:
                    extension_counts.update(extensions)

            if not extension_counts:
                # Fallback: check repositories for primary language
                repos = supabase_client.table("freelancer_repositories").select(
                    "primary_language, languages"
                ).eq("profile_id", profile_id).execute().data

                # Use primary languages as proxy
                lang_counts = Counter()
                for repo in repos:
                    if repo.get("primary_language"):
                        lang_counts[repo["primary_language"]] += 1

                if not lang_counts:
                    return json.dumps({
                        "status": "error",
                        "message": "No file extension or language data found"
                    })

                # Map languages to extensions (approximation)
                lang_to_ext = {
                    "Python": "py", "JavaScript": "js", "TypeScript": "ts",
                    "Java": "java", "Go": "go", "Rust": "rs", "C++": "cpp",
                    "Ruby": "rb", "PHP": "php", "Swift": "swift", "Kotlin": "kt"
                }
                for lang, count in lang_counts.items():
                    ext = lang_to_ext.get(lang, lang.lower())
                    extension_counts[ext] = count

            # ============================================================
            # STEP 2: Map Extensions to Domains
            # ============================================================
            domain_mapping = {
                "frontend": ["js", "jsx", "ts", "tsx", "vue", "html", "css", "scss", "sass", "less"],
                "backend": ["py", "java", "go", "rs", "rb", "php", "cs", "kt", "scala", "ex"],
                "mobile": ["swift", "kt", "java", "dart", "m", "mm"],
                "ml_data_science": ["py", "ipynb", "r", "jl", "mat"],
                "devops": ["yml", "yaml", "dockerfile", "sh", "bash", "tf", "hcl"],
                "database": ["sql", "psql", "mysql", "mongo", "graphql"],
                "testing": ["test.js", "spec.js", "test.py", "spec.ts"],
                "documentation": ["md", "rst", "txt", "doc", "docx"],
            }

            domain_counts = Counter()

            for ext, count in extension_counts.items():
                ext_lower = ext.lower().strip(".")
                matched = False

                for domain, extensions in domain_mapping.items():
                    if ext_lower in extensions or any(ext_lower.endswith(e) for e in extensions):
                        domain_counts[domain] += count
                        matched = True
                        break

                if not matched:
                    domain_counts["other"] += count

            # ============================================================
            # STEP 3: Calculate Percentages
            # ============================================================
            total = sum(domain_counts.values())
            domain_expertise = {
                domain: round((count / total) * 100, 2)
                for domain, count in domain_counts.items()
                if count > 0
            }

            # Sort by percentage
            domain_expertise = dict(sorted(domain_expertise.items(), key=lambda x: x[1], reverse=True))

            # ============================================================
            # STEP 4: Update Profile
            # ============================================================
            supabase_client.table("freelancer_github_profiles").update({
                "domain_expertise": domain_expertise,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", profile_id).execute()

            logger.info(f"Domain expertise calculated: {domain_expertise}")

            return json.dumps({
                "status": "success",
                "domain_expertise": domain_expertise,
                "total_files_analyzed": total,
                "message": f"✅ Calculated domain expertise from {total} file changes"
            })

        except Exception as e:
            logger.error(f"Failed to calculate domain expertise: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"❌ Failed to calculate domain expertise: {str(e)}"
            })

    @tool
    async def calculate_impact_score(profile_id: str) -> str:
        """
        Calculate impact score based on commit complexity, repo popularity, and code volume.

        This tool:
        1. Weighs commits by lines changed (additions + deletions)
        2. Considers repository stars and forks
        3. Factors in commit frequency and consistency
        4. Updates profile with impact_score

        Args:
            profile_id: Freelancer profile UUID

        Returns:
            JSON string with impact_score (0-100)
        """
        try:
            logger.info(f"Calculating impact score for profile {profile_id}")

            # ============================================================
            # STEP 1: Fetch Profile Data
            # ============================================================
            profile = supabase_client.table("freelancer_github_profiles").select(
                "public_repos, followers, total_stars"
            ).eq("id", profile_id).single().execute().data

            if not profile:
                return json.dumps({
                    "status": "error",
                    "message": "Profile not found"
                })

            # ============================================================
            # STEP 2: Fetch Repository Metrics
            # ============================================================
            repos = supabase_client.table("freelancer_repositories").select(
                "stars, forks, user_commits"
            ).eq("profile_id", profile_id).execute().data

            total_stars = sum(r.get("stars", 0) for r in repos)
            total_forks = sum(r.get("forks", 0) for r in repos)
            total_user_commits = sum(r.get("user_commits", 0) for r in repos)

            # ============================================================
            # STEP 3: Fetch Commit Metrics
            # ============================================================
            commits = supabase_client.table("freelancer_commits").select(
                "additions, deletions, files_changed"
            ).eq("profile_id", profile_id).execute().data

            total_additions = sum(c.get("additions", 0) for c in commits if c.get("additions"))
            total_deletions = sum(c.get("deletions", 0) for c in commits if c.get("deletions"))

            # ============================================================
            # STEP 4: Calculate Impact Score (0-100)
            # ============================================================
            # Component 1: Repository Popularity (0-30 points)
            star_score = min(30, (total_stars / 100) * 30)  # 100 stars = max

            # Component 2: Code Volume (0-30 points)
            code_volume = total_additions + total_deletions
            volume_score = min(30, (code_volume / 10000) * 30)  # 10k lines = max

            # Component 3: Commit Activity (0-20 points)
            commit_score = min(20, (total_user_commits / 500) * 20)  # 500 commits = max

            # Component 4: Collaboration (0-20 points)
            fork_score = min(10, (total_forks / 50) * 10)  # 50 forks = max
            follower_score = min(10, (profile.get("followers", 0) / 100) * 10)  # 100 followers = max
            collaboration_score = fork_score + follower_score

            # Total Impact Score
            impact_score = round(star_score + volume_score + commit_score + collaboration_score, 2)

            # ============================================================
            # STEP 5: Update Profile
            # ============================================================
            supabase_client.table("freelancer_github_profiles").update({
                "impact_score": impact_score,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", profile_id).execute()

            logger.info(f"Impact score: {impact_score}")

            return json.dumps({
                "status": "success",
                "impact_score": impact_score,
                "components": {
                    "repository_popularity": round(star_score, 2),
                    "code_volume": round(volume_score, 2),
                    "commit_activity": round(commit_score, 2),
                    "collaboration": round(collaboration_score, 2)
                },
                "metrics": {
                    "total_stars": total_stars,
                    "total_commits": total_user_commits,
                    "total_code_changes": code_volume,
                    "total_forks": total_forks
                },
                "message": f"✅ Impact score calculated: {impact_score}/100"
            })

        except Exception as e:
            logger.error(f"Failed to calculate impact score: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"❌ Failed to calculate impact score: {str(e)}"
            })

    @tool
    async def generate_developer_story(profile_id: str) -> str:
        """
        Generate a narrative developer story using LLM.

        This tool:
        1. Fetches all profile data from Supabase
        2. Uses LLM to create a compelling narrative
        3. Highlights key projects, skills, and contributions
        4. Updates profile with developer_story

        Args:
            profile_id: Freelancer profile UUID

        Returns:
            JSON string with developer_story text
        """
        try:
            logger.info(f"Generating developer story for profile {profile_id}")

            # ============================================================
            # STEP 1: Fetch All Profile Data
            # ============================================================
            profile = supabase_client.table("freelancer_github_profiles").select("*").eq(
                "id", profile_id
            ).single().execute().data

            if not profile:
                return json.dumps({
                    "status": "error",
                    "message": "Profile not found"
                })

            repos = supabase_client.table("freelancer_repositories").select(
                "repo_name, description, stars, primary_language, technologies"
            ).eq("profile_id", profile_id).order("stars", desc=True).limit(10).execute().data

            commits = supabase_client.table("freelancer_commits").select(
                "commit_type"
            ).eq("profile_id", profile_id).execute().data

            # ============================================================
            # STEP 2: Prepare Data Summary for LLM
            # ============================================================
            # Commit type distribution
            commit_types = Counter(c.get("commit_type", "other") for c in commits)
            top_commit_types = commit_types.most_common(3)

            # Top repositories
            top_repos_summary = "\n".join([
                f"- {r['repo_name']}: {r.get('description', 'No description')} ({r.get('stars', 0)} stars) - {r.get('primary_language', 'Unknown')}"
                for r in repos[:5]
            ])

            # ============================================================
            # STEP 3: Generate Story with LLM
            # ============================================================
            story_prompt = f"""Create a compelling 2-3 paragraph developer story for this GitHub profile.

**Profile Info:**
- Name: {profile.get('name') or profile.get('github_username')}
- Bio: {profile.get('bio') or 'Not provided'}
- Location: {profile.get('location') or 'Unknown'}
- Public Repos: {profile.get('public_repos', 0)}
- Followers: {profile.get('followers', 0)}
- Total Stars: {profile.get('total_stars', 0)}

**Top Repositories:**
{top_repos_summary or 'No repositories found'}

**Expertise:**
- Domain Expertise: {json.dumps(profile.get('domain_expertise', {}), indent=2)}
- Consistency Score: {profile.get('commit_consistency_score', 'N/A')}/100
- Impact Score: {profile.get('impact_score', 'N/A')}/100

**Commit Patterns:**
- Most common commit types: {', '.join([f'{t[0]} ({t[1]} commits)' for t in top_commit_types])}

**Instructions:**
1. Write a narrative that highlights their strengths and specialization
2. Mention notable projects or achievements
3. Describe their coding style and contribution patterns
4. Keep it professional but engaging
5. Focus on what makes them unique as a developer
6. Do NOT use markdown formatting, just plain text paragraphs

Write the story now:"""

            response = await llm.ainvoke(story_prompt)
            developer_story = response.content.strip()

            # ============================================================
            # STEP 4: Update Profile
            # ============================================================
            supabase_client.table("freelancer_github_profiles").update({
                "developer_story": developer_story,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", profile_id).execute()

            logger.info("Developer story generated successfully")

            return json.dumps({
                "status": "success",
                "developer_story": developer_story,
                "message": "✅ Developer story generated"
            })

        except Exception as e:
            logger.error(f"Failed to generate developer story: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"❌ Failed to generate developer story: {str(e)}"
            })

    @tool
    async def verify_skill_evidence(
        freelancer_id: str,
        skill_id: Optional[str] = None,
        max_skills: int = 20
    ) -> str:
        """
        Verify self-reported skills against GitHub evidence.

        This tool:
        1. Gets skills from freelancer_skills table (or specific skill)
        2. Searches RAG service for skill mentions in READMEs
        3. Checks commit file extensions for relevant patterns
        4. Calculates confidence score based on evidence
        5. Updates freelancer_skills with evidence_strength

        Args:
            freelancer_id: The freelancer's UUID
            skill_id: Optional specific skill UUID to verify (verifies all if not provided)
            max_skills: Maximum number of skills to verify in one call (default 20)

        Returns:
            JSON string with verification results for each skill
        """
        try:
            print(f"[DEBUG] verify_skill_evidence START for freelancer {freelancer_id}")
            logger.info(f"Verifying skills for freelancer {freelancer_id}")

            # ============================================================
            # STEP 1: Get Freelancer's GitHub Profile
            # ============================================================
            print("[DEBUG] STEP 1: Getting freelancer's GitHub profile...")

            # Get the profile_id from freelancer_github_profiles using freelancer_id
            profile_result = supabase_client.table("freelancer_github_profiles").select(
                "id, github_username"
            ).eq("freelancer_id", freelancer_id).execute()

            if not profile_result.data:
                return json.dumps({
                    "status": "error",
                    "message": f"No GitHub profile found for freelancer {freelancer_id}"
                })

            profile = profile_result.data[0]
            profile_id = profile["id"]
            github_username = profile.get("github_username")
            print(f"[DEBUG] STEP 1: Found profile {profile_id} for user {github_username}")

            # ============================================================
            # STEP 2: Get Skills Needing Verification
            # ============================================================
            print("[DEBUG] STEP 2: Getting skills needing verification...")

            if skill_id:
                # Verify specific skill
                skills_query = supabase_client.table("freelancer_skills").select(
                    "id, skill_id, proficiency_level, evidence_strength, confidence_score"
                ).eq("freelancer_id", freelancer_id).eq("skill_id", skill_id)
            else:
                # Get skills that need verification (self_reported or never verified)
                skills_query = supabase_client.table("freelancer_skills").select(
                    "id, skill_id, proficiency_level, evidence_strength, confidence_score"
                ).eq("freelancer_id", freelancer_id).in_(
                    "evidence_strength", ["self_reported", None]
                ).limit(max_skills)

            skills_result = skills_query.execute()

            if not skills_result.data:
                return json.dumps({
                    "status": "success",
                    "message": "No skills needing verification found",
                    "skills_verified": 0
                })

            freelancer_skills = skills_result.data
            print(f"[DEBUG] STEP 2: Found {len(freelancer_skills)} skills to verify")

            # ============================================================
            # STEP 3: Get Skill Details and Aliases
            # ============================================================
            print("[DEBUG] STEP 3: Getting skill details and aliases...")

            skill_ids = [s["skill_id"] for s in freelancer_skills]

            # Get skill canonical names
            skills_details = supabase_client.table("skills").select(
                "id, canonical_name, normalized_name"
            ).in_("id", skill_ids).execute().data

            skill_map = {s["id"]: s for s in skills_details}

            # Get aliases for all skills
            aliases_result = supabase_client.table("skill_aliases").select(
                "skill_id, alias, alias_normalized"
            ).in_("skill_id", skill_ids).eq("is_active", True).execute()

            alias_map = {}
            for alias in (aliases_result.data or []):
                sid = alias["skill_id"]
                if sid not in alias_map:
                    alias_map[sid] = []
                alias_map[sid].append(alias["alias_normalized"] or alias["alias"].lower())

            print(f"[DEBUG] STEP 3: Loaded details for {len(skill_map)} skills with {sum(len(v) for v in alias_map.values())} aliases")

            # ============================================================
            # STEP 4: Get Commit File Extensions for Profile
            # ============================================================
            print("[DEBUG] STEP 4: Getting commit file extensions...")

            commits_result = supabase_client.table("freelancer_commits").select(
                "file_extensions"
            ).eq("profile_id", profile_id).execute()

            # Aggregate all file extensions
            all_extensions = Counter()
            for commit in (commits_result.data or []):
                extensions = commit.get("file_extensions") or []
                all_extensions.update([ext.lower().strip(".") for ext in extensions])

            total_files = sum(all_extensions.values())
            print(f"[DEBUG] STEP 4: Found {total_files} files with {len(all_extensions)} unique extensions")

            # ============================================================
            # STEP 5: Verify Each Skill
            # ============================================================
            print("[DEBUG] STEP 5: Verifying each skill...")

            verification_results = []

            async with httpx.AsyncClient(timeout=30.0) as http_client:
                for fs in freelancer_skills:
                    sid = fs["skill_id"]
                    skill_info = skill_map.get(sid, {})
                    canonical_name = skill_info.get("canonical_name", "Unknown")
                    normalized_name = skill_info.get("normalized_name", canonical_name.lower())

                    print(f"[DEBUG] STEP 5: Verifying skill '{canonical_name}'...")

                    # Build search terms (skill name + aliases)
                    search_terms = [normalized_name]
                    search_terms.extend(alias_map.get(sid, []))

                    # Evidence collection
                    evidence = {
                        "rag_mentions": [],
                        "file_extension_matches": [],
                        "search_terms_used": search_terms,
                    }

                    confidence = 0.0

                    # ----- RAG Search -----
                    try:
                        # Search for skill mentions in READMEs
                        rag_query = f"{canonical_name} {' '.join(alias_map.get(sid, [])[:3])}"

                        rag_response = await http_client.post(
                            f"{RAG_SERVICE_URL}/mcp/tools/search_knowledge_base",
                            json={
                                "query": rag_query,
                                "match_threshold": 0.6,
                                "match_count": 5,
                                # Filter by documents belonging to this freelancer's repos
                            }
                        )

                        if rag_response.status_code == 200:
                            rag_data = rag_response.json()
                            results = rag_data.get("results", [])

                            # Check if any results mention this freelancer's repos
                            for result in results:
                                content = result.get("content", "").lower()
                                similarity = result.get("similarity", 0)

                                # Check if skill or aliases are mentioned
                                for term in search_terms:
                                    if term in content:
                                        evidence["rag_mentions"].append({
                                            "term": term,
                                            "similarity": similarity,
                                            "excerpt": content[:200]
                                        })
                                        confidence += 0.15 * similarity
                                        break

                            print(f"[DEBUG] RAG search found {len(evidence['rag_mentions'])} mentions")
                        else:
                            print(f"[DEBUG] RAG search failed: {rag_response.status_code}")

                    except Exception as e:
                        print(f"[DEBUG] RAG search error: {e}")
                        logger.warning(f"RAG search failed for skill {canonical_name}: {e}")

                    # ----- File Extension Matching -----
                    skill_extensions = SKILL_TO_EXTENSIONS.get(normalized_name, [])

                    # Also check aliases
                    for alias in alias_map.get(sid, []):
                        if alias in SKILL_TO_EXTENSIONS:
                            skill_extensions.extend(SKILL_TO_EXTENSIONS[alias])

                    skill_extensions = list(set(skill_extensions))

                    if skill_extensions:
                        matched_count = 0
                        for ext in skill_extensions:
                            ext_count = all_extensions.get(ext, 0)
                            if ext_count > 0:
                                matched_count += ext_count
                                evidence["file_extension_matches"].append({
                                    "extension": ext,
                                    "count": ext_count
                                })

                        if matched_count > 0 and total_files > 0:
                            # Calculate extension-based confidence
                            ext_ratio = matched_count / total_files
                            ext_confidence = min(0.5, ext_ratio * 2)  # Max 0.5 from extensions
                            confidence += ext_confidence
                            print(f"[DEBUG] File extensions: {matched_count}/{total_files} files match ({ext_confidence:.2f} confidence)")

                    # ----- Determine Evidence Strength -----
                    confidence = min(1.0, confidence)  # Cap at 1.0

                    if confidence >= 0.7:
                        new_evidence_strength = "github_verified"
                    elif confidence >= 0.4:
                        new_evidence_strength = "github_partial"
                    else:
                        new_evidence_strength = "self_reported"

                    print(f"[DEBUG] Skill '{canonical_name}': confidence={confidence:.2f}, strength={new_evidence_strength}")

                    # ----- Update Database -----
                    update_data = {
                        "evidence_strength": new_evidence_strength,
                        "confidence_score": max(fs.get("confidence_score", 0) or 0, confidence),
                        "github_verified_at": datetime.now(timezone.utc).isoformat(),
                        "github_repos_checked": len(commits_result.data or []),
                        "metadata": {
                            **(fs.get("metadata") or {}),
                            "github_evidence": evidence
                        },
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }

                    supabase_client.table("freelancer_skills").update(
                        update_data
                    ).eq("id", fs["id"]).execute()

                    verification_results.append({
                        "skill_name": canonical_name,
                        "skill_id": sid,
                        "previous_strength": fs.get("evidence_strength"),
                        "new_strength": new_evidence_strength,
                        "confidence": round(confidence, 2),
                        "rag_mentions": len(evidence["rag_mentions"]),
                        "file_matches": len(evidence["file_extension_matches"])
                    })

            # ============================================================
            # STEP 6: Return Results
            # ============================================================
            print(f"[DEBUG] verify_skill_evidence COMPLETE: Verified {len(verification_results)} skills")

            # Summarize results
            verified_count = sum(1 for r in verification_results if r["new_strength"] == "github_verified")
            partial_count = sum(1 for r in verification_results if r["new_strength"] == "github_partial")

            return json.dumps({
                "status": "success",
                "skills_verified": len(verification_results),
                "github_verified": verified_count,
                "github_partial": partial_count,
                "self_reported": len(verification_results) - verified_count - partial_count,
                "results": verification_results,
                "message": f"✅ Verified {len(verification_results)} skills: {verified_count} verified, {partial_count} partial"
            })

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[DEBUG] verify_skill_evidence FAILED: {e}")
            print(f"[DEBUG] Full traceback:\n{error_trace}")
            logger.error(f"Failed to verify skill evidence: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"❌ Failed to verify skill evidence: {str(e)}",
                "traceback": error_trace
            })

    return [
        analyze_commit_patterns,
        calculate_domain_expertise,
        calculate_impact_score,
        generate_developer_story,
        verify_skill_evidence,
    ]
