"""
Data Extraction Tools

MCP-style tools for extracting GitHub data and storing in Supabase.
These tools do NOT use LLM - just GitHub API calls and database writes.
"""

import json
import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
import httpx
from langchain_core.tools import tool

from github_client import GitHubClient, parse_language_percentages

logger = logging.getLogger(__name__)


def create_extraction_tools(github_client: GitHubClient, supabase_client):
    """
    Factory function to create extraction tools with dependencies.

    Args:
        github_client: Initialized GitHubClient instance
        supabase_client: Initialized Supabase client

    Returns:
        List of extraction tools
    """

    @tool
    async def extract_github_data_to_database(
        github_username: str,
        client_id: Optional[str] = None,
        max_repos: int = 50,
        commits_per_repo: int = 100,
        sample_size: int = 20
    ) -> str:
        """
        Extract ALL GitHub data for a user and store in Supabase.

        This is the main extraction tool that:
        1. Fetches user profile from GitHub
        2. Fetches all repositories (up to max_repos)
        3. Fetches commits for top repositories
        4. Samples commits for deep analysis
        5. Stores everything in 3 Supabase tables

        NO LLM calls - just GitHub API + database writes (fast & cheap).

        Args:
            github_username: GitHub username to analyze
            client_id: Optional client ID for multi-tenancy
            max_repos: Maximum repositories to fetch (default 50)
            commits_per_repo: Max commits per repo (default 100)
            sample_size: Commits to sample per repo for deep analysis (default 20)

        Returns:
            JSON string with:
            - profile_id: UUID of created profile
            - repos_stored: Number of repositories stored
            - commits_stored: Number of commits stored
            - sampled_commits: Number of commits marked for LLM analysis
        """
        try:
            logger.info(f"Starting GitHub data extraction for @{github_username}")

            # ============================================================
            # STEP 1: Fetch User Profile
            # ============================================================
            logger.info(f"Fetching profile for @{github_username}")
            user_profile = await github_client.get_user_profile(github_username)

            # Check if profile already exists
            existing_profile = supabase_client.table("freelancer_github_profiles").select("id").eq(
                "github_username", github_username
            ).execute()

            if existing_profile.data:
                profile_id = existing_profile.data[0]["id"]
                logger.info(f"Profile exists, updating: {profile_id}")
                action = "updated"
            else:
                profile_id = None
                action = "created"

            # Prepare profile data
            profile_data = {
                "github_username": user_profile["login"],
                "github_user_id": user_profile["id"],
                "name": user_profile.get("name"),
                "bio": user_profile.get("bio"),
                "location": user_profile.get("location"),
                "company": user_profile.get("company"),
                "email": user_profile.get("email"),
                "blog_url": user_profile.get("websiteUrl"),
                "twitter_username": user_profile.get("twitterUsername"),
                "avatar_url": user_profile.get("avatarUrl"),
                "public_repos": user_profile["repositories"]["totalCount"],
                "followers": user_profile["followers"]["totalCount"],
                "following": user_profile["following"]["totalCount"],
                "total_stars": user_profile["starredRepositories"]["totalCount"],
            }

            if client_id:
                profile_data["client_id"] = client_id

            # Insert or update profile
            if profile_id:
                supabase_client.table("freelancer_github_profiles").update(profile_data).eq("id", profile_id).execute()
            else:
                result = supabase_client.table("freelancer_github_profiles").insert(profile_data).execute()
                profile_id = result.data[0]["id"]

            logger.info(f"Profile {action}: {profile_id}")

            # ============================================================
            # STEP 2: Fetch Repositories
            # ============================================================
            logger.info(f"Fetching repositories (max {max_repos})")
            repositories = await github_client.get_user_repositories(
                github_username,
                limit=max_repos
            )

            logger.info(f"Found {len(repositories)} repositories")

            # Sort by stars (most popular first)
            repositories.sort(key=lambda r: r.get("stargazerCount", 0), reverse=True)

            # Build batch of repository data
            repo_batch = []
            for repo in repositories:
                repo_data = {
                    "profile_id": profile_id,
                    "repo_name": repo["name"],
                    "repo_full_name": repo["nameWithOwner"],
                    "repo_url": repo["url"],
                    "description": repo.get("description"),
                    "stars": repo.get("stargazerCount", 0),
                    "forks": repo.get("forkCount", 0),
                    "watchers": repo.get("watchers", {}).get("totalCount", 0),
                    "open_issues": repo.get("openIssues", {}).get("totalCount", 0),
                    "is_fork": repo.get("isFork", False),
                    "primary_language": repo.get("primaryLanguage", {}).get("name") if repo.get("primaryLanguage") else None,
                }

                # Parse language percentages
                if repo.get("languages"):
                    languages = parse_language_percentages(repo["languages"])
                    repo_data["languages"] = languages

                # Extract commit count
                default_branch = repo.get("defaultBranchRef")
                if default_branch and default_branch.get("target"):
                    repo_data["total_commits"] = default_branch["target"]["history"]["totalCount"]

                # Parse GitHub timestamps
                if repo.get("createdAt"):
                    repo_data["created_at_github"] = repo["createdAt"]
                if repo.get("updatedAt"):
                    repo_data["updated_at_github"] = repo["updatedAt"]
                if repo.get("pushedAt"):
                    repo_data["pushed_at_github"] = repo["pushedAt"]

                repo_batch.append(repo_data)

            # Batch upsert all repositories (single request)
            if repo_batch:
                result = supabase_client.table("freelancer_repositories").upsert(
                    repo_batch,
                    on_conflict="profile_id,repo_full_name"
                ).execute()

                # Build repo_ids list from upsert response
                repo_ids = [
                    {
                        "repo_id": r["id"],
                        "repo_full_name": r["repo_full_name"],
                        "stars": r.get("stars", 0)
                    }
                    for r in result.data
                ]
                repos_stored = len(repo_ids)
            else:
                repo_ids = []
                repos_stored = 0

            logger.info(f"Upserted {repos_stored} repositories")

            # ============================================================
            # STEP 3: Fetch Commits (Top 10 Repos Only)
            # ============================================================
            top_repos = repo_ids[:10]  # Only analyze top 10 repos by stars
            logger.info(f"Fetching commits for top {len(top_repos)} repositories")

            total_commits_stored = 0
            total_sampled = 0

            for repo_info in top_repos:
                owner, repo_name = repo_info["repo_full_name"].split("/")

                try:
                    # Fetch commits by this author
                    commits = await github_client.get_repository_commits(
                        owner, repo_name, github_username, limit=commits_per_repo
                    )

                    if not commits:
                        logger.debug(f"No commits found for {repo_info['repo_full_name']}")
                        continue

                    logger.info(f"Found {len(commits)} commits in {repo_info['repo_full_name']}")

                    # Update repository with user commit count
                    supabase_client.table("freelancer_repositories").update({
                        "user_commits": len(commits)
                    }).eq("id", repo_info["repo_id"]).execute()

                    # Sample commits for deep analysis
                    sampled_indices = _sample_commit_indices(len(commits), sample_size)

                    # Build batch of commit data
                    commit_batch = []
                    for idx, commit in enumerate(commits):
                        is_sampled = idx in sampled_indices

                        commit_batch.append({
                            "profile_id": profile_id,
                            "repository_id": repo_info["repo_id"],
                            "commit_sha": commit["sha"],
                            "commit_message": commit.get("commit", {}).get("message"),
                            "commit_date": commit.get("commit", {}).get("author", {}).get("date"),
                            "author_name": commit.get("commit", {}).get("author", {}).get("name"),
                            "author_email": commit.get("commit", {}).get("author", {}).get("email"),
                            "is_sampled": is_sampled,
                            "sample_reason": "evenly_distributed" if is_sampled else None
                        })

                        if is_sampled:
                            total_sampled += 1

                    # Batch upsert all commits (single request per repo)
                    # ON CONFLICT updates metadata but preserves LLM analysis fields
                    if commit_batch:
                        supabase_client.table("freelancer_commits").upsert(
                            commit_batch,
                            on_conflict="repository_id,commit_sha"
                        ).execute()
                        total_commits_stored += len(commit_batch)
                        logger.debug(f"Upserted {len(commit_batch)} commits for {repo_info['repo_full_name']}")

                except Exception as e:
                    logger.warning(f"Error fetching commits for {repo_info['repo_full_name']}: {e}")
                    continue

            logger.info(f"Stored {total_commits_stored} commits ({total_sampled} sampled)")

            # ============================================================
            # STEP 4: Fetch File Details for Sampled Commits
            # ============================================================
            print("[DEBUG] STEP 4: Starting file extraction for sampled commits...")
            logger.info("Fetching file details for sampled commits...")
            files_extracted = 0

            # Query ALL sampled commits for this profile
            sampled_commits = supabase_client.table("freelancer_commits").select(
                "id, commit_sha, repository_id, file_extensions"
            ).eq("profile_id", profile_id).eq("is_sampled", True).execute()

            print(f"[DEBUG] STEP 4: Found {len(sampled_commits.data) if sampled_commits.data else 0} sampled commits")
            logger.info(f"Found {len(sampled_commits.data) if sampled_commits.data else 0} sampled commits")

            if sampled_commits.data:
                # Build repo_id to repo_full_name mapping from database
                # (in case repos were from a previous run)
                repos_for_mapping = supabase_client.table("freelancer_repositories").select(
                    "id, repo_full_name"
                ).eq("profile_id", profile_id).execute()

                repo_map = {r["id"]: r["repo_full_name"] for r in repos_for_mapping.data}
                print(f"[DEBUG] STEP 4: Built repo_map with {len(repo_map)} repos: {list(repo_map.values())[:3]}...")
                logger.info(f"Built repo_map with {len(repo_map)} repos")

                # Batch updates for efficiency
                file_updates = []
                skipped_already_has = 0
                skipped_no_repo = 0

                for commit in sampled_commits.data:
                    # Skip if already has file_extensions
                    if commit.get("file_extensions"):
                        skipped_already_has += 1
                        continue

                    repo_full_name = repo_map.get(commit["repository_id"])
                    if not repo_full_name:
                        skipped_no_repo += 1
                        print(f"[DEBUG] STEP 4: No repo for repository_id={commit['repository_id']}")
                        logger.warning(f"No repo found for repository_id {commit['repository_id']}")
                        continue

                    owner, repo_name = repo_full_name.split("/")

                    try:
                        # Fetch commit details with file info
                        print(f"[DEBUG] STEP 4: Fetching details for {owner}/{repo_name}/{commit['commit_sha'][:7]}...")
                        details = await github_client.get_commit_details(
                            owner, repo_name, commit["commit_sha"]
                        )

                        files = details.get("files", [])
                        stats = details.get("stats", {})

                        print(f"[DEBUG] STEP 4: Commit {commit['commit_sha'][:7]}: {len(files)} files, +{stats.get('additions', 0)}/-{stats.get('deletions', 0)}")
                        logger.info(f"Commit {commit['commit_sha'][:7]}: {len(files)} files changed")

                        # Extract file extensions and directories
                        extensions = set()
                        directories = set()

                        for f in files:
                            filename = f.get("filename", "")
                            if "/" in filename:
                                # Extract directory path
                                dir_path = "/".join(filename.split("/")[:-1])
                                directories.add(dir_path)

                            # Extract extension
                            if "." in filename:
                                ext = filename.rsplit(".", 1)[-1].lower()
                                extensions.add(f".{ext}")

                        print(f"[DEBUG] STEP 4: Extracted extensions={list(extensions)}, dirs={list(directories)[:3]}")

                        # Prepare update data (include profile_id for NOT NULL constraint)
                        file_updates.append({
                            "id": commit["id"],
                            "profile_id": profile_id,
                            "repository_id": commit["repository_id"],
                            "commit_sha": commit["commit_sha"],
                            "file_extensions": list(extensions) if extensions else [],
                            "directories": list(directories) if directories else [],
                            "files_changed": len(files),
                            "additions": stats.get("additions", 0),
                            "deletions": stats.get("deletions", 0),
                            "changed_files": [
                                {
                                    "filename": f.get("filename"),
                                    "status": f.get("status"),
                                    "additions": f.get("additions", 0),
                                    "deletions": f.get("deletions", 0)
                                }
                                for f in files[:50]  # Limit to 50 files per commit
                            ]
                        })

                        files_extracted += 1

                    except Exception as e:
                        import traceback
                        print(f"[DEBUG] STEP 4: Error fetching commit {commit['commit_sha'][:7]}: {e}")
                        print(f"[DEBUG] STEP 4: Traceback: {traceback.format_exc()}")
                        logger.error(f"Error fetching details for commit {commit['commit_sha'][:7]}: {e}")
                        continue

                print(f"[DEBUG] STEP 4: Summary - extracted={files_extracted}, skipped_already_has={skipped_already_has}, skipped_no_repo={skipped_no_repo}")

                # Batch upsert file details
                if file_updates:
                    print(f"[DEBUG] STEP 4: Upserting {len(file_updates)} commits with file details...")
                    logger.info(f"Upserting {len(file_updates)} commits with file details...")
                    result = supabase_client.table("freelancer_commits").upsert(
                        file_updates,
                        on_conflict="repository_id,commit_sha"
                    ).execute()
                    print(f"[DEBUG] STEP 4: Upsert result - {len(result.data) if result.data else 0} rows affected")
                    logger.info(f"✅ Extracted file details for {files_extracted} sampled commits")
                else:
                    print("[DEBUG] STEP 4: No file_updates to upsert")
                    logger.info("No commits needed file extraction (all already have file_extensions)")
            else:
                print("[DEBUG] STEP 4: No sampled commits found in database")

            # ============================================================
            # STEP 5: Return Summary
            # ============================================================
            return json.dumps({
                "status": "success",
                "action": action,
                "profile_id": profile_id,
                "github_username": github_username,
                "repos_stored": repos_stored,
                "commits_stored": total_commits_stored,
                "sampled_commits": total_sampled,
                "files_extracted": files_extracted,
                "message": f"✅ Extracted data for @{github_username}: {repos_stored} repos, {total_commits_stored} commits, {files_extracted} with file details"
            })

        except Exception as e:
            logger.error(f"Failed to extract GitHub data: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"❌ Failed to extract GitHub data: {str(e)}"
            })

    @tool
    async def store_readme_in_rag(
        profile_id: str,
        repository_id: str,
        repo_full_name: str,
        github_username: str,
        client_id: Optional[str] = None
    ) -> str:
        """
        Store repository README in RAG vector store for semantic search.

        This tool:
        1. Fetches README from GitHub
        2. Creates document metadata in Supabase documents table
        3. Calls RAG service to vectorize and store chunks

        Args:
            profile_id: Freelancer profile UUID
            repository_id: Repository UUID
            repo_full_name: Full repository name (e.g., "octocat/Hello-World")
            github_username: GitHub username
            client_id: Optional client ID

        Returns:
            JSON string with document_id and chunks_stored
        """
        try:
            owner, repo_name = repo_full_name.split("/")

            # Fetch README
            logger.info(f"Fetching README for {repo_full_name}")
            readme_content = await github_client.get_repository_readme(owner, repo_name)

            if not readme_content:
                return json.dumps({
                    "status": "skipped",
                    "message": f"No README found for {repo_full_name}"
                })

            # Create document metadata in Supabase
            doc_metadata = {
                "title": f"{repo_full_name} README",
                "source": f"https://github.com/{repo_full_name}",
                "source_type": "github_readme",
                "metadata": {
                    "profile_id": profile_id,
                    "repository_id": repository_id,
                    "github_username": github_username,
                    "repo_full_name": repo_full_name
                }
            }

            if client_id:
                doc_metadata["client_id"] = client_id

            # Store in documents table
            doc_result = supabase_client.table("documents").insert(doc_metadata).execute()
            document_id = doc_result.data[0]["id"]

            logger.info(f"Created document {document_id}, calling RAG service")

            # Call RAG service to process and vectorize
            rag_service_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8012")
            rag_endpoint = f"{rag_service_url.rstrip('/')}/mcp/tools/process_document"

            payload = {
                "document_id": document_id,
                "client_id": client_id,
                "content": readme_content,
                "source_type": "text"
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(rag_endpoint, json=payload)
                response.raise_for_status()
                rag_result = response.json()

            return json.dumps({
                "status": "success",
                "document_id": document_id,
                "chunk_count": rag_result.get("chunk_count", 0),
                "message": f"✅ Vectorized README for {repo_full_name}"
            })

        except Exception as e:
            logger.error(f"Failed to store README in RAG: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"❌ Failed to store README: {str(e)}"
            })

    @tool
    def get_profile_summary(profile_id: str) -> str:
        """
        Read stored GitHub data from Supabase for LLM analysis.

        This tool queries the database to retrieve:
        - Profile metadata and stats
        - Repository list with languages
        - Commit metadata (for sampled commits)

        Used by analysis tools to get data for LLM processing.

        Args:
            profile_id: Freelancer profile UUID

        Returns:
            JSON string with all stored data
        """
        try:
            # Fetch profile
            profile = supabase_client.table("freelancer_github_profiles").select("*").eq("id", profile_id).single().execute()

            if not profile.data:
                return json.dumps({
                    "status": "error",
                    "message": f"Profile {profile_id} not found"
                })

            # Fetch repositories
            repos = supabase_client.table("freelancer_repositories").select("*").eq("profile_id", profile_id).execute()

            # Fetch sampled commits (for LLM analysis)
            sampled_commits = supabase_client.table("freelancer_commits").select(
                "commit_sha, commit_message, commit_date, commit_type, file_extensions, directories"
            ).eq("profile_id", profile_id).eq("is_sampled", True).execute()

            # Fetch ALL commits metadata (for statistics)
            all_commits = supabase_client.table("freelancer_commits").select(
                "commit_date, commit_type, files_changed, additions, deletions, file_extensions"
            ).eq("profile_id", profile_id).execute()

            return json.dumps({
                "status": "success",
                "profile": profile.data,
                "repositories": repos.data,
                "sampled_commits": sampled_commits.data,
                "all_commits_metadata": all_commits.data,
                "message": f"✅ Retrieved data for profile {profile_id}"
            })

        except Exception as e:
            logger.error(f"Failed to get profile summary: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"❌ Failed to get profile summary: {str(e)}"
            })

    return [
        extract_github_data_to_database,
        store_readme_in_rag,
        get_profile_summary,
    ]


# ============================================================
# Helper Functions
# ============================================================

def _sample_commit_indices(total: int, sample_size: int) -> List[int]:
    """
    Sample commit indices evenly distributed across the range.

    Args:
        total: Total number of commits
        sample_size: Desired sample size

    Returns:
        List of indices to sample
    """
    if total <= sample_size:
        return list(range(total))

    # Evenly distribute samples
    step = total / sample_size
    indices = [int(i * step) for i in range(sample_size)]

    return indices
