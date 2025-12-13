"""
GitHub GraphQL API Client

Handles all GitHub API interactions using GraphQL for efficient data fetching.
Includes REST API fallbacks for specific operations.
"""

import os
import logging
from typing import Dict, List, Optional, Any
import httpx
import base64
from datetime import datetime

logger = logging.getLogger(__name__)


class GitHubClient:
    """GitHub GraphQL API client for fetching user and repository data."""

    GRAPHQL_ENDPOINT = "https://api.github.com/graphql"
    REST_API_BASE = "https://api.github.com"

    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub client.

        Args:
            token: GitHub personal access token (or from GITHUB_TOKEN env)

        Raises:
            ValueError: If token is not provided
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN is required")

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github.v3+json"
        }

    async def get_user_profile(self, username: str) -> Dict[str, Any]:
        """
        Fetch user profile information via GraphQL.

        Args:
            username: GitHub username

        Returns:
            Dictionary with user profile data:
            - id, login, name, bio, location, company
            - email, website_url, twitter_username
            - avatar_url, followers, following
            - repositories count, starred repositories count
            - created_at, updated_at

        Raises:
            httpx.HTTPError: On API request failure
            ValueError: On GraphQL errors
        """
        query = """
        query($username: String!) {
          user(login: $username) {
            id
            login
            name
            bio
            location
            company
            email
            websiteUrl
            twitterUsername
            avatarUrl
            followers {
              totalCount
            }
            following {
              totalCount
            }
            repositories(privacy: PUBLIC) {
              totalCount
            }
            starredRepositories {
              totalCount
            }
            createdAt
            updatedAt
          }
        }
        """

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.GRAPHQL_ENDPOINT,
                json={"query": query, "variables": {"username": username}},
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                error_msg = "; ".join([e.get("message", str(e)) for e in data["errors"]])
                raise ValueError(f"GitHub GraphQL error: {error_msg}")

            user_data = data.get("data", {}).get("user")
            if not user_data:
                raise ValueError(f"User '{username}' not found")

            return user_data

    async def get_user_repositories(
        self,
        username: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch user's public repositories with language breakdown via GraphQL.

        Args:
            username: GitHub username
            limit: Maximum repositories to fetch (default 100)

        Returns:
            List of repository dictionaries with:
            - name, nameWithOwner, description, url
            - stargazerCount, forkCount, watchers
            - primaryLanguage, languages (with percentages)
            - defaultBranchRef, total commits
            - isFork, createdAt, updatedAt, pushedAt

        Raises:
            httpx.HTTPError: On API request failure
            ValueError: On GraphQL errors
        """
        query = """
        query($username: String!, $limit: Int!) {
          user(login: $username) {
            repositories(
              first: $limit,
              privacy: PUBLIC,
              orderBy: {field: UPDATED_AT, direction: DESC},
              isFork: false
            ) {
              nodes {
                name
                nameWithOwner
                description
                url
                stargazerCount
                forkCount
                isFork
                watchers {
                  totalCount
                }
                openIssues: issues(states: OPEN) {
                  totalCount
                }
                primaryLanguage {
                  name
                }
                languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
                  edges {
                    size
                    node {
                      name
                    }
                  }
                  totalSize
                }
                defaultBranchRef {
                  name
                  target {
                    ... on Commit {
                      history(first: 0) {
                        totalCount
                      }
                    }
                  }
                }
                createdAt
                updatedAt
                pushedAt
              }
            }
          }
        }
        """

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.GRAPHQL_ENDPOINT,
                json={"query": query, "variables": {"username": username, "limit": limit}},
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                error_msg = "; ".join([e.get("message", str(e)) for e in data["errors"]])
                raise ValueError(f"GitHub GraphQL error: {error_msg}")

            repositories = data.get("data", {}).get("user", {}).get("repositories", {}).get("nodes", [])
            return repositories if repositories else []

    async def get_repository_commits(
        self,
        owner: str,
        repo_name: str,
        author: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch commits by a specific author in a repository via REST API.

        Note: Using REST API because GraphQL doesn't support filtering by author username.

        Args:
            owner: Repository owner
            repo_name: Repository name
            author: GitHub username (commit author filter)
            limit: Maximum commits to fetch (default 100, max 100 per page)

        Returns:
            List of commit dictionaries with:
            - sha, commit.message, commit.author (name, email, date)
            - commit.committer, author.login
            - stats (additions, deletions, total)

        Raises:
            httpx.HTTPError: On API request failure
        """
        rest_url = f"{self.REST_API_BASE}/repos/{owner}/{repo_name}/commits"
        params = {
            "author": author,
            "per_page": min(limit, 100)
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                rest_url,
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            commits = response.json()

            return commits if isinstance(commits, list) else []

    async def get_commit_details(
        self,
        owner: str,
        repo_name: str,
        commit_sha: str
    ) -> Dict[str, Any]:
        """
        Fetch detailed commit information including file diffs via REST API.

        Used for deep analysis of sampled commits.

        Args:
            owner: Repository owner
            repo_name: Repository name
            commit_sha: Commit SHA hash

        Returns:
            Commit details with:
            - sha, commit.message, commit.author
            - stats (additions, deletions, total)
            - files[] (filename, status, additions, deletions, changes, patch)

        Raises:
            httpx.HTTPError: On API request failure
        """
        rest_url = f"{self.REST_API_BASE}/repos/{owner}/{repo_name}/commits/{commit_sha}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(rest_url, headers=self.headers)
            response.raise_for_status()
            return response.json()

    async def get_repository_readme(
        self,
        owner: str,
        repo_name: str
    ) -> Optional[str]:
        """
        Fetch repository README content via REST API.

        Args:
            owner: Repository owner
            repo_name: Repository name

        Returns:
            README content as markdown text, or None if not found

        Raises:
            httpx.HTTPError: On non-404 API errors
        """
        rest_url = f"{self.REST_API_BASE}/repos/{owner}/{repo_name}/readme"

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.get(rest_url, headers=self.headers)
                response.raise_for_status()

                # GitHub returns base64-encoded content
                content_b64 = response.json().get("content", "")
                if content_b64:
                    content = base64.b64decode(content_b64).decode("utf-8")
                    return content
                return None

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning(f"No README found for {owner}/{repo_name}")
                    return None
                raise

    async def get_repository_languages(
        self,
        owner: str,
        repo_name: str
    ) -> Dict[str, int]:
        """
        Fetch repository language statistics via REST API.

        Args:
            owner: Repository owner
            repo_name: Repository name

        Returns:
            Dictionary mapping language names to byte counts
            Example: {"Python": 52341, "JavaScript": 12453}

        Raises:
            httpx.HTTPError: On API request failure
        """
        rest_url = f"{self.REST_API_BASE}/repos/{owner}/{repo_name}/languages"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(rest_url, headers=self.headers)
            response.raise_for_status()
            return response.json()

    async def get_repository_contents(
        self,
        owner: str,
        repo_name: str,
        path: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Fetch repository contents at a specific path via REST API.

        Useful for detecting dependency files (package.json, requirements.txt, etc.)

        Args:
            owner: Repository owner
            repo_name: Repository name
            path: Path within repository (default: root)

        Returns:
            List of file/directory objects with:
            - name, path, type (file/dir)
            - size, sha, url, download_url

        Raises:
            httpx.HTTPError: On API request failure
        """
        rest_url = f"{self.REST_API_BASE}/repos/{owner}/{repo_name}/contents/{path}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.get(rest_url, headers=self.headers)
                response.raise_for_status()
                contents = response.json()

                # Ensure it's a list (directory) not a single file
                if isinstance(contents, dict):
                    return [contents]
                return contents if isinstance(contents, list) else []

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning(f"Path '{path}' not found in {owner}/{repo_name}")
                    return []
                raise

    async def get_file_content(
        self,
        owner: str,
        repo_name: str,
        file_path: str
    ) -> Optional[str]:
        """
        Fetch specific file content via REST API.

        Used for reading dependency files (package.json, requirements.txt, etc.)

        Args:
            owner: Repository owner
            repo_name: Repository name
            file_path: Path to file

        Returns:
            File content as string, or None if not found

        Raises:
            httpx.HTTPError: On non-404 API errors
        """
        rest_url = f"{self.REST_API_BASE}/repos/{owner}/{repo_name}/contents/{file_path}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.get(rest_url, headers=self.headers)
                response.raise_for_status()

                file_data = response.json()

                # Decode base64 content
                if "content" in file_data:
                    content_b64 = file_data["content"]
                    content = base64.b64decode(content_b64).decode("utf-8")
                    return content

                return None

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.debug(f"File '{file_path}' not found in {owner}/{repo_name}")
                    return None
                raise
            except Exception as e:
                logger.error(f"Error decoding file content: {e}")
                return None

    async def check_rate_limit(self) -> Dict[str, Any]:
        """
        Check GitHub API rate limit status.

        Returns:
            Dictionary with rate limit info:
            - resources.core.limit (max requests per hour)
            - resources.core.remaining (requests left)
            - resources.core.reset (timestamp when limit resets)

        Raises:
            httpx.HTTPError: On API request failure
        """
        rest_url = f"{self.REST_API_BASE}/rate_limit"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(rest_url, headers=self.headers)
            response.raise_for_status()
            return response.json()


# ============================================================
# Utility Functions
# ============================================================

def parse_language_percentages(languages_data: Dict) -> Dict[str, float]:
    """
    Parse GraphQL languages response into percentages.

    Args:
        languages_data: GraphQL languages object with edges and totalSize

    Returns:
        Dictionary mapping language names to percentages
        Example: {"Python": 65.5, "JavaScript": 30.2, "HTML": 4.3}
    """
    if not languages_data or not languages_data.get("edges"):
        return {}

    total_size = languages_data.get("totalSize", 0)
    if total_size == 0:
        return {}

    percentages = {}
    for edge in languages_data["edges"]:
        lang_name = edge["node"]["name"]
        lang_size = edge["size"]
        percentage = (lang_size / total_size * 100) if total_size > 0 else 0
        percentages[lang_name] = round(percentage, 2)

    return percentages


def extract_commit_metadata(commit_data: Dict) -> Dict[str, Any]:
    """
    Extract standardized metadata from commit response.

    Args:
        commit_data: Commit object from GitHub API

    Returns:
        Standardized commit metadata dictionary
    """
    commit_info = commit_data.get("commit", {})
    author_info = commit_info.get("author", {})

    return {
        "sha": commit_data.get("sha"),
        "message": commit_info.get("message"),
        "author_name": author_info.get("name"),
        "author_email": author_info.get("email"),
        "author_date": author_info.get("date"),
        "committer_name": commit_info.get("committer", {}).get("name"),
        "committer_email": commit_info.get("committer", {}).get("email"),
        "committer_date": commit_info.get("committer", {}).get("date"),
    }
