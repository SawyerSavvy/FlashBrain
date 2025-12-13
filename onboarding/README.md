# GitHub Onboarding Agent

A specialized ReAct agent for analyzing GitHub developer profiles and creating comprehensive skill assessments for freelancer matching.

## Overview

The GitHub Onboarding Agent extracts and analyzes developer activity from GitHub to create detailed profiles that help match freelancers with appropriate projects. It uses an **Extract-First Architecture** to separate fast data extraction from expensive LLM analysis.

### Key Features

- 📊 **Comprehensive GitHub Analysis**: Extracts profile, repositories, and commits
- 🧠 **LLM-Powered Insights**: Classifies commit patterns and generates developer stories
- 📈 **Scoring System**: Calculates consistency, domain expertise, and impact scores
- 💾 **Dual Storage**: Structured data in Supabase + vector embeddings in RAG service
- 🔄 **A2A Protocol**: Integrates with FlashBrain's multi-agent system
- ⚡ **Cost-Efficient**: Separates cheap GitHub API calls from expensive LLM analysis

## Architecture

### Extract-First Approach

The agent uses a two-phase architecture to optimize cost and performance:

**Phase 1: Data Extraction** (Fast & Cheap)
```
GitHub API → Supabase Tables
```
- Fetches ALL profile, repository, and commit data
- Samples ~20 commits per repo for deep analysis
- No LLM calls - just HTTP requests and database writes
- Returns `profile_id` for analysis phase

**Phase 2: LLM Analysis** (Smart & Targeted)
```
Supabase Tables → LLM Analysis → Updated Profile
```
- Analyzes stored data to classify commits
- Calculates consistency, expertise, and impact scores
- Generates developer story narrative
- Updates profile with insights

### Tools (7 Total)

#### Extraction Tools (No LLM)
1. **extract_github_data_to_database** - Fetches and stores ALL GitHub data
2. **store_readme_in_rag** - Vectorizes repository READMEs
3. **get_profile_summary** - Retrieves stored data for analysis

#### Analysis Tools (With LLM)
4. **analyze_commit_patterns** - Classifies commits and calculates consistency score
5. **calculate_domain_expertise** - Maps technical domains from file changes
6. **calculate_impact_score** - Weights contributions by complexity and popularity
7. **generate_developer_story** - Creates compelling narrative summary

## Database Schema

### freelancer_github_profiles
Primary profile data with calculated scores:
- Basic info: username, name, bio, location
- Stats: public repos, followers, stars
- Scores: consistency, quality, impact (0-100)
- Tech stack (JSONB array)
- Domain expertise (JSONB object)
- Developer story (LLM-generated narrative)

### freelancer_repositories
Repository metadata per developer:
- Repository details and metrics
- Language breakdown (JSONB)
- Technologies/frameworks detected
- Commit counts (total and by user)

### freelancer_commits
Sampled commits for deep analysis:
- Commit metadata (SHA, message, date)
- File changes (JSONB)
- LLM classifications (type, complexity)
- Sampling flag for cost control

## Installation

### 1. Clone and Navigate

```bash
cd /onboarding
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase admin key
- `SUPABASE_POOLER` - PostgreSQL connection string (optional, for checkpointing)
- `GITHUB_TOKEN` - GitHub Personal Access Token ([create here](https://github.com/settings/tokens))
- `GOOGLE_API_KEY` - Google AI API key ([get here](https://makersuite.google.com/app/apikey))
- `RAG_SERVICE_URL` - RAG service endpoint (optional, for README vectorization)

### 4. Run Database Migrations

Execute the SQL migration in Supabase:

```bash
psql $SUPABASE_POOLER < migrations/001_freelancer_profiles.sql
```

Or use the Supabase SQL Editor to run `migrations/001_freelancer_profiles.sql`.

## Usage

### Local Development

Start the agent server:

```bash
python __main__.py --host localhost --port 8013
```

The agent will be available at `http://localhost:8013`.

### Testing

Run the test suite:

```bash
# Interactive mode
python test_onboarding.py

# Test specific GitHub user
python test_onboarding.py --username octocat
```

### Example Workflow

**User Request:**
```
Analyze GitHub developer @torvalds
```

**Agent Actions:**
1. Calls `extract_github_data_to_database(github_username="torvalds")`
   - Fetches profile, repos, commits from GitHub
   - Stores in 3 Supabase tables
   - Returns profile_id

2. Calls `analyze_commit_patterns(profile_id="...")`
   - Classifies commit types with LLM
   - Calculates consistency score

3. Calls `calculate_domain_expertise(profile_id="...")`
   - Maps file extensions to technical domains
   - Returns percentage distribution

4. Calls `calculate_impact_score(profile_id="...")`
   - Weighs by complexity, stars, volume
   - Returns 0-100 score

5. Calls `generate_developer_story(profile_id="...")`
   - LLM creates narrative summary
   - Updates profile

**Agent Response:**
```
✅ Analysis complete for Linus Torvalds

Profile: @torvalds
- Consistency: 92/100 (extremely active, daily commits)
- Code Quality: 95/100 (high-quality, well-documented)
- Impact: 98/100 (Linux kernel, 200k+ stars)
- Top Skills: C, Shell, Makefile, Git
- Domain Expertise: 85% systems/kernel, 10% tools, 5% DevOps

Developer Story: Linus is the creator and principal maintainer of the Linux kernel...
```

## Deployment

### Cloud Run Deployment

1. Build and deploy using Cloud Build:

```bash
gcloud builds submit --config cloudbuild.yaml
```

2. Or deploy manually:

```bash
# Build image
docker build -t gcr.io/PROJECT_ID/github-onboarding-agent .

# Push to registry
docker push gcr.io/PROJECT_ID/github-onboarding-agent

# Deploy to Cloud Run
gcloud run deploy github-onboarding-agent \
  --image gcr.io/PROJECT_ID/github-onboarding-agent \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 5 \
  --memory 1Gi \
  --cpu 1 \
  --timeout 600 \
  --set-env-vars SUPABASE_URL=...,GITHUB_TOKEN=...
```

3. Set environment variables in Cloud Run console or via:

```bash
gcloud run services update github-onboarding-agent \
  --region us-central1 \
  --set-env-vars "SUPABASE_URL=...,SUPABASE_SERVICE_ROLE_KEY=...,GITHUB_TOKEN=...,GOOGLE_API_KEY=..."
```

### Integration with FlashBrain

Add the onboarding agent URL to FlashBrain's available agents:

```python
# In FlashBrain Orchestrator
AGENTS_URLS = [
    "https://project-decomp-agent-xxx.run.app",
    "https://select-freelancer-agent-xxx.run.app",
    "https://github-onboarding-agent-xxx.run.app",  # NEW
]
```

FlashBrain can then hand off to the onboarding agent when users request developer analysis.

## API Reference

### A2A Protocol

The agent exposes an A2A protocol endpoint that accepts:

**Input:**
```json
{
  "message": {
    "parts": [{"text": "Analyze @octocat"}],
    "metadata": {
      "github_username": "octocat",  // Optional
      "profile_id": "abc-123",       // Optional
      "client_id": "client-xyz"      // Optional
    }
  }
}
```

**Output:**
```json
{
  "task": {
    "state": "completed",
    "artifacts": [{
      "name": "onboarding_analysis",
      "parts": [{"text": "Analysis results..."}]
    }]
  }
}
```

### Tool Reference

#### extract_github_data_to_database
```python
extract_github_data_to_database(
    github_username: str,
    client_id: Optional[str] = None,
    max_repos: int = 50,
    commits_per_repo: int = 100,
    sample_size: int = 20
) -> str
```

#### analyze_commit_patterns
```python
analyze_commit_patterns(profile_id: str) -> str
```

#### calculate_domain_expertise
```python
calculate_domain_expertise(profile_id: str) -> str
```

#### calculate_impact_score
```python
calculate_impact_score(profile_id: str) -> str
```

#### generate_developer_story
```python
generate_developer_story(profile_id: str) -> str
```

## Configuration

### GitHub Token Scopes

Your GitHub Personal Access Token needs these scopes:
- `repo` - Access public and private repositories
- `read:user` - Read user profile data
- `read:org` - Read organization data (optional)

### Supabase Tables

The agent requires 3 tables in Supabase:
- `freelancer_github_profiles` - Developer profiles
- `freelancer_repositories` - Repository metadata
- `freelancer_commits` - Commit samples

See `migrations/001_freelancer_profiles.sql` for schema.

### LLM Configuration

The agent uses **gemini-2.0-flash-exp** with:
- Temperature: 0.1 (deterministic)
- Used for: commit classification, story generation
- Not used for: API calls, data extraction, simple aggregations

## Cost Optimization

### Extract-First Strategy

1. **Cheap Data Extraction** (~$0.01 per developer)
   - GitHub API calls (free with token)
   - Database writes (minimal cost)
   - Stores ALL data once

2. **Targeted LLM Analysis** (~$0.05-$0.10 per developer)
   - Only analyzes sampled commits (~20 per repo)
   - Reads from database, not live API
   - Can re-analyze without re-fetching

### Sampling Strategy

- Fetches ALL commit metadata (cheap)
- Samples ~20 commits per repo for LLM analysis
- Evenly distributed sampling across commit history
- Marked with `is_sampled` flag for easy filtering

## Troubleshooting

### GitHub API Rate Limits

```
Error: API rate limit exceeded
```

**Solution**: Use a Personal Access Token with higher rate limits (5000/hour vs 60/hour).

### Missing Environment Variables

```
Error: Missing required environment variables
```

**Solution**: Ensure `.env` has all required variables. Check with:
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print([v for v in ['SUPABASE_URL', 'GITHUB_TOKEN', 'GOOGLE_API_KEY'] if not os.getenv(v)])"
```

### Checkpointer Fails

```
Warning: Failed to initialize checkpointer
```

**Solution**: This is non-critical. Checkpointing requires `SUPABASE_POOLER`. Set it or ignore if not using conversation state.

### README Vectorization Fails

```
Error: Failed to store README in RAG
```

**Solution**: Ensure `RAG_SERVICE_URL` is set and the RAG service is running. This is optional - agent works without it.

## Development

### Project Structure

```
onboarding/
├── __main__.py              # Server entry point
├── onboarding_agent.py      # ReAct agent core
├── agent_executor.py        # A2A wrapper
├── github_client.py         # GitHub API client
├── extraction_tools.py      # Data extraction (3 tools)
├── analysis_tools.py        # LLM analysis (4 tools)
├── prompts.py              # System prompts
├── requirements.txt        # Dependencies
├── Dockerfile              # Container image
├── cloudbuild.yaml         # GCP deployment
├── .env.example            # Environment template
├── test_onboarding.py      # Test suite
├── README.md               # Documentation
└── migrations/
    └── 001_freelancer_profiles.sql  # Database schema
```

### Running Tests

```bash
# Unit tests (TODO: add pytest)
pytest test_onboarding.py

# Integration test
python test_onboarding.py --username octocat

# Interactive test
python test_onboarding.py
```

### Adding New Tools

1. Add tool function in `extraction_tools.py` or `analysis_tools.py`
2. Return tool from factory function
3. Tool will automatically be available to agent

## Roadmap

- [ ] Webhook integration for automatic profile updates
- [ ] Periodic refresh scheduler
- [ ] Code quality metrics (linting, test coverage)
- [ ] Collaboration patterns analysis
- [ ] Technology trend detection
- [ ] Export profiles to PDF/JSON

## License

Part of the FlashBrain multi-agent system.

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs: `gcloud run logs read github-onboarding-agent`
3. Open an issue in the FlashBrain repository

---

**Version**: 1.0.0
**Last Updated**: 2025-12-06
