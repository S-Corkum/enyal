# Enyal Usage Examples

This guide provides practical workflows and examples for using Enyal with AI coding assistants.

## Table of Contents

1. [Project Conventions Workflow](#project-conventions-workflow)
2. [Architecture Decision Records](#architecture-decision-records)
3. [Code Review Context](#code-review-context)
4. [Multi-Project Knowledge Base](#multi-project-knowledge-base)
5. [Scope-Aware Search](#scope-aware-search)
6. [Duplicate Detection](#duplicate-detection)
7. [Entry Management](#entry-management)
8. [Team Onboarding](#team-onboarding)
9. [Debugging Context](#debugging-context)
10. [API Documentation Context](#api-documentation-context)
11. [Testing Strategy Context](#testing-strategy-context)

---

## Project Conventions Workflow

Store and recall coding conventions to ensure consistency across your codebase.

### Storing Conventions

**Via AI Assistant:**

    You: Remember these project conventions:
         - Use camelCase for JavaScript variables
         - Use snake_case for Python variables  
         - All API endpoints return JSON
         - Error responses use {error: string, code: number} format

    Assistant: [Calls enyal_remember for each convention]
    I have stored 4 conventions for this project:
    1. JavaScript variable naming (camelCase)
    2. Python variable naming (snake_case)
    3. API response format (JSON)
    4. Error response structure

**Via CLI:**

    enyal remember "Use camelCase for JavaScript variables" --type convention --tags javascript,naming
    enyal remember "Use snake_case for Python variables" --type convention --tags python,naming
    enyal remember "All API endpoints return JSON" --type convention --tags api,format
    enyal remember "Error responses use {error: string, code: number}" --type convention --tags api,errors

### Recalling Conventions

    You: How should I name variables in this JavaScript file?

    Assistant: [Calls enyal_recall with query "javascript variable naming"]
    Based on your project conventions, you should use camelCase for JavaScript variables.
    For example: `userName`, `totalCount`, `isActive`

### Updating Conventions

Add a new convention:

    You: Actually, let's use PascalCase for React components

    Assistant: [Calls enyal_remember]
    I have added a new convention: Use PascalCase for React component names.
    This complements your existing camelCase convention for regular variables.

Modify an existing convention:

    You: Update the JavaScript naming convention to specify it's for variables and functions

    Assistant: [Calls enyal_recall to find the entry, then enyal_update]
    I have updated the existing convention. It now reads:
    "Use camelCase for JavaScript variables and functions"

---

## Architecture Decision Records

Track important technical decisions and their rationale.

### Recording Decisions

    You: We decided to use PostgreSQL instead of MongoDB for the database.
         The reasons were:
         - Better support for complex queries
         - ACID compliance needed for financial data
         - Team has more SQL experience

    Assistant: [Calls enyal_remember with type=decision]
    I have recorded this architecture decision about choosing PostgreSQL over MongoDB,
    including the rationale about query support, ACID compliance, and team experience.

### Recalling Past Decisions

    You: Why did we choose PostgreSQL for this project?

    Assistant: [Calls enyal_recall with query "database choice postgresql"]
    According to your recorded decisions, PostgreSQL was chosen over MongoDB for these reasons:
    1. Better support for complex queries
    2. ACID compliance needed for financial data
    3. Team has more SQL experience

### Reviewing All Decisions

**Via CLI:**

    enyal recall "architecture decision" --type decision --limit 20

---

## Code Review Context

Provide context during code reviews to ensure consistency.

### Before Review

    You: I am about to review a pull request. What conventions should I check for?

    Assistant: [Calls enyal_recall with query "code conventions review"]
    Based on your project conventions, here are key things to check:
    
    Naming:
    - JavaScript: camelCase for variables
    - Python: snake_case for variables
    - React: PascalCase for components
    
    API:
    - All endpoints return JSON
    - Error format: {error: string, code: number}
    
    Testing:
    - All new features need unit tests
    - Use pytest for Python tests

### During Review

    You: Is this error response format correct?
         return {"message": "Not found", "status": 404}

    Assistant: [Calls enyal_recall with query "error response format"]
    This does not match your project convention. The expected format is:
    {error: string, code: number}
    
    The correct format would be:
    return {"error": "Not found", "code": 404}

---

## Multi-Project Knowledge Base

Organize knowledge across multiple projects using scopes.

### Global Knowledge (Applies Everywhere)

    You: Remember globally that I prefer 4-space indentation

    Assistant: [Calls enyal_remember with scope=global]
    I have stored your global preference for 4-space indentation.
    This will apply across all your projects.

**Via CLI:**

    enyal remember "Prefer 4-space indentation" --type preference --scope global

### Workspace Knowledge (Related Projects)

    You: For all projects in my microservices workspace, we use gRPC for inter-service communication

    Assistant: [Calls enyal_remember with scope=workspace]
    I have stored the workspace convention about using gRPC for inter-service communication.

**Via CLI:**

    enyal remember "Use gRPC for inter-service communication" \
      --type convention \
      --scope workspace \
      --scope-path /Users/dev/microservices

### Project-Specific Knowledge

    You: This specific project uses REST instead of gRPC because it is customer-facing

    Assistant: [Calls enyal_remember with scope=project]
    I have stored the project-specific decision to use REST for this customer-facing service.
    Note: This overrides the workspace-level gRPC convention for this project only.

### Querying Across Scopes

When you ask a question, Enyal automatically searches from most specific to most general:

    You: What communication protocol should I use for this service?

    Assistant: [Calls enyal_recall - searches file -> project -> workspace -> global]
    For this project, you should use REST because it is customer-facing.
    (Note: Other projects in your microservices workspace use gRPC)

---

## Scope-Aware Search

Automatically search context relevant to your current file location.

### Automatic Context Resolution

The `enyal_recall_by_scope` tool automatically determines applicable scopes based on your current file:

    You: [Working in /Users/dev/myapp/src/api/auth.py]
        What conventions apply to this file?

    Assistant: [Calls enyal_recall_by_scope with file_path="/Users/dev/myapp/src/api/auth.py"]
    Based on context from all applicable scopes:

    From file scope (/Users/dev/myapp/src/api/auth.py):
    - This file handles JWT token validation

    From project scope (/Users/dev/myapp):
    - Use PostgreSQL for user data
    - All API endpoints return JSON

    From workspace scope (/Users/dev):
    - Use gRPC for inter-service communication

    From global scope:
    - Prefer 4-space indentation

### Scope Priority

Results are weighted by specificity—file-level context ranks higher than global:

    You: [Working in /Users/dev/myapp/src/api/users.py]
        What database should I use?

    Assistant: [Calls enyal_recall_by_scope with query "database"]
    For this file, you should use PostgreSQL.

    This comes from your project-level decision (higher priority than
    any global preferences).

### IDE Integration

When your AI assistant knows your current file, it can automatically use scope-aware search:

    You: How should I handle errors here?

    Assistant: [Calls enyal_recall_by_scope with current file context]
    Based on conventions for this project:
    - Return {error: string, code: number} format
    - Log errors to the centralized logging service
    - Use HTTP 4xx for client errors, 5xx for server errors

---

## Duplicate Detection

Prevent storing redundant or near-duplicate context.

### Basic Duplicate Detection

Enable duplicate checking when storing new context:

    You: Remember that we use pytest for testing

    Assistant: [Calls enyal_remember with check_duplicate=true]
    I found an existing similar entry (92% match):
    "Always use pytest for testing in this project"

    I have not stored the duplicate. The existing entry covers this.

### Duplicate Actions

Control what happens when a duplicate is detected:

**Reject (default):** Don't store, return existing entry

    You: Remember that we use 4-space indentation

    Assistant: [Calls enyal_remember with check_duplicate=true, on_duplicate="reject"]
    This is already stored (89% match). Keeping the existing entry:
    "Use 4-space indentation for all Python files"

**Merge:** Update existing entry with new content

    You: Remember that we use pytest with coverage reporting

    Assistant: [Calls enyal_remember with check_duplicate=true, on_duplicate="merge"]
    I found a similar entry about pytest (87% match). I have merged your
    new information into the existing entry, which now reads:
    "Use pytest with coverage reporting for all testing"

**Store:** Store as new entry despite similarity

    You: Remember that the auth service also uses pytest (store even if similar)

    Assistant: [Calls enyal_remember with check_duplicate=true, on_duplicate="store"]
    I found a similar entry (85% match) but stored this as a separate entry
    as requested, since it's specific to the auth service.

### Adjusting Sensitivity

Control the similarity threshold (default 0.85 = 85%):

**Via Python:**

```python
# Strict matching (90% similarity required)
store.remember(
    content="Use black for formatting",
    check_duplicate=True,
    duplicate_threshold=0.90,
    on_duplicate="reject"
)

# Loose matching (70% similarity triggers duplicate detection)
store.remember(
    content="Format code with black",
    check_duplicate=True,
    duplicate_threshold=0.70,
    on_duplicate="merge"
)
```

### Batch Import Deduplication

When importing multiple entries, use duplicate detection to avoid redundancy:

**Via Python:**

```python
conventions = [
    "Use pytest for testing",
    "Use black for code formatting",
    "Use pytest for unit tests",  # Near-duplicate of first
    "Format Python with black",   # Near-duplicate of second
]

for conv in conventions:
    result = store.remember(
        content=conv,
        content_type="convention",
        check_duplicate=True,
        on_duplicate="reject"
    )
    if isinstance(result, dict) and result.get("action") == "existing":
        print(f"Skipped duplicate: {conv}")
```

---

## Entry Management

Retrieve, update, and manage individual context entries.

### Getting a Specific Entry

Retrieve full details of an entry by ID:

    You: Show me the details of that database decision we made

    Assistant: [Calls enyal_recall to find it, then enyal_get with the entry ID]
    Here are the full details:

    ID: abc123-def456
    Content: "Use PostgreSQL instead of MongoDB for better query support"
    Type: decision
    Scope: project
    Confidence: 0.95
    Created: 2026-01-10
    Updated: 2026-01-10
    Tags: database, postgresql, architecture
    Access Count: 12

### Updating Entry Content

Modify an existing entry's content (automatically regenerates embedding):

    You: Update that PostgreSQL decision to mention we're using version 15

    Assistant: [Calls enyal_update with new content]
    I have updated the entry. It now reads:
    "Use PostgreSQL 15 instead of MongoDB for better query support and JSONB features"

    The semantic embedding has been regenerated to reflect the new content.

### Adjusting Confidence

Increase or decrease confidence in stored context:

    You: I'm less sure about that caching decision now. Lower its confidence.

    Assistant: [Calls enyal_update with confidence=0.5]
    I have reduced the confidence of the Redis caching decision from 0.9 to 0.5.
    It will now rank lower in search results.

### Updating Tags

Refine categorization of entries:

    You: Add the "performance" tag to all our caching-related entries

    Assistant: [Calls enyal_recall to find caching entries, then enyal_update for each]
    I have added the "performance" tag to 3 entries:
    1. Redis caching decision
    2. Cache invalidation pattern
    3. CDN configuration convention

### Entry Lifecycle Example

Complete workflow from creation to deprecation:

```python
# 1. Create entry
entry_id = store.remember(
    content="Use Redis for session storage",
    content_type="decision",
    tags=["redis", "sessions"]
)

# 2. Retrieve entry
entry = store.get(entry_id)
print(f"Confidence: {entry.confidence}")

# 3. Update entry
store.update(
    entry_id=entry_id,
    content="Use Redis 7.0 for session storage with TLS",
    tags=["redis", "sessions", "security"]
)

# 4. Deprecate when outdated
store.forget(entry_id)  # Soft delete - can be restored

# 5. Permanently remove if needed
store.forget(entry_id, hard_delete=True)
```

---

## Team Onboarding

Capture institutional knowledge to help new team members.

### Documenting Tribal Knowledge

    You: Remember these things new developers should know:
         - The CI pipeline runs on every PR
         - Feature branches should be named feature/TICKET-description
         - Code owners must approve changes to /core directory
         - Database migrations require DBA review

    Assistant: [Calls enyal_remember multiple times]
    I have stored 4 pieces of onboarding information:
    1. CI pipeline process
    2. Branch naming convention
    3. Code owners policy
    4. Database migration review requirement

### New Developer Questions

    You: I am new to this project. How do I name my feature branch?

    Assistant: [Calls enyal_recall with query "branch naming"]
    According to project conventions, feature branches should be named:
    feature/TICKET-description
    
    For example: feature/PROJ-123-add-user-auth

### Export for Documentation

**Via CLI:**

    # Export all onboarding context
    enyal recall "onboarding OR new developer" --limit 50 --json > onboarding.json

---

## Debugging Context

Store and recall debugging insights for recurring issues.

### Recording Bug Patterns

    You: Remember that the "connection timeout" error usually means Redis is not running

    Assistant: [Calls enyal_remember with type=pattern]
    I have stored this debugging pattern: connection timeout errors typically indicate
    Redis is not running.

### Recalling During Debugging

    You: I am seeing "connection timeout" errors in the logs

    Assistant: [Calls enyal_recall with query "connection timeout"]
    Based on recorded patterns, this error usually means Redis is not running.
    
    Try these steps:
    1. Check if Redis is running: redis-cli ping
    2. If not, start it: redis-server
    3. Check the connection settings in your config

### Building a Debugging Knowledge Base

**Via CLI:**

    # Common error patterns
    enyal remember "ECONNREFUSED on port 5432 means PostgreSQL is down" --type pattern --tags debugging,postgresql
    enyal remember "JWT decode error usually means token expired" --type pattern --tags debugging,auth
    enyal remember "CORS errors in browser need backend header configuration" --type pattern --tags debugging,cors

---

## API Documentation Context

Store API-specific knowledge for consistent development.

### Recording API Contracts

    You: Remember these API conventions:
         - All endpoints are versioned: /api/v1/...
         - Authentication uses Bearer tokens
         - Rate limit is 100 requests per minute
         - Pagination uses cursor-based pagination

    Assistant: [Calls enyal_remember for each]
    I have stored 4 API conventions covering versioning, authentication,
    rate limiting, and pagination.

### Implementing New Endpoints

    You: I need to create a new endpoint for user profiles

    Assistant: [Calls enyal_recall with query "API conventions endpoint"]
    Based on your API conventions, your endpoint should:
    
    1. URL: /api/v1/users/profile (versioned)
    2. Auth: Require Bearer token in Authorization header
    3. Response: JSON format
    4. Pagination: Use cursor-based pagination if returning lists
    
    Example implementation:
    @app.get("/api/v1/users/profile")
    async def get_profile(authorization: str = Header(...)):
        # Validate Bearer token
        ...

---

## Testing Strategy Context

Maintain consistent testing practices across the project.

### Recording Testing Conventions

    You: Remember our testing strategy:
         - Unit tests for all business logic
         - Integration tests for API endpoints
         - E2E tests for critical user flows
         - Minimum 80% code coverage
         - Use pytest fixtures for database setup

    Assistant: [Calls enyal_remember for each]
    I have stored 5 testing conventions covering unit tests, integration tests,
    E2E tests, coverage requirements, and pytest fixtures.

### Writing Tests

    You: I just wrote a new service class. What tests do I need?

    Assistant: [Calls enyal_recall with query "testing requirements"]
    Based on your testing strategy, for a new service class you need:
    
    1. Unit tests for all business logic methods
    2. Use pytest fixtures for any database setup
    3. Aim for 80% code coverage minimum
    
    If this service has API endpoints, also add:
    4. Integration tests for the API endpoints

### Test Organization

    You: Where should I put the tests for my new UserService?

    Assistant: [Calls enyal_recall with query "test file organization"]
    Following your project conventions:
    - Unit tests: tests/unit/test_user_service.py
    - Integration tests: tests/integration/test_user_api.py
    - Use pytest fixtures from tests/conftest.py

---

## Tips for Effective Usage

### Be Specific When Storing

Good:

    "Use pytest with pytest-asyncio for async tests in Python files"

Less useful:

    "Use pytest for tests"

### Use Consistent Tags

Develop a tagging taxonomy:
- Language: `python`, `javascript`, `typescript`
- Domain: `api`, `database`, `auth`, `frontend`
- Type: `naming`, `testing`, `deployment`, `debugging`

### Review and Update Regularly

    You: Show me all conventions we have stored

    Assistant: [Calls enyal_recall with broad query]
    Here are your stored conventions...

    You: The one about tabs is outdated, we use spaces now.

    Assistant: Would you like me to update it or remove it?

    You: Update it to say we use 4 spaces

    Assistant: [Calls enyal_update]
    I have updated the indentation convention to: "Use 4-space indentation"

Or remove outdated entries entirely:

    You: Actually, just forget the old formatting convention

    Assistant: [Calls enyal_forget]
    I have deprecated the outdated convention. It won't appear in searches
    but can be restored if needed.

### Scope Appropriately

| Scope | Use For |
|-------|---------|
| Global | Personal preferences, universal standards |
| Workspace | Shared conventions across related projects |
| Project | Project-specific decisions and patterns |
| File | File-specific notes (rarely needed) |

### Combine with CLAUDE.md

Enyal complements static CLAUDE.md files:
- **CLAUDE.md**: Static, version-controlled project instructions
- **Enyal**: Dynamic, accumulated knowledge that grows over time

Example workflow:
1. Core conventions go in CLAUDE.md
2. Decisions, patterns, and evolving knowledge go in Enyal
3. Periodically promote stable Enyal knowledge to CLAUDE.md
