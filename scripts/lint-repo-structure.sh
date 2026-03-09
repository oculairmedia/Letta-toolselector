#!/usr/bin/env bash
# lint-repo-structure.sh — Validate repo structure conventions.
# Used by: CI workflow (.github/workflows/repo-lint.yml)
#          Pre-commit hook (.git/hooks/pre-commit)
#
# Rules:
#   1. No new markdown files at root (only README.md, AGENTS.md allowed)
#   2. No generated artifacts tracked (coverage, test results, caches)
#   3. No new Python scripts at root (only whitelisted shared modules)
#   4. No duplicate env entries in compose.yaml
#
# Exit 0 = clean, Exit 1 = violations found

set -euo pipefail

ERRORS=()

# --- Configurable whitelist ---
# Markdown files allowed at root
ALLOWED_ROOT_MD=(
  "README.md"
  "AGENTS.md"
)

# Python files allowed at root (shared modules imported by services)
ALLOWED_ROOT_PY=(
  "letta_tool_utils.py"
  "tool_selector_client.py"
  "qwen3_reranker_utils.py"
  "ollama_reranker_adapter.py"
  "find_tools.py"
  "find_tools_enhanced.py"
  "pytest.ini"
)

# Generated artifacts that must never be tracked
FORBIDDEN_TRACKED=(
  ".coverage"
  "coverage.xml"
  "htmlcov/"
  ".benchmarks/"
  ".pytest_cache/"
)

# --- Determine files to check ---
# If called with --staged (pre-commit), only check staged files.
# Otherwise check all tracked files (CI).
if [[ "${1:-}" == "--staged" ]]; then
  FILES=$(git diff --cached --name-only --diff-filter=ACR 2>/dev/null || true)
else
  FILES=$(git ls-files 2>/dev/null || true)
fi

if [[ -z "$FILES" ]]; then
  exit 0
fi

# --- Rule 1: No new root markdown files ---
root_md=$(echo "$FILES" | grep -E '^[^/]+\.md$' || true)
for f in $root_md; do
  basename_f=$(basename "$f")
  allowed=false
  for a in "${ALLOWED_ROOT_MD[@]}"; do
    if [[ "$basename_f" == "$a" ]]; then
      allowed=true
      break
    fi
  done
  if [[ "$allowed" == "false" ]]; then
    ERRORS+=("❌ Root markdown not allowed: $f → move to docs/ or docs/archive/")
  fi
done

# --- Rule 2: No generated artifacts tracked ---
for pattern in "${FORBIDDEN_TRACKED[@]}"; do
  matches=$(echo "$FILES" | grep -E "^${pattern}" || true)
  if [[ -n "$matches" ]]; then
    while IFS= read -r m; do
      ERRORS+=("❌ Generated artifact tracked: $m → add to .gitignore and git rm --cached")
    done <<< "$matches"
  fi
done

# --- Rule 3: No new root Python scripts ---
root_py=$(echo "$FILES" | grep -E '^[^/]+\.py$' || true)
for f in $root_py; do
  basename_f=$(basename "$f")
  allowed=false
  for a in "${ALLOWED_ROOT_PY[@]}"; do
    if [[ "$basename_f" == "$a" ]]; then
      allowed=true
      break
    fi
  done
  if [[ "$allowed" == "false" ]]; then
    ERRORS+=("❌ Root Python script not allowed: $f → move to scripts/ or tests/")
  fi
done

# --- Rule 4: No duplicate env entries within same service in compose.yaml ---
if [[ -f "compose.yaml" ]] && command -v python3 >/dev/null 2>&1; then
  dupes=$(python3 -c "
import yaml, sys
try:
    data = yaml.safe_load(open('compose.yaml'))
    for svc, cfg in (data.get('services') or {}).items():
        env = cfg.get('environment') or []
        if isinstance(env, list):
            keys = [e.split('=')[0].strip().lstrip('- ') for e in env if '=' in str(e)]
            seen = set()
            for k in keys:
                if k in seen:
                    print(f'{svc}:{k}')
                seen.add(k)
except Exception:
    pass
" 2>/dev/null || true)
  if [[ -n "$dupes" ]]; then
    while IFS= read -r entry; do
      svc=$(echo "$entry" | cut -d: -f1)
      key=$(echo "$entry" | cut -d: -f2)
      ERRORS+=("❌ Duplicate env var in compose.yaml service '$svc': $key")
    done <<< "$dupes"
  fi
fi

# --- Report ---
if [[ ${#ERRORS[@]} -gt 0 ]]; then
  echo ""
  echo "=== Repo Structure Lint: ${#ERRORS[@]} violation(s) found ==="
  echo ""
  for err in "${ERRORS[@]}"; do
    echo "  $err"
  done
  echo ""
  echo "Fix these issues before committing/pushing."
  echo "See scripts/lint-repo-structure.sh for rules and whitelists."
  echo ""
  exit 1
fi

echo "✅ Repo structure lint passed"
exit 0
