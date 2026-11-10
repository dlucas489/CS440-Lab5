#!/bin/bash
# context-switcher.sh
# Dynamically loads relevant context based on git branch and user query safely.

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
CLAUDE_MD="$PROJECT_DIR/CLAUDE.md"
BRANCH=$(git -C "$PROJECT_DIR" branch --show-current 2>/dev/null)
PROMPT=$(jq -r '.prompt // ""')

touch "$CLAUDE_MD"

add_context() {
    local context_ref="$1"
    grep -qxF "$context_ref" "$CLAUDE_MD" || echo "$context_ref" >> "$CLAUDE_MD"
}

# Branch-based context
case $BRANCH in
  "feature/qlearning"*)
    add_context "@./context/qlearning-math.md"
    ;;
esac
