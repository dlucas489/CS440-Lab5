# Lab 4: Vacuum Robot Q-Learning & Context Engineering

**Video Demonstration:** [Insert YouTube/Loom Link Here]

## Context Engineering Strategies Used
1. **Strategy 1 (Persistent Memory / Write):** Located in the `strategy_1_persistent_memory` directory. Uses a Git hook and shell script to dynamically inject context without bloating the context window.
2. **Strategy 4 (Isolation via Sub-agents):** Located in the `strategy_4_subagents` directory. Uses a Google Gemini LLM as an isolated sub-agent to handle hyperparameter tuning (epsilon decay) without cluttering the main navigation loop's context.

## How to Run
1. Navigate to `strategy_4_subagents`.
2. Ensure `vacuum_local_search.py` is present.
3. Export your Gemini API key: `export GEMINI_API_KEY="your_key"`
4. Run: `python q_learning_agent.py`
