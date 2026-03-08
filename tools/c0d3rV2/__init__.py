"""
c0d3r V2 — modular, OOP rewrite of the c0d3r AI engineering assistant.

Architecture (recursive three-step process flow):

  Step 1   User sends a request via the CLI input field.
  Step 2   Context is injected (6 layers):
             1. Local      — system info, cwd, time, weather.
             2. Memory     — rolling summary + 10 key points.
             3. Transcript — recent conversation (char-budget-limited).
             4. Tools      — ALL tool descriptions (always present).
             5. Accumulated — every tool output from the task tree
                              (the cross-tool feedback loop).
             6. Tree       — current task tree position and status.
  Step 3   Orchestrator reformulates the request in scientific / engineering
           vernacular, plans branches in a TaskTree, then executes each
           branch recursively.  Every AI call sees ALL tools + ALL
           accumulated results so tools feed back into each other.
  Step 3A  Each step has a self-regulatory validation loop.

Module map:
  c0d3rV2_cli          CLI entry point — argument parsing, dependency wiring.
  process_flow         Main coordinator (Steps 1-3, REPL loop).
  context_builder      Step 2: 6-layer context assembly.
  orchestrator         Step 3/3A: recursive agent loop with reformulation.
  task_tree            TaskNode + TaskTree for branch tracking.
  tool_registry        Tool base class, concrete wrappers, ToolRegistry.
  executor             Runs shell commands (pwsh / cmd / bash).
  petal_system         Dynamic steps that wilt when ineffective.
  web_search           Ethical DuckDuckGo search + AI summary.
  matrix_helpers       Equation matrix search (Kuzu-accelerated).
  vm_playground        Isolated VM experiment subsystem.
  hazy_hash            Context-anchored file-location memory.
  side_load_st_mem_file_location   Session-scoped file location memory (ST).
  side_load_lt_mem_file_location   Cross-session file location memory (LT).
  lt_mem               Full transcript store (JSONL).
  sessions             Factory wrapper for C0d3rSession.
  usage_tracker        Token / status tracking (data only).
  budget_tracker       API spend guard.
  header_renderer      Session header display.
  terminal_ui          Multi-backend TUI (Textual / prompt_toolkit / Rich).
  helpers              Shared utility functions (paths, env, heartbeat).
  tui                  Legacy compatibility shim — prefer individual modules.
"""
