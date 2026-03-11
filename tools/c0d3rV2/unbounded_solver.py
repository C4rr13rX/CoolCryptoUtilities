"""
Unbounded Problem Solver — equation-matrix-driven resolution engine.

When the AI model declares a problem "impossible" or "out of scope," this
engine treats that response as a *starting point*.  Every place the model
says "we don't know X" represents a void in the environmental equation
matrix.  The solver:

  1. Asks: "Can we answer the original question?"
  2. If not, identifies the sub-questions that must be answered first.
  3. For each sub-question, researches, converts findings to equations,
     ingests into the matrix, detects gaps, generates hypotheses.
  4. Recurses: each sub-question may spawn its own sub-questions.
  5. When a leaf question is answered (backed by equations), the answer
     propagates up.  When ALL sub-questions of a parent are answered,
     the parent can be answered.
  6. The process terminates when the root question — the user's original
     request — is answered.

No caps.  No predefined disciplines.  No artificial limits.  The problem
defines the disciplines.  Everything is physics — proven with research
and equations.  It runs until the problem is solved.

This resembles Einstein/Planck thought experiments: hypotheses must fit
into known models, equations must be each other's answers, and paradoxes
become targets rather than stop signs.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionNode:
    """
    A node in the recursive question tree.

    The root is the user's original question.  Each unanswered aspect
    becomes a child.  Children can have their own children.  When all
    leaves are answered, answers propagate up until the root is answered.
    """

    question: str
    parent: QuestionNode | None = None
    children: list[QuestionNode] = field(default_factory=list)
    answer: str = ""
    equations: list[str] = field(default_factory=list)
    research_notes: str = ""
    research_links: list[str] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    status: str = "open"  # open | researching | answered | blocked

    @property
    def is_answered(self) -> bool:
        return self.status == "answered" and bool(self.answer)

    @property
    def all_children_answered(self) -> bool:
        return all(c.is_answered for c in self.children)

    @property
    def depth(self) -> int:
        d = 0
        node = self.parent
        while node:
            d += 1
            node = node.parent
        return d

    def add_child(self, question: str) -> QuestionNode:
        child = QuestionNode(question=question, parent=self)
        self.children.append(child)
        return child

    def tree_summary(self, indent: int = 0) -> str:
        prefix = "  " * indent
        mark = "[x]" if self.is_answered else "[ ]"
        eq_count = len(self.equations)
        line = f"{prefix}{mark} {self.question}"
        if eq_count:
            line += f" ({eq_count} equations)"
        if self.answer:
            line += f"\n{prefix}    -> {self.answer[:200]}"
        lines = [line]
        for child in self.children:
            lines.append(child.tree_summary(indent + 1))
        return "\n".join(lines)

    def count_open(self) -> int:
        count = 1 if self.status == "open" else 0
        for c in self.children:
            count += c.count_open()
        return count

    def count_answered(self) -> int:
        count = 1 if self.is_answered else 0
        for c in self.children:
            count += c.count_answered()
        return count

    def all_equations(self) -> list[str]:
        """Collect all equations from this node and descendants."""
        eqs = list(self.equations)
        for c in self.children:
            eqs.extend(c.all_equations())
        return eqs


@dataclass
class HypothesisEntry:
    """A hypothesis generated to reconcile a paradox or fill a void."""

    statement: str
    equation: str = ""
    supporting_equations: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    score: float = 0.0
    status: str = "proposed"  # proposed | tested | validated | rejected


@dataclass
class SolverResult:
    """Output of the solve process."""

    answered: bool = False
    answer: str = ""
    questions_total: int = 0
    questions_answered: int = 0
    equations_added: int = 0
    hypotheses: list[HypothesisEntry] = field(default_factory=list)
    research_links: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    question_tree: str = ""
    payload: dict = field(default_factory=dict)


class UnboundedSolver:
    """
    Resolve unbounded problems by recursively decomposing questions and
    filling the environmental equation matrix.

    No predefined disciplines.  No cycle caps.  No equation limits.
    The problem defines the disciplines.  It runs until the original
    question is answered — even if that means answering a question to a
    question to a question, and then propagating answers back up.

    Completion heuristic: the root question is answered when all its
    sub-questions are answered, backed by research-proven equations.
    """

    CONTROL_PREFIX: str = (
        "You are a closed-loop systems-engineering control system. "
        "Frame every decision as a hypothesis with measurable acceptance "
        "criteria.  Return deterministic, schema-compliant JSON only. "
        "No markdown fences, no prose outside the JSON object."
    )

    def __init__(
        self,
        session: Any,
        web_search: Any | None = None,
    ) -> None:
        self.session = session
        self.web_search = web_search
        # Running totals across the entire solve.
        self._total_equations: int = 0
        self._all_hypotheses: list[HypothesisEntry] = []
        self._all_research_links: list[str] = []
        self._all_anomalies: list[str] = []
        self._MAX_HYPOTHESES = 200
        self._MAX_LINKS = 500
        self._MAX_ANOMALIES = 200

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self, prompt: str, ai_response: str = "") -> SolverResult:
        """
        Run the full recursive solving process.

        *prompt*      — the user's original request (e.g. FTL travel).
        *ai_response* — the AI's "impossible" response (starting context).

        Runs until the root question is answered or truly blocked.
        """
        self._total_equations = 0
        self._all_hypotheses = []
        self._all_research_links = []
        self._all_anomalies = []

        # Build the root question node.
        root = QuestionNode(question=prompt)

        # Provide the AI's initial response as context for decomposition.
        if ai_response:
            root.research_notes = ai_response

        # Recursive solve.
        self._solve_node(root, ai_context=ai_response)

        # Build result.
        result = SolverResult(
            answered=root.is_answered,
            answer=root.answer,
            questions_total=root.count_open() + root.count_answered(),
            questions_answered=root.count_answered(),
            equations_added=self._total_equations,
            hypotheses=self._all_hypotheses,
            research_links=self._all_research_links,
            anomalies=self._all_anomalies,
            question_tree=root.tree_summary(),
            payload=self._build_payload(root),
        )

        # Persist.
        self._persist(prompt, result, result.payload)

        return result

    # ------------------------------------------------------------------
    # Recursive question resolution
    # ------------------------------------------------------------------

    def _solve_node(
        self,
        node: QuestionNode,
        ai_context: str = "",
    ) -> None:
        """
        Recursively solve a question node.

        1. Can we answer this question with current knowledge + matrix?
        2. If yes, record the answer and equations.  Done.
        3. If no, identify what sub-questions must be answered first.
        4. Recurse into each sub-question.
        5. After all children are answered, attempt to answer this node
           again using the accumulated child answers.
        """
        # Step 1: Attempt to answer directly.
        direct = self._attempt_answer(node, ai_context)

        if direct.get("answerable"):
            node.answer = str(direct.get("answer", ""))
            node.equations.extend(direct.get("equations") or [])
            node.status = "answered"
            self._ingest_node_equations(node)
            return

        # Step 2: Not directly answerable.  Decompose into sub-questions.
        node.status = "researching"
        sub_questions = direct.get("sub_questions") or []
        anomalies = direct.get("anomalies") or []
        self._all_anomalies.extend(anomalies)
        if len(self._all_anomalies) > self._MAX_ANOMALIES:
            self._all_anomalies = self._all_anomalies[-self._MAX_ANOMALIES:]

        # If the AI couldn't even decompose, research and try again.
        if not sub_questions:
            sub_questions = self._research_and_decompose(node)

        if not sub_questions:
            # Truly stuck — mark what we have and return.
            node.status = "blocked"
            return

        # Step 3: Create child nodes and recurse.
        for sq in sub_questions:
            q_text = str(sq) if isinstance(sq, str) else str(sq.get("question", sq))
            child = node.add_child(q_text)

        for child in node.children:
            self._solve_node(child)

        # Step 4: After all children, attempt to answer this node again.
        if node.all_children_answered:
            child_context = self._collect_child_answers(node)
            retry = self._attempt_answer(node, child_context)

            if retry.get("answerable"):
                node.answer = str(retry.get("answer", ""))
                node.equations.extend(retry.get("equations") or [])
                node.status = "answered"
                self._ingest_node_equations(node)
            else:
                # Children answered but still can't answer parent.
                # Generate hypotheses to bridge the remaining gap.
                hypotheses = self._generate_hypotheses(node)
                node.hypotheses.extend(
                    {"statement": h.statement, "equation": h.equation}
                    for h in hypotheses
                )
                self._all_hypotheses.extend(hypotheses)
                if len(self._all_hypotheses) > self._MAX_HYPOTHESES:
                    self._all_hypotheses = self._all_hypotheses[-self._MAX_HYPOTHESES:]

                # Try one more time with hypotheses as context.
                hyp_context = child_context + "\n\nHypotheses:\n" + "\n".join(
                    f"- {h.statement} ({h.equation})" for h in hypotheses
                )
                final = self._attempt_answer(node, hyp_context)
                if final.get("answerable"):
                    node.answer = str(final.get("answer", ""))
                    node.equations.extend(final.get("equations") or [])
                    node.status = "answered"
                    self._ingest_node_equations(node)
                else:
                    # Discover NEW sub-questions from the gap and recurse.
                    new_subs = final.get("sub_questions") or []
                    if new_subs:
                        for sq in new_subs:
                            q_text = str(sq) if isinstance(sq, str) else str(sq.get("question", sq))
                            child = node.add_child(q_text)
                        for child in node.children:
                            if not child.is_answered:
                                self._solve_node(child)
                        # Final attempt after new children.
                        if node.all_children_answered:
                            final_ctx = self._collect_child_answers(node)
                            last = self._attempt_answer(node, final_ctx)
                            if last.get("answerable"):
                                node.answer = str(last.get("answer", ""))
                                node.equations.extend(last.get("equations") or [])
                                node.status = "answered"
                                self._ingest_node_equations(node)
        else:
            # Some children still unanswered — partial progress.
            pass

    def _attempt_answer(self, node: QuestionNode, context: str = "") -> dict:
        """
        Ask AI: can we answer this question given current knowledge?

        Returns {answerable: bool, answer: str, equations: [str],
                 sub_questions: [str], anomalies: [str]}.
        """
        # Gather all equations accumulated so far in the tree.
        all_eqs = node.all_equations()

        # Load matrix equations relevant to this question.
        matrix_eqs = self._matrix_search(node.question)

        system = (
            self.CONTROL_PREFIX
            + " Return ONLY JSON with keys: answerable (bool), "
            "answer (string — the full answer if answerable, empty if not), "
            "equations (list of SymPy-parseable equation strings that back the answer), "
            "sub_questions (list of strings — questions that must be answered first, "
            "if not answerable), "
            "anomalies (list of paradoxes or unknowns encountered), "
            "research_needed (list of specific research queries if not answerable). "
            "IMPORTANT: Do NOT say something is impossible.  If you cannot answer "
            "directly, break it into sub-questions that CAN be researched and "
            "answered with proven physics, mathematics, and engineering.  "
            "The problem defines the disciplines — do not limit yourself to "
            "predefined categories.  Everything is physics at some level.  "
            "Every equation must be research-backed and SymPy-parseable."
        )

        prompt_parts = [f"Question:\n{node.question}"]

        if context:
            prompt_parts.append(f"\nContext:\n{context[:6000]}")

        if all_eqs:
            prompt_parts.append(
                f"\nEquations accumulated so far:\n"
                + "\n".join(f"- {eq}" for eq in all_eqs[-50:])
            )

        if matrix_eqs:
            prompt_parts.append(
                f"\nRelevant equations from matrix:\n"
                + "\n".join(f"- [{e.get('domain','')}] {e.get('equation','')}"
                            for e in matrix_eqs[:20])
            )

        if node.children:
            prompt_parts.append(
                f"\nSub-questions already explored:\n"
                + "\n".join(
                    f"- {'[answered]' if c.is_answered else '[open]'} {c.question}"
                    for c in node.children
                )
            )

        prompt = "\n".join(prompt_parts)

        try:
            raw = self.session.send(
                prompt=prompt, stream=False, system=system,
            )
            result = self._safe_json(raw or "")
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return {"answerable": False, "sub_questions": [], "equations": []}

    def _research_and_decompose(self, node: QuestionNode) -> list[str]:
        """
        Research the question and use findings to decompose it into
        sub-questions.
        """
        # Research.
        research = ""
        if self.web_search:
            try:
                result = self.web_search.search(node.question)
                research = result.get("summary", "")
                node.research_notes = research
                for r in result.get("results") or []:
                    url = r.get("url", "")
                    if url:
                        node.research_links.append(url)
                        if len(self._all_research_links) < self._MAX_LINKS:
                            self._all_research_links.append(url)
            except Exception:
                pass

        # Extract equations from research.
        if research:
            eqs = self._equations_from_text(research, node.question)
            node.equations.extend(eqs)
            self._total_equations += len(eqs)
            self._ingest_node_equations(node)

        # Ask AI to decompose using research.
        system = (
            self.CONTROL_PREFIX
            + " Return ONLY JSON with key 'sub_questions' (list of strings). "
            "Break this question into the specific sub-questions that must "
            "be answered — each answerable through research, equations, and "
            "proven science.  Do not limit to predefined disciplines.  "
            "The problem defines what fields are relevant."
        )
        prompt = f"Question: {node.question}"
        if research:
            prompt += f"\n\nResearch findings:\n{research[:4000]}"

        try:
            raw = self.session.send(
                prompt=prompt, stream=False, system=system,
            )
            result = self._safe_json(raw or "")
            if isinstance(result, dict):
                return result.get("sub_questions") or []
        except Exception:
            pass
        return []

    def _collect_child_answers(self, node: QuestionNode) -> str:
        """Collect all child answers + equations into a context string."""
        parts: list[str] = []
        for child in node.children:
            status = "ANSWERED" if child.is_answered else "UNANSWERED"
            parts.append(f"[{status}] {child.question}")
            if child.answer:
                parts.append(f"  Answer: {child.answer}")
            if child.equations:
                parts.append(f"  Equations: {child.equations}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Equation extraction and ingestion
    # ------------------------------------------------------------------

    def _equations_from_text(self, text: str, question: str) -> list[str]:
        """
        Ask AI to extract/generate equations from research text.
        Validates with SymPy.  No cap on count.
        """
        system = (
            self.CONTROL_PREFIX
            + " Return ONLY JSON with key 'equations' (list of strings). "
            "Extract or derive all mathematical equations from the text.  "
            "Each equation must be SymPy-parseable.  Include units, "
            "constants, and variable definitions as comments after each "
            "equation if needed.  No limit on how many."
        )
        try:
            raw = self.session.send(
                prompt=(
                    f"Question context: {question[:500]}\n\n"
                    f"Text:\n{text[:5000]}\n\n"
                    "Extract ALL equations.  Convert natural language "
                    "relationships into mathematical form."
                ),
                stream=False,
                system=system,
            )
            result = self._safe_json(raw or "")
            if isinstance(result, dict):
                equations = result.get("equations") or []
                # Validate with SymPy — keep only parseable ones.
                valid: list[str] = []
                try:
                    import sympy as _sp
                    for eq in equations:
                        try:
                            if "=" in eq:
                                left, right = eq.split("=", 1)
                                _sp.Eq(_sp.sympify(left), _sp.sympify(right))
                            else:
                                _sp.sympify(eq)
                            valid.append(eq)
                        except Exception:
                            continue
                except ImportError:
                    valid = list(equations)
                return valid
        except Exception:
            pass
        return []

    def _ingest_node_equations(self, node: QuestionNode) -> None:
        """Ingest a node's equations into Django + Kuzu."""
        if not node.equations:
            return
        try:
            from helpers import _ensure_django_ready
            if not _ensure_django_ready():
                return
            from services.equation_graph import ingest_equations

            eq_text = "\n".join(node.equations)
            ingest_equations(
                eq_text,
                source_title="Unbounded Solver",
                discipline_tags=["derived"],
                tool_used="unbounded_solver",
                raw_excerpt=node.question[:500],
            )
            self._total_equations += len(node.equations)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Matrix integration
    # ------------------------------------------------------------------

    def _matrix_search(self, question: str) -> list[dict]:
        """Search the equation matrix for equations relevant to a question."""
        try:
            from matrix_helpers import _matrix_search
            result = _matrix_search(question, limit=20)
            return result.get("hits") or []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    def _generate_hypotheses(self, node: QuestionNode) -> list[HypothesisEntry]:
        """
        Generate hypotheses to bridge the gap between child answers
        and the parent question.  Each hypothesis must be expressible
        as an equation.
        """
        child_summary = self._collect_child_answers(node)

        system = (
            self.CONTROL_PREFIX
            + " Return ONLY JSON with key 'hypotheses' (list).  "
            "Each hypothesis: {statement: str, equation: str, "
            "supporting_equations: [str], contradictions: [str], "
            "score_0_1: float}. "
            "The child questions have been answered but the parent "
            "question cannot yet be answered.  Generate hypotheses that "
            "bridge the gap.  Each hypothesis MUST include an equation "
            "that is SymPy-parseable and integrates with the existing "
            "equations.  Hypotheses must fit into known models and be "
            "backed by the child answers."
        )

        try:
            raw = self.session.send(
                prompt=(
                    f"Parent question: {node.question}\n\n"
                    f"Child answers:\n{child_summary[:4000]}\n\n"
                    "Generate bridging hypotheses with equations."
                ),
                stream=False,
                system=system,
            )
            result = self._safe_json(raw or "")
            if isinstance(result, dict):
                entries: list[HypothesisEntry] = []
                for h in result.get("hypotheses") or []:
                    entries.append(HypothesisEntry(
                        statement=str(h.get("statement", "")),
                        equation=str(h.get("equation", "")),
                        supporting_equations=h.get("supporting_equations") or [],
                        contradictions=h.get("contradictions") or [],
                        score=float(h.get("score_0_1") or 0),
                    ))
                return entries
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Math grounding (convert prompt to equations)
    # ------------------------------------------------------------------

    def math_grounding(self, prompt: str) -> dict:
        """
        Convert the prompt into mathematical form: variables, unknowns,
        equations, constraints.  Research missing constants.  Solve with
        SymPy where possible.
        """
        from helpers import _strip_context_block
        base_request = _strip_context_block(prompt)

        # Pass 1: extract mathematical structure.
        system = (
            self.CONTROL_PREFIX
            + " Return ONLY JSON with keys: variables (list), unknowns (list), "
            "equations (list of strings), constraints (list), "
            "mapping (list), research_questions (list), "
            "symbol_definitions (dict), equation_units (dict). "
            "Express the request as math; if you cannot form explicit "
            "equations, provide symbolic placeholders AND research_questions "
            "to identify missing constants/relations. "
            "Provide units for equations and define symbols explicitly "
            "so they are measurable.  No limit on equations."
        )

        try:
            raw = self.session.send(
                prompt=base_request, stream=False, system=system,
            )
            payload = self._safe_json(raw or "")
        except Exception:
            payload = {}

        if not isinstance(payload, dict):
            payload = {}

        # Research missing pieces.
        research_questions = payload.get("research_questions") or [
            f"Find equations, constants, or constraints needed to model: {base_request}"
        ]
        research_notes = self._research_questions(research_questions)

        # Pass 2: convert research into equations.
        if research_notes:
            system2 = (
                self.CONTROL_PREFIX
                + " Return ONLY JSON with keys: equations (list), constraints (list), "
                "gap_fill_steps (list), research_links (list). "
                "Convert the research into explicit equations and constraints, "
                "stating assumptions and measurement units."
            )
            try:
                raw2 = self.session.send(
                    prompt=(
                        f"Request:\n{base_request}\n\n"
                        f"Research:\n{research_notes}"
                    ),
                    stream=False,
                    system=system2,
                )
                payload2 = self._safe_json(raw2 or "")
                if isinstance(payload2, dict):
                    for key in ("equations", "constraints", "gap_fill_steps", "research_links"):
                        if payload2.get(key):
                            payload[key] = payload2[key]
            except Exception:
                pass

            # Extract links from research notes.
            if not payload.get("research_links"):
                links = re.findall(r"https?://[^\s\]\)\"']+", research_notes)
                if links:
                    payload["research_links"] = links

            # Ingest equations.
            try:
                from helpers import _ensure_django_ready
                if _ensure_django_ready():
                    from services.equation_graph import (
                        ingest_equations,
                        extract_equations,
                    )
                    if not payload.get("equations"):
                        extracted = extract_equations(research_notes)
                        payload["equations"] = [e.text for e in extracted]
                    for link in (payload.get("research_links") or [])[:10]:
                        ingest_equations(
                            research_notes,
                            source_title="Math Grounding Research",
                            source_url=str(link),
                            discipline_tags=["cross-disciplinary"],
                            raw_excerpt=research_notes[:1000],
                        )
            except Exception:
                pass

        # Solve with SymPy.
        solutions = self._solve_with_sympy(payload)

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request": base_request,
            "variables": payload.get("variables") or [],
            "unknowns": payload.get("unknowns") or [],
            "equations": payload.get("equations") or [],
            "constraints": payload.get("constraints") or [],
            "gap_fill_steps": payload.get("gap_fill_steps") or [],
            "research_links": payload.get("research_links") or [],
            "symbol_definitions": payload.get("symbol_definitions") or {},
            "equation_units": payload.get("equation_units") or {},
            "solutions": solutions,
        }

        self._persist_grounding(record)
        return record

    # ------------------------------------------------------------------
    # Context block formatting
    # ------------------------------------------------------------------

    def format_context_block(self, result: SolverResult) -> str:
        """Format the solver result as a context block for injection."""
        lines: list[str] = []

        if result.answer:
            lines.append(f"Answer: {result.answer}")

        lines.append(
            f"Progress: {result.questions_answered}/{result.questions_total} "
            f"questions answered, {result.equations_added} equations derived"
        )

        if result.hypotheses:
            lines.append("Hypotheses:")
            for h in result.hypotheses:
                lines.append(f"- [{h.score:.2f}] {h.statement}")
                if h.equation:
                    lines.append(f"  Equation: {h.equation}")

        if result.anomalies:
            lines.append("Anomalies targeted:")
            for item in result.anomalies:
                lines.append(f"- {item}")

        if result.question_tree:
            lines.append(f"\nQuestion tree:\n{result.question_tree}")

        if not lines:
            return ""
        return "\n[unbounded_resolution]\n" + "\n".join(lines)

    def format_grounding_block(self, record: dict) -> str:
        """Format a math grounding record as a context block."""
        return "\n".join([
            "[math_grounding]",
            f"variables: {record.get('variables', [])}",
            f"unknowns: {record.get('unknowns', [])}",
            f"equations: {record.get('equations', [])}",
            f"constraints: {record.get('constraints', [])}",
            f"gap_fill_steps: {record.get('gap_fill_steps', [])}",
            f"research_links: {record.get('research_links', [])}",
            f"solutions: {record.get('solutions', [])}",
        ])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _research_questions(self, questions: list[str]) -> str:
        """Research questions via web search and return combined notes."""
        if not self.web_search:
            return ""
        notes: list[str] = []
        for q in questions:
            try:
                result = self.web_search.search(str(q))
                summary = result.get("summary", "")
                if summary:
                    notes.append(summary)
                for r in result.get("results") or []:
                    url = r.get("url", "")
                    if url:
                        if len(self._all_research_links) < self._MAX_LINKS:
                            self._all_research_links.append(url)
            except Exception:
                continue
        return "\n\n".join(notes)

    def _solve_with_sympy(self, payload: dict) -> list:
        """Attempt to solve equations with SymPy."""
        try:
            import sympy as _sp
        except ImportError:
            return []

        eqs = []
        symbols: dict[str, Any] = {}
        for eq in payload.get("equations") or []:
            try:
                if "=" in eq:
                    left, right = eq.split("=", 1)
                    expr = _sp.Eq(_sp.sympify(left), _sp.sympify(right))
                else:
                    expr = _sp.sympify(eq)
                eqs.append(expr)
            except Exception:
                continue

        for var in payload.get("unknowns") or []:
            try:
                symbols[str(var)] = _sp.symbols(str(var))
            except Exception:
                continue

        if eqs and symbols:
            try:
                sol = _sp.solve(eqs, list(symbols.values()), dict=True)
                return sol if isinstance(sol, list) else [sol]
            except Exception:
                return []
        return []

    def _build_payload(self, root: QuestionNode) -> dict:
        """Build a flat payload dict from the question tree for persistence."""
        all_eqs = root.all_equations()
        return {
            "question": root.question,
            "answered": root.is_answered,
            "answer": root.answer,
            "equations": all_eqs,
            "hypotheses": [
                {"statement": h.statement, "equation": h.equation, "score": h.score}
                for h in self._all_hypotheses
            ],
            "anomalies": self._all_anomalies,
            "research_links": self._all_research_links,
            "question_tree": root.tree_summary(),
            "total_questions": root.count_open() + root.count_answered(),
            "answered_questions": root.count_answered(),
        }

    def _persist(self, prompt: str, result: SolverResult, payload: dict) -> None:
        """Persist the solve result to Django and to disk."""
        try:
            from helpers import _ensure_django_ready
            if _ensure_django_ready():
                from core.models import UnboundedMatrixRecord
                UnboundedMatrixRecord.objects.create(
                    prompt=prompt[:2000],
                    equations=payload.get("equations") or [],
                    hypotheses=payload.get("hypotheses") or [],
                    anomalies=self._all_anomalies,
                    research_links=self._all_research_links,
                    bounded_task=result.answer[:2000] if result.answer else "",
                    payload=payload,
                )
        except Exception:
            pass

        try:
            from helpers import _runtime_root
            out_dir = _runtime_root()
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "unbounded_payload.json"
            path.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _persist_grounding(self, record: dict) -> None:
        """Persist math grounding record to disk."""
        try:
            from helpers import _runtime_root
            out_dir = _runtime_root()
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "math_grounding.json").write_text(
                json.dumps(record, indent=2, default=str),
                encoding="utf-8",
            )
            with (out_dir / "math_grounding_history.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    @staticmethod
    def _safe_json(text: str) -> Any:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None
