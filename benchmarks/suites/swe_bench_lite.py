"""SWE-bench Lite suite — dry-run stub.

Full runner (git clone + uv venv + patch apply + pytest per problem) is
tracked in Sylveste-r8g. This stub exists so the bead's verification command
`--suite=swe-bench-lite --dry-run` still works end-to-end against the CLI
dispatcher: it returns 2 canned problems and a deterministic 0/2 pass@1
result without touching the network, Docker, or venvs.

Calling `load_problems(dry_run=False)` or `run_problem` with a non-fixture
input raises NotImplementedError, pointing operators at the follow-up bead.
"""

from __future__ import annotations

from dataclasses import dataclass

FOLLOWUP_BEAD = "Sylveste-r8g"


DRY_RUN_FIXTURE: list[dict] = [
    {
        "instance_id": "swebench_dryrun_fix_a",
        "repo": "dry-run/example",
        "base_commit": "0000000000000000000000000000000000000000",
        "problem_statement": "Fix the off-by-one in example_fn.",
        "fail_to_pass": ["tests/test_example.py::test_off_by_one"],
        "pass_to_pass": [],
        "difficulty": "stub",
    },
    {
        "instance_id": "swebench_dryrun_fix_b",
        "repo": "dry-run/example",
        "base_commit": "0000000000000000000000000000000000000000",
        "problem_statement": "Handle empty input to helper().",
        "fail_to_pass": ["tests/test_helper.py::test_empty"],
        "pass_to_pass": [],
        "difficulty": "stub",
    },
]


@dataclass
class SWEBenchProblem:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    difficulty: str

    @classmethod
    def from_raw(cls, raw: dict) -> SWEBenchProblem:
        return cls(
            instance_id=str(raw.get("instance_id", "unknown")),
            repo=raw.get("repo", ""),
            base_commit=raw.get("base_commit", ""),
            problem_statement=raw.get("problem_statement", ""),
            fail_to_pass=raw.get("fail_to_pass", []),
            pass_to_pass=raw.get("pass_to_pass", []),
            difficulty=raw.get("difficulty", "unknown"),
        )


def load_problems(
    limit: int | None = None, dry_run: bool = False
) -> list[SWEBenchProblem]:
    if not dry_run:
        raise NotImplementedError(
            f"SWE-bench Lite runner not yet implemented. Tracked in {FOLLOWUP_BEAD}. "
            "Use --dry-run to exercise the CLI dispatcher with fixture problems."
        )
    raws = DRY_RUN_FIXTURE[: limit or None]
    return [SWEBenchProblem.from_raw(r) for r in raws]


def run_problem(
    problem: SWEBenchProblem, model_output: str, per_test_timeout_s: float = 300.0
) -> dict:
    """Stub executor — always returns failure for fixture problems."""
    if problem.difficulty != "stub":
        raise NotImplementedError(
            f"SWE-bench Lite executor not yet implemented. Tracked in {FOLLOWUP_BEAD}."
        )
    return {
        "instance_id": problem.instance_id,
        "passed": False,
        "tests_total": len(problem.fail_to_pass),
        "tests_passed": 0,
        "error": f"stub_runner (see bead {FOLLOWUP_BEAD})",
        "elapsed_s": 0.0,
    }
