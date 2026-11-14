r"""Run an experiment across several API keys: free keys first, the paid key last.

Starts scripts/run_experiment.py with one Gemini key (grader) and one Groq key (simulator).
When the run stops on a quota limit, it checks which of the two keys is used up, moves that
role on to its next key, and continues; finished interviews are skipped as usual. It stops when
a role has no keys left: run it again the next day to continue.

    .venv/bin/python scripts/run_rotating.py --grader-keys 2 3 main --simulator-keys main 2 3 4 \
        --grader gemini --simulator groq --simulator-model openai/gpt-oss-120b \
        --systems B1 A2E A4 ADAPT12 A5E --run reviewed-main-v1 --operator kushal

With --by-candidate FILE (one candidate id per line) each candidate goes through every system
before the next one starts, so a run stopped early still holds complete, paired candidates.

Key names refer to .env: `main` is GEMINI_API_KEY / GROQ_API_KEY, `2` is GEMINI_API_KEY_2, and so
on. List the paid key last (or leave it out) so it is only used once the free keys are used up.
Every option other than --grader-keys and --simulator-keys goes to run_experiment.py unchanged.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.client import LLMQuotaError, OpenAICompatClient, load_env  # noqa: E402
from llm.providers import PROVIDERS  # noqa: E402

QUOTA_LINE = "quota is used up"        # printed by part4.runner on a quota stop
PAUSE_S = 120                          # when no key is used up, the limit was a short-term one
MAX_PAUSES = 3                         # ... but after this many in a row, move the grader on anyway


def key_value(provider: str, name: str) -> str | None:
    var = PROVIDERS[provider].key_var + ("" if name == "main" else f"_{name}")
    return os.environ.get(var)


def key_used_up(provider: str, key: str, model: str) -> bool:
    """One tiny uncached request: does this key still answer?"""
    p = PROVIDERS[provider]
    client = OpenAICompatClient(p.default_base, key, cache=False, max_attempts=1, max_wait_s=5)
    try:
        client.complete([{"role": "user", "content": "Reply OK."}], model=model, max_tokens=50,
                        reasoning_effort=p.reasoning_effort)
    except LLMQuotaError:
        return True
    except Exception:
        return False          # another problem is not a reason to give up the key
    return False


def run_once(rest: list[str], gemini_key: str, groq_key: str) -> tuple[bool, str]:
    """Run the experiment with these keys: (stopped on a quota limit?, the quota error lines)."""
    env = dict(os.environ, GEMINI_API_KEY=gemini_key, GROQ_API_KEY=groq_key)
    cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), *rest]
    quota, errors = False, []
    with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            quota = quota or QUOTA_LINE in line
            if "LLMQuotaError" in line:
                errors.append(line)
    return quota, "".join(errors)


def used_up_roles(error_text: str, simulator_model: str) -> tuple[bool, bool] | None:
    """(grader used up, simulator used up) read from the quota error, or None if it does not say.
    A tiny test request can still pass on a key whose daily tokens are nearly gone, so the
    error message is the more reliable sign."""
    text = error_text.lower()
    if simulator_model.lower() in text or "groq" in text:
        return False, True
    if "gemini" in text or "generativelanguage" in text:
        return True, False
    return None


def option(rest: list[str], name: str, default: str) -> str:
    return rest[rest.index(name) + 1] if name in rest else default


def read_ids(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class Keys:
    """The current grader and simulator key; each moves on only when it is used up."""

    GRADER_MODEL = "gemini-3.5-flash-lite"

    def __init__(self, grader_keys: list[str], simulator_keys: list[str], simulator_model: str):
        self.grader_keys, self.simulator_keys = grader_keys, simulator_keys
        self.simulator_model = simulator_model
        self.g = self.s = 0

    def run(self, rest: list[str]) -> bool:
        """Run until the experiment finishes (True) or a role has no keys left (False)."""
        pauses = 0
        while True:
            grader_key = key_value("gemini", self.grader_keys[self.g])
            simulator_key = key_value("groq", self.simulator_keys[self.s])
            print(f"\n=== Grader key {self.grader_keys[self.g]}, simulator key {self.simulator_keys[self.s]}",
                  flush=True)
            quota, error_text = run_once(rest, grader_key, simulator_key)
            if not quota:
                return True
            roles = used_up_roles(error_text, self.simulator_model)
            if roles:
                grader_out, simulator_out = roles
            else:
                grader_out = key_used_up("gemini", grader_key, self.GRADER_MODEL)
                simulator_out = key_used_up("groq", simulator_key, self.simulator_model)
            if not (grader_out or simulator_out):
                pauses += 1
                if pauses < MAX_PAUSES:
                    print(f"No key is used up; pausing {PAUSE_S}s before continuing.", flush=True)
                    time.sleep(PAUSE_S)
                    continue
                grader_out = True     # repeated short-term limits: give this grader key a rest
            pauses = 0
            self.g, self.s = self.g + grader_out, self.s + simulator_out
            for role, i, names in (("grader", self.g, self.grader_keys), ("simulator", self.s, self.simulator_keys)):
                if i == len(names):
                    print(f"\nEvery {role} key is used up for today. Run the same command again tomorrow "
                          "to continue; finished interviews are kept.", flush=True)
                    return False


def main() -> int:
    # no abbreviations: --grader (passed through) must not be read as --grader-keys
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--grader-keys", nargs="+", required=True, help="Gemini key names, free ones first")
    parser.add_argument("--simulator-keys", nargs="+", required=True, help="Groq key names, in order of use")
    parser.add_argument("--by-candidate", metavar="IDS_FILE",
                        help="one candidate at a time through every system, in the file's order")
    args, rest = parser.parse_known_args()
    if option(rest, "--grader", "") != "gemini" or option(rest, "--simulator", "") != "groq":
        parser.error("this script rotates Gemini grader keys and Groq simulator keys: "
                     "pass --grader gemini --simulator groq")
    load_env()
    for provider, names in (("gemini", args.grader_keys), ("groq", args.simulator_keys)):
        missing = [n for n in names if not key_value(provider, n)]
        if missing:
            parser.error(f"no {provider} key named {', '.join(missing)} in .env")

    candidates = read_ids(args.by_candidate) if args.by_candidate else [None]
    if args.by_candidate and ("--ids" in rest or "--ids-file" in rest):
        parser.error("--by-candidate already chooses the candidates; drop --ids / --ids-file")
    keys = Keys(args.grader_keys, args.simulator_keys, option(rest, "--simulator-model", PROVIDERS["groq"].default_model))
    for cid in candidates:
        if cid:
            print(f"\n##### {cid}: every system", flush=True)
        if not keys.run(rest + (["--ids", cid] if cid else [])):
            return 3
    return 0

if __name__ == "__main__":
    sys.exit(main())
