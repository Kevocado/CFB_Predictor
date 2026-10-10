"""A merge to main must deploy to the VPS, and nothing else should.

The deploy workflow existed only for Azure, was `workflow_dispatch`-only, and
had no VPS job at all, so a merge deployed nothing: every deploy was a manual
image build plus a hand-run `bin/deploy`. This pins the replacement, because
the failure mode it guards is silent — a workflow that does not run looks
exactly like a workflow with nothing to deploy.

Two things here are not obvious and are the reason the test exists:

* **The paths filter is a whitelist**, so excluding the files the scheduled
  snapshot jobs commit is implicit. That means nothing stops someone adding
  `data/**` and putting every scheduled refresh into a deploy loop. The
  exclusion is asserted directly instead of trusted to the filter's shape.

* **Azure is gated with `== 'true'`, not `!= 'false'`.** The uncommitted local
  edit in the sibling checkouts used the failing-open form, which turns Azure
  back on for any repo that has not explicitly set the variable. This is
  legacy infrastructure being cut over, so it has to fail closed.

Every check below is also run against a deliberately broken copy of the file
to prove it can fail. A workflow guard that passes vacuously is worse than no
guard, because it is trusted.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO / ".github" / "workflows"
WORKFLOW = WORKFLOW_DIR / "deploy.yml"

# Must match `image:` in vps-stack/compose.yml. If either side changes, the
# deploy pushes a tag the stack never pulls.
IMAGE = "ghcr.io/kevocado/cfb-predictor"
# Must match the service key in vps-stack/compose.yml, which is what
# `bin/deploy` is called with.
SERVICE = "cfb"

# Baked into the image by the Dockerfile, and hand-committed: a change to any
# of these changes the artifact, so it must deploy.
MUST_DEPLOY = [
    "src/**",
    "models/**",
    "Dockerfile",
    "pyproject.toml",
    ".github/workflows/deploy.yml",
]

# Written by .github/workflows/refresh-public-snapshot.yml on a timer. The
# runtime polls them from raw.githubusercontent.com, so a new snapshot needs no
# deploy -- and if it triggered one, every repo would redeploy several times a
# day for nothing.
MUST_NOT_DEPLOY = [
    "data/public_snapshot.json",
    "data/**",
]

# Paths a scheduled job commits that this filter deliberately DOES watch, with
# the reason. Each entry is a decision, not an oversight.
#
# This exists because "a scheduled commit must never redeploy" is the right rule
# for four of the five sites and the wrong rule for NBA. Those four poll
# raw.githubusercontent.com at request time, so a new snapshot is picked up
# without a new image and excluding it is free. NBA does neither: it has no
# runtime poll and the VPS runs it with PUBLIC_MODE unset, so data/cache/ and
# models/ are baked into the image and read per request. Excluding them would
# make the deploy filter ship a frozen schedule and a frozen model, which is
# worse than an extra deploy. The cost is that NBA redeploys when its daily
# retrain lands; the alternative is to give NBA the runtime poll its siblings
# have, and then this entry can go.
WATCHED_ANYWAY = {
    # Glob patterns, matched against the paths the other workflows `git add`.
    # The first version used bare directory names ("models/") and matched
    # nothing: fnmatch needs the whole string, so "models/" never matches
    # "models/*.pkl" and the classification silently rejected a path it had just
    # been written to accept.
    "models/**": "read per request; the daily retrain only reaches the VPS by redeploy",
    "data/cache/**": "read per request; no runtime poll, so a new schedule only lands by redeploy",
}

GATE_AZURE = "vars.DEPLOY_AZURE == 'true'"
GATE_VPS = "vars.VPS_HOST != ''"


def workflow_text() -> str:
    return WORKFLOW.read_text()


def jobs(text: str) -> dict[str, str]:
    """Split the `jobs:` block into {name: body}, so a gate can be read."""
    out: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    in_jobs = False
    for line in text.splitlines():
        if re.match(r"^jobs:\s*$", line):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if m:
            if name:
                out[name] = "\n".join(buf)
            name, buf = m.group(1), []
        elif name is not None:
            buf.append(line)
    if name:
        out[name] = "\n".join(buf)
    return out


def paths_filter(text: str) -> list[str]:
    """The `paths:` entries of the push trigger."""
    m = re.search(r"^on:[ \t]*$", text, re.M)
    assert m, "the workflow has no top-level `on:` block"
    # The `on:` block is everything up to the next top-level key. Comments at
    # column zero belong to it, and this workflow's have to: the exclusion
    # rationale is written there. The first version of this reader stopped at
    # the first non-indented line and so saw an empty block whenever a comment
    # sat between `on:` and `push:` -- which made the exclusion assertions
    # vacuous rather than failing.
    block: list[str] = []
    for line in text[m.end():].splitlines():
        if line.strip() == "" or line.lstrip().startswith("#") or line[:1] in (" ", "\t"):
            block.append(line)
        else:
            break
    trigger = "\n".join(block)
    p = re.search(r"^[ \t]+paths:\n((?:[ \t]+-[ \t]*.+\n)+)", trigger, re.M)
    assert p, "the push trigger has no `paths:` filter, so every data refresh redeploys"
    return [
        line.strip().lstrip("- ").strip().strip("'\"")
        for line in p.group(1).splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------- the checks


def check_triggers(text: str) -> None:
    assert re.search(r"^on:[ \t]*$", text, re.M), "no top-level `on:` block"
    assert re.search(r"^[ \t]+workflow_dispatch:", text, re.M), (
        "workflow_dispatch is gone, so a deploy can no longer be run by hand"
    )
    assert re.search(r"^[ \t]+push:[ \t]*$", text, re.M), "the workflow does not run on push"
    assert re.search(r"^[ \t]+branches:[ \t]*\[?[ \t]*main", text, re.M), (
        "the push trigger is not limited to main"
    )
    paths = paths_filter(text)
    for required in MUST_DEPLOY:
        assert required in paths, f"{required!r} is not in the paths filter, so a change to it deploys nothing: {paths}"
    for forbidden in MUST_NOT_DEPLOY:
        assert forbidden not in paths, (
            f"{forbidden!r} is in the paths filter. The scheduled refresh commits it, so this "
            f"would redeploy on every refresh: {paths}"
        )


def check_image(text: str) -> None:
    j = jobs(text)
    assert "build" in j, f"no `build` job to produce the image: {sorted(j)}"
    body = j["build"]
    assert "docker/login-action" in body, "the build job never logs in to a registry"
    # GITHUB_TOKEN is the target state and the reason this assertion exists: a PAT is
    # a credential that outlives the repo and has to be rotated by hand.
    #
    # GHCR_PAT is accepted ONLY because the package is not linked to this repository.
    # ghcr.io/kevocado/cfb-predictor is user-scoped, and GITHUB_TOKEN may only write to
    # packages linked to its own repository -- so the push died with
    # `denied: permission_denied: write_package` AFTER the image had built and
    # tagged correctly. Linking a package is a one-time action in package settings and
    # has no API, so neither a workflow nor CI can do it.
    #
    # This is not a new long-lived credential: GHCR_PAT is already a secret on this
    # repo and is what these images have always been pushed with. If the package is
    # ever linked, delete the GHCR_PAT alternative here AND in the workflow -- that
    # linked state is what this test was written to prefer.
    assert re.search(
        r"password:\s*\$\{\{\s*secrets\.(?:GITHUB_TOKEN|GHCR_PAT)\s*\}\}", body
    ), (
        "GHCR login must use secrets.GITHUB_TOKEN (or GHCR_PAT while the package is "
        "unlinked) -- some other credential is neither"
    )
    assert re.search(r"registry:\s*ghcr\.io", body), "the registry is not ghcr.io"
    assert f"{IMAGE}:${{{{ github.sha }}}}" in body, (
        f"the build must push {IMAGE}:${{{{{{ github.sha }}}}}} — the stack pins a sha"
    )
    assert f"{IMAGE}:latest" in body, f"the build must also push {IMAGE}:latest"
    # Pushed either by two explicit `docker push` commands, or by docker/build-push-action with `push: true` (which
    # pushes every tag listed above, so the two tag assertions already cover "both tags").
    pushed_by_action = bool(re.search(r"uses:\s*docker/build-push-action@", body)) and bool(re.search(r"push:\s*true", body))
    assert body.count("docker push") >= 2 or pushed_by_action, "both the sha and latest tags must be pushed"


def check_vps(text: str) -> None:
    j = jobs(text)
    vps = [n for n, b in j.items() if "vars.VPS_HOST" in b]
    assert len(vps) == 1, f"expected exactly one VPS job, found {vps}"
    body = j[vps[0]]
    assert re.search(r"^[ \t]+needs:[ \t]*build[ \t]*$", body, re.M), (
        "the VPS job must need `build`, or it can restart the stack on an image that was never pushed"
    )
    assert re.search(rf"^[ \t]+if:[ \t]*{re.escape(GATE_VPS)}[ \t]*$", body, re.M), (
        f"the VPS job must be gated on `{GATE_VPS}`, so a repo without VPS_HOST skips it"
    )
    assert re.search(rf"^[ \t]+concurrency:[ \t]*vps-deploy-{SERVICE}[ \t]*$", body, re.M), (
        f"the VPS job must serialise on `concurrency: vps-deploy-{SERVICE}`: two repos deploying at "
        f"once race on /opt/stack/.env"
    )
    assert re.search(r"\$\{\{\s*secrets\.VPS_SSH_KEY\s*\}\}", body), "VPS_SSH_KEY is never used"
    assert re.search(r"\$\{\{\s*secrets\.VPS_KNOWN_HOSTS\s*\}\}", body), (
        "VPS_KNOWN_HOSTS is never used, so the SSH host is not verified"
    )
    expected = f"ssh deploy@${{{{ vars.VPS_HOST }}}} deploy {SERVICE} ${{{{ github.sha }}}}"
    assert expected in body, f"the deploy command must be exactly: {expected}"
    # Known hosts must be written before the first connection, or ssh prompts
    # and the job hangs until it times out.
    assert body.index("known_hosts") < body.index("ssh deploy@"), "known_hosts is written after the ssh call"


def check_azure_fails_closed(text: str) -> None:
    assert "DEPLOY_AZURE != 'false'" not in text, (
        "`vars.DEPLOY_AZURE != 'false'` fails OPEN: any repo that has not set the variable "
        "turns Azure back on. Legacy infrastructure being cut over must use `== 'true'`"
    )
    j = jobs(text)
    uses_azure = {
        n: b for n, b in j.items()
        if re.search(r"^[ \t]*(az [ \t]|.*azure/login|.*containerapp)", b, re.M)
    }
    assert uses_azure, "the Azure deploy steps are gone entirely"
    for name, body in uses_azure.items():
        assert re.search(rf"^[ \t]+if:[ \t]*{re.escape(GATE_AZURE)}[ \t]*$", body, re.M), (
            f"job {name!r} touches Azure but is not gated on `{GATE_AZURE}`. Every repo would "
            f"deploy to Azure on every merge"
        )


def check_no_secret_material(text: str) -> None:
    for m in re.finditer(r"secrets\.([A-Za-z0-9_]+)", text):
        assert m.group(1).isupper(), f"secret {m.group(1)!r} is not a UPPER_CASE name"
    for marker in ("BEGIN OPENSSH PRIVATE KEY", "BEGIN RSA PRIVATE KEY", "ghp_", "github_pat_"):
        assert marker not in text, f"the workflow contains credential material ({marker!r})"
    # A secret's value must only ever be read from the environment, never
    # echoed into a log line.
    for line in text.splitlines():
        if "secrets." in line and ("echo" in line or "::" in line):
            raise AssertionError(f"a secret reaches the log: {line.strip()!r}")


def check_no_catch_all(text: str) -> None:
    """No entry may match every commit.

    The other exclusions in this file are named files and directories that
    existed when it was written. This one is the backstop: a bare `**`, `.` or
    `*` in the filter matches everything, so any future refresh job -- in this
    repo or any repo this is copied to -- silently redeploys the service on
    every run, and the named exclusions would never fire because they would be
    redundant. It is also the only exclusion that is not vacuous in a repo with
    no refresh job of its own.
    """
    for path in paths_filter(text):
        assert path not in ("**", ".", "*", ""), (
            f"the paths filter contains {path!r}, which matches every commit: a scheduled "
            f"refresh anywhere in this repo would then redeploy the service each time"
        )


def check_yaml_shape(text: str) -> None:
    """Indentation is a multiple of two spaces, everywhere.

    Added after this file's own checks all passed on a workflow that was not
    valid YAML: the Azure steps had been re-indented by two and ended up
    nested under the preceding step. Nothing here noticed, because a text-based
    guard cannot see structure. actionlint caught it, and it is required for
    that reason -- but a merge should not be the first place invalid YAML is
    discovered, so the cheap invariant is asserted too.
    """
    for n, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if stripped and indent % 2:
            raise AssertionError(
                f"line {n} is indented {indent} spaces, which is not a multiple of two: "
                f"{line!r}"
            )
    # Every `steps:` key must be followed by list items at exactly six spaces.
    for m in re.finditer(r"^[ \t]+steps:[ \t]*$", text, re.M):
        rest = text[m.end():].lstrip("\n")
        first = rest.splitlines()[0] if rest.splitlines() else ""
        assert first.startswith("      - "), (
            f"a `steps:` block whose first item is not at six spaces: {first!r}. "
            f"Wrong indentation here silently nests one step under another."
        )


CHECKS = {
    "shape": check_yaml_shape,
    "catchall": check_no_catch_all,
    "triggers": check_triggers,
    "image": check_image,
    "vps": check_vps,
    "azure": check_azure_fails_closed,
    "secrets": check_no_secret_material,
}


# ---------------------------------------------------------------- the tests


def test_the_workflow_file_exists():
    assert WORKFLOW.exists(), f"{WORKFLOW} is missing; a merge deploys nothing"
    assert len(workflow_text()) > 400, "the workflow looks like a stub"


def test_each_check_can_fail():
    """A guard that cannot fail is not a guard.

    Each check is run against the real file, and against a copy with one thing
    broken. If a broken copy passes, the check is vacuous and this fails.
    """
    good = workflow_text()
    for name, check in CHECKS.items():
        check(good)  # the real file satisfies it
    # Each mutation must be a string that appears EXACTLY ONCE, or it is not
    # breaking the thing it claims to. The first version of this table replaced
    # one of two `:latest` occurrences and the check still passed, which is
    # exactly what the count assertion exists to prevent.
    broken = {
        "triggers": ("workflow_dispatch:", "workflow_DISABLED:"),
        # A third credential: neither GITHUB_TOKEN nor the GHCR_PAT that an unlinked
        # package currently forces, so check_image still has something it can reject.
        "image": ("secrets.GHCR_PAT", "secrets.REGISTRY_TOKEN"),
        "vps": (f"deploy {SERVICE} ${{{{ github.sha }}}}", f"deploy {SERVICE}"),
        "azure": (GATE_AZURE, "vars.DEPLOY_AZURE != 'false'"),
        # Lower-cased, so check_no_secret_material's UPPER_CASE rule fires. Anchored on
        # the credential actually in the file: GITHUB_TOKEN appears zero times now,
        # and a mutation whose anchor is absent is a mutation that breaks nothing.
        "secrets": ("secrets.GHCR_PAT", "secrets.ghcr_pat"),
        # A unique anchor: `runs-on: ubuntu-latest` appears once per job, and the
        # count assertion below rejects a mutation that is not unique.
        "shape": (
            f"concurrency: vps-deploy-{SERVICE}",
            f" concurrency: vps-deploy-{SERVICE}",
        ),
        "catchall": (f"'{MUST_DEPLOY[0]}'", "'**'"),
    }
    for name, (old, new) in broken.items():
        assert good.count(old) == 1, (
            f"the {name!r} mutation {old!r} appears {good.count(old)} times, so breaking one "
            f"copy leaves the check satisfied by another"
        )
        try:
            CHECKS[name](good.replace(old, new, 1))
        except AssertionError:
            continue
        raise AssertionError(f"the {name!r} check passed a workflow with {old!r} broken")


def test_the_paths_reader_reads_a_normal_workflow():
    """`paths_filter` is load-bearing for the exclusion assertions, and a reader
    that quietly returned [] would make them vacuous. Pinned against a sample
    shaped like the real file, comments at column zero included.
    """
    sample = (
        "on:\n"
        "# why this filter looks like this, at column zero\n"
        "  push:\n"
        "    branches: [main]\n"
        "    paths:\n"
        "      - 'src/**'\n"
        "      - \"Dockerfile\"\n"
        "  workflow_dispatch: {}\n"
        "permissions:\n"
        "  contents: read\n"
    )
    assert paths_filter(sample) == ["src/**", "Dockerfile"]

    with pytest.raises(AssertionError, match="no `paths:` filter"):
        paths_filter(sample.replace("    paths:\n", ""))


def test_a_merge_to_main_triggers_a_deploy():
    check_triggers(workflow_text())


def test_a_data_refresh_does_not_redeploy():
    check_triggers(workflow_text())


def test_it_builds_and_pushes_the_image_the_stack_pulls():
    check_image(workflow_text())


def test_it_reaches_the_vps_and_asks_for_this_service():
    check_vps(workflow_text())


def test_azure_stays_off_unless_asked_for():
    check_azure_fails_closed(workflow_text())


def test_no_secret_material_in_the_workflow():
    check_no_secret_material(workflow_text())


def test_the_filter_has_no_catch_all_entry():
    check_no_catch_all(workflow_text())


def test_the_yaml_indentation_is_sane():
    check_yaml_shape(workflow_text())


def committed_paths() -> dict[str, set[str]]:
    """What every OTHER workflow in this repo commits, by workflow name.

    A GitHub Actions job cannot push to the branch it is running on, so these
    workflows commit to `main` via a checkout with a token. That is the whole
    mechanism by which a deploy loop starts: the refresh job commits a path the
    deploy filter watches, the commit triggers a build, and the job that made
    the commit has no idea it did.
    """
    out: dict[str, set[str]] = {}
    for path in sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml")):
        if path.name == WORKFLOW.name:
            continue
        # Join shell line continuations first, or a `git add a b \\` line hides
        # everything after the backslash.
        text = path.read_text().replace("\\\n", " ")
        added: set[str] = set()
        for m in re.finditer(r"git add\s+(.*)", text):
            for token in m.group(1).split():
                token = token.strip("'\"")
                if token.startswith("-"):  # git add -f path
                    continue
                if "/" in token or token.endswith((".json", ".db")):
                    added.add(token)
        if added:
            out[path.name] = added
    return out


def test_every_scheduled_commit_is_deliberately_classified():
    """Derived from the other workflows, not from a hand-written list.

    The refresh jobs commit to `main` (a workflow cannot push to its own branch,
    so they check out main with a token). Any path such a job commits that the
    deploy filter watches puts the repo in a loop: the commit triggers a build,
    and the job that made the commit has no idea it did.

    Rather than forbid the overlap outright -- which is wrong for NBA, see
    WATCHED_ANYWAY -- every overlap has to be classified. A new refresh job, or
    a new file it writes, lands here as an unclassified path and fails, instead
    of quietly deploying several times a day.
    """
    watched = paths_filter(workflow_text())
    classified = set(WATCHED_ANYWAY)
    for name, added in committed_paths().items():
        for path in sorted(added):
            if any(fnmatch.fnmatch(path, pat) for pat in MUST_NOT_DEPLOY):
                continue  # explicitly excluded, and check_triggers keeps it that way
            if any(fnmatch.fnmatch(path, pat) for pat in classified):
                continue  # deliberately watched, with a recorded reason
            watched_by = [w for w in watched if fnmatch.fnmatch(path, w)]
            raise AssertionError(
                f"{name} commits {path!r}"
                + (f", which the deploy paths filter watches via {watched_by}" if watched_by else "")
                + ", but it is in neither MUST_NOT_DEPLOY nor WATCHED_ANYWAY. Classify it: "
                "either exclude it from the filter, or record why a scheduled commit "
                "should redeploy the service."
            )
