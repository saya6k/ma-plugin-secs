"""
Make tests/fixtures/stereo.secs: the L and R of the SECS 2R sample room as a 2.0 project.

The engine's sample room is 7.1.4 with three subwoofers, and changing the layout of a project
clears its measurements, so this keeps only the L and R speakers and measurements of a sample.

    uv run python scripts/make_stereo_fixture.py
"""

import json
import shutil
import tempfile
from pathlib import Path

import secs2r

FIXTURES = Path(__file__).parents[1] / "tests" / "fixtures"
KEEP = {"L", "R"}
POSITIONS = 2


def main() -> None:
    """Write the fixture."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        engine = secs2r.Engine(work / "settings.json")
        sample = work / "room.secs"
        engine.call(
            {
                "method": "project_new",
                "params": {"path": str(sample), "preset": "2.0", "sub_count": 0, "sample": True},
            }
        )
        job = engine.call({"method": "sample_load", "params": {"positions": POSITIONS}})
        assert engine.wait_job(job["data"]["job"], 600)["status"] == "succeeded"
        engine.call({"method": "project_save", "params": {}})

        project = json.loads(sample.read_text(encoding="utf-8"))
        project["name"] = "stereo"
        speakers = project["config"]["speakers"]
        project["config"]["speakers"] = [s for s in speakers if s["role"] in KEEP]
        for measurement in project["measurements"]:
            for key, value in measurement.items():
                if isinstance(value, dict) and "L" in value:
                    measurement[key] = {r: v for r, v in value.items() if r in KEEP}
            measurement["ir"] = {
                role: path.replace("room.irs/", "stereo.irs/")
                for role, path in measurement["ir"].items()
            }

        irs = FIXTURES / "stereo.irs"
        shutil.rmtree(irs, ignore_errors=True)
        irs.mkdir(parents=True)
        for role in KEEP:
            for position in range(1, POSITIONS + 1):
                name = f"{role}_P{position}.wav"
                shutil.copy(work / "room.irs" / name, irs / name)
        (FIXTURES / "stereo.secs").write_text(
            json.dumps(project, ensure_ascii=False, indent=1), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
