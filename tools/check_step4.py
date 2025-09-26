from pathlib import Path
import json
import glob

root = Path(".")
reports = sorted(glob.glob("models/trained_models/*_report.json"))
if not reports:
    print("No hay reportes en models/trained_models/")
else:
    for rp in reports:
        with open(rp, "r", encoding="utf-8") as f:
            js = json.load(f)
        print("\n", rp)
        print("  model:", js["model"], "| calib:", js["calibration"])
        print("  valid:", js["valid"])
        print("  test :", js["test"])

sumfile = root / "models/trained_models/summary_step4.json"
if sumfile.exists():
    print("\nResumen ranking:")
    print(sumfile.read_text(encoding="utf-8"))
