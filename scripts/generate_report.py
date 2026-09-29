from src.reporting.report_generator import generate_report


if __name__ == "__main__":
    generate_report("reports/generated/sample_report.pdf", "demo_patient", {"lesion_risk": 0.5})
    print("Sample report created.")
