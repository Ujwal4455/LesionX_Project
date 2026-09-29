from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_report(path, patient_id, predictions=None):
    c = canvas.Canvas(path, pagesize=letter)
    c.drawString(50, 750, "LesionX Report")
    c.drawString(50, 730, f"Patient ID: {patient_id}")
    c.drawString(50, 710, "LesionX is an academic research and decision-support prototype. Its predictions are not confirmed medical diagnoses and should not replace professional medical evaluation.")
    c.drawString(50, 690, f"Predictions: {predictions if predictions is not None else 'not available'}")
    c.save()
