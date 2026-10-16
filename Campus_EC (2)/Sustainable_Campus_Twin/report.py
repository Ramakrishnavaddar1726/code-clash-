"""PDF sustainability report (ReportLab)."""
import io
import re
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _md(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(text))


def build_pdf(issues: list, kpis: dict, ai_text: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    st = getSampleStyleSheet()
    body = []
    body.append(Paragraph("AI-Powered Sustainable Campus - Report", st["Title"]))
    body.append(Paragraph(datetime.now().strftime("Generated %d %b %Y, %H:%M"), st["Normal"]))
    body.append(Spacer(1, 6 * mm))

    kt = [["Metric", "Value"],
          ["Energy (24h)", f"{kpis['energy_kwh_24h']:,} kWh"],
          ["Water (24h)", f"{kpis['water_kl_24h']} kL"],
          ["CO2 from energy (24h)", f"{kpis['co2_kg_24h']:,} kg"],
          ["Active issues", str(kpis["active_issues"])],
          ["Recurring loss if unresolved", f"INR {kpis['monthly_loss_inr']:,} / month"],
          ["Avoidable CO2", f"{kpis['monthly_co2_kg']:,} kg / month"]]
    t = Table(kt, colWidths=[75 * mm, 80 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f7a4d")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white])]))
    body += [t, Spacer(1, 6 * mm)]

    body.append(Paragraph("Detected issues", st["Heading2"]))
    if issues:
        rows = [["Building", "Issue", "Severity", "Since", "INR/month"]]
        for i in issues:
            rows.append([i["building"], i["type"], i["severity"].upper(),
                         i["since"][5:16], f"{i['monthly_cost_inr']:,}"])
        it = Table(rows, repeatRows=1)
        it.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        body.append(it)
    else:
        body.append(Paragraph("No active issues.", st["Normal"]))
    body.append(Spacer(1, 6 * mm))

    body.append(Paragraph("AI analysis and recommendations", st["Heading2"]))
    for line in ai_text.splitlines():
        s = line.strip()
        if not s:
            body.append(Spacer(1, 2 * mm))
        elif s.startswith("#"):
            body.append(Paragraph(_md(s.lstrip("# ")), st["Heading3"]))
        elif s.startswith(("- ", "* ")):
            body.append(Paragraph("&bull; " + _md(s[2:]), st["Normal"]))
        else:
            body.append(Paragraph(_md(s), st["Normal"]))
    doc.build(body)
    return buf.getvalue()
