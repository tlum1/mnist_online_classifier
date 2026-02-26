const API_BASE = ""; // если web отдаётся FastAPI на том же порту
const PREDICT_URL = `${API_BASE}/predict`;

export async function predictFromCanvas(canvas) {
  const blob = await new Promise(res => canvas.toBlob(res, "image/png"));
  const form = new FormData();
  form.append("file", blob, "digit.png");

  const r = await fetch(PREDICT_URL, { method: "POST", body: form });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const j = await r.json();

  if (!j.probs || j.probs.length !== 10) throw new Error("Bad response: probs");
  return j;
}