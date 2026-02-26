export function initBars(barsEl) {
  const barFills = [];
  const pctEls = [];

  for (let d = 0; d < 10; d++) {
    const row = document.createElement("div");
    row.className = "barRow";

    const digit = document.createElement("div");
    digit.className = "digit";
    digit.textContent = d;

    const track = document.createElement("div");
    track.className = "barTrack";
    const fill = document.createElement("div");
    fill.className = "barFill";
    track.appendChild(fill);

    const pct = document.createElement("div");
    pct.className = "pct";
    pct.textContent = "0.0%";

    row.appendChild(digit);
    row.appendChild(track);
    row.appendChild(pct);

    barsEl.appendChild(row);

    barFills.push(fill);
    pctEls.push(pct);
  }

  return { barFills, pctEls };
}

export function setStatus(kind, text) {
  const dot = document.getElementById("dot");
  const statusText = document.getElementById("statusText");
  dot.classList.remove("ok", "bad");
  if (kind === "ok") dot.classList.add("ok");
  if (kind === "bad") dot.classList.add("bad");
  statusText.textContent = text;
}

export function renderProbs(probs, barFills, pctEls) {
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;

  document.getElementById("pred").textContent = best;
  document.getElementById("predInfo").textContent = `уверенность ${(probs[best]*100).toFixed(1)}%`;

  for (let i = 0; i < 10; i++) {
    const p = probs[i] || 0;
    const w = Math.max(0, Math.min(1, p)) * 100;
    barFills[i].style.width = `${w}%`;
    pctEls[i].textContent = `${w.toFixed(1)}%`;
    barFills[i].style.opacity = (i === best) ? "1" : "0.55";
  }
}

export function resetPrediction(barFills, pctEls) {
  document.getElementById("pred").textContent = "–";
  document.getElementById("predInfo").textContent = "Нарисуй что-нибудь";
  renderProbs(new Array(10).fill(0), barFills, pctEls);
  setStatus("", "ожидание");
}