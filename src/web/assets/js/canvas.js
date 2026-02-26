export function setupCanvas(canvas, brushInput, brushValEl, onStroke) {
  const ctx = canvas.getContext("2d");

  function reset() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  reset();

  ctx.lineCap = "round";
  ctx.strokeStyle = "white";
  ctx.lineWidth = Number(brushInput.value);

  brushInput.addEventListener("input", () => {
    ctx.lineWidth = Number(brushInput.value);
    brushValEl.textContent = brushInput.value;
  });

  let drawing = false;
  let last = null;

  function pos(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  canvas.addEventListener("pointerdown", (e) => {
    drawing = true;
    last = pos(e);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const p = pos(e);
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    last = p;
    onStroke?.("move");
  });

  function stop() {
    if (!drawing) return;
    drawing = false;
    last = null;
    onStroke?.("end");
  }

  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointerleave", stop);

  return { reset };
}