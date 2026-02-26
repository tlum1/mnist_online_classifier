import { throttleTrailing } from "./throttle.js";
import { predictFromCanvas } from "./api.js";
import { initBars, renderProbs, resetPrediction, setStatus } from "./ui.js";
import { setupCanvas } from "./canvas.js";

const canvas = document.getElementById("c");
const brush = document.getElementById("brush");
const brushVal = document.getElementById("brushVal");
const barsEl = document.getElementById("bars");

const { barFills, pctEls } = initBars(barsEl);
resetPrediction(barFills, pctEls);

let inflight = false;

async function doPredict() {
  if (inflight) return;
  inflight = true;

  try {
    setStatus("", "предикт...");
    const j = await predictFromCanvas(canvas);
    renderProbs(j.probs, barFills, pctEls);
    setStatus("ok", "ok");
  } catch (e) {
    console.error(e);
    setStatus("bad", "ошибка запроса");
  } finally {
    inflight = false;
  }
}

const doPredictThrottled = throttleTrailing(doPredict, 120);

const { reset } = setupCanvas(canvas, brush, brushVal, (evt) => {
  if (evt === "move") doPredictThrottled();
  if (evt === "end") doPredict();
});

document.getElementById("clear").onclick = () => {
  reset();
  resetPrediction(barFills, pctEls);
};

document.getElementById("predict").onclick = () => {
  doPredict();
};