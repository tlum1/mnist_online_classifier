export function throttleTrailing(fn, intervalMs = 120) {
  let lastTs = 0;
  let timer = null;
  let pendingArgs = null;

  return (...args) => {
    const now = Date.now();
    const dt = now - lastTs;

    if (dt >= intervalMs) {
      lastTs = now;
      fn(...args);
      return;
    }

    pendingArgs = args;
    if (timer) return;

    timer = setTimeout(() => {
      timer = null;
      lastTs = Date.now();
      fn(...pendingArgs);
      pendingArgs = null;
    }, intervalMs - dt);
  };
}