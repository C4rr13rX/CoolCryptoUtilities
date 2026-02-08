export type EdgeAutoScrollOptions = {
  edgePx?: number;
  delayMs?: number;
  maxSpeedPxPerSec?: number;
  minSpeedPxPerSec?: number;
  stopMoveThresholdPx?: number;
};

type ScrollDirection = -1 | 0 | 1;

export function attachEdgeAutoScroll(element: HTMLElement, options: EdgeAutoScrollOptions = {}) {
  const edgePx = options.edgePx ?? 28;
  const delayMs = options.delayMs ?? 500;
  const maxSpeed = options.maxSpeedPxPerSec ?? 520;
  const minSpeed = options.minSpeedPxPerSec ?? 120;
  const stopMoveThreshold = options.stopMoveThresholdPx ?? 2;

  let dwellTimer: number | undefined;
  let rafId: number | undefined;
  let scrolling = false;
  let pointerInside = false;
  let pointerX = 0;
  let pointerY = 0;
  let lastMoveX = 0;
  let lastMoveY = 0;
  let lastFrame = 0;

  const clearDwell = () => {
    if (dwellTimer) {
      window.clearTimeout(dwellTimer);
      dwellTimer = undefined;
    }
  };

  const stopScroll = () => {
    scrolling = false;
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = undefined;
    }
  };

  const getDirection = (): ScrollDirection => {
    if (!pointerInside) return 0;
    const rect = element.getBoundingClientRect();
    if (pointerX < rect.left || pointerX > rect.right || pointerY < rect.top || pointerY > rect.bottom) {
      return 0;
    }
    const topDist = pointerY - rect.top;
    const bottomDist = rect.bottom - pointerY;
    if (topDist <= edgePx) return -1;
    if (bottomDist <= edgePx) return 1;
    return 0;
  };

  const computeSpeed = (direction: ScrollDirection) => {
    const rect = element.getBoundingClientRect();
    const distance = direction < 0 ? pointerY - rect.top : rect.bottom - pointerY;
    const clamped = Math.max(0, Math.min(edgePx, distance));
    const factor = 1 - clamped / edgePx;
    return minSpeed + (maxSpeed - minSpeed) * factor;
  };

  const step = (timestamp: number) => {
    if (!scrolling) return;
    const direction = getDirection();
    if (direction === 0) {
      stopScroll();
      return;
    }
    if (!lastFrame) lastFrame = timestamp;
    const delta = (timestamp - lastFrame) / 1000;
    lastFrame = timestamp;
    const speed = computeSpeed(direction);
    const deltaPx = direction * speed * delta;
    const prev = element.scrollTop;
    element.scrollTop += deltaPx;
    if (
      (direction < 0 && (element.scrollTop <= 0 || element.scrollTop === prev)) ||
      (direction > 0 && element.scrollTop + element.clientHeight >= element.scrollHeight - 1)
    ) {
      stopScroll();
      return;
    }
    rafId = requestAnimationFrame(step);
  };

  const scheduleScroll = () => {
    clearDwell();
    const direction = getDirection();
    if (direction === 0) return;
    dwellTimer = window.setTimeout(() => {
      if (getDirection() === 0) return;
      if (element.scrollHeight <= element.clientHeight) return;
      scrolling = true;
      lastFrame = 0;
      rafId = requestAnimationFrame(step);
    }, delayMs);
  };

  const handlePointerMove = (event: PointerEvent) => {
    const moved =
      Math.abs(event.clientX - lastMoveX) > stopMoveThreshold ||
      Math.abs(event.clientY - lastMoveY) > stopMoveThreshold;
    lastMoveX = event.clientX;
    lastMoveY = event.clientY;
    pointerX = event.clientX;
    pointerY = event.clientY;
    pointerInside = true;
    if (scrolling && moved) {
      stopScroll();
    }
    const direction = getDirection();
    if (direction === 0) {
      clearDwell();
      return;
    }
    scheduleScroll();
  };

  const handlePointerLeave = () => {
    pointerInside = false;
    clearDwell();
    stopScroll();
  };

  const handleInterrupt = () => {
    clearDwell();
    stopScroll();
  };

  element.addEventListener('pointermove', handlePointerMove);
  element.addEventListener('pointerleave', handlePointerLeave);
  element.addEventListener('pointerdown', handleInterrupt);
  element.addEventListener('wheel', handleInterrupt, { passive: true });

  return () => {
    clearDwell();
    stopScroll();
    element.removeEventListener('pointermove', handlePointerMove);
    element.removeEventListener('pointerleave', handlePointerLeave);
    element.removeEventListener('pointerdown', handleInterrupt);
    element.removeEventListener('wheel', handleInterrupt);
  };
}
