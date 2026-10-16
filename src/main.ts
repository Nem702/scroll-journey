import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const FRAME_COUNT = 150;
const STILL_FRAME = 75; // the one frame shown when reduced motion is on
const MAX_DPR = 2; // the source is 1440 px tall, so sharper backing stores add cost, not detail

const canvas = document.querySelector<HTMLCanvasElement>('#scene')!;
const ctx = canvas.getContext('2d', { alpha: false })!;
const loader = document.querySelector<HTMLElement>('#loader')!;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

let frames: HTMLImageElement[] = [];
let current = -1;
const recordDraw = new URLSearchParams(location.search).has('stats') ? startStats() : null;

function frameUrl(n: number): string {
  return `frames/proof/${String(n).padStart(4, '0')}.webp`;
}

async function loadFrame(n: number): Promise<HTMLImageElement> {
  const img = new Image();
  img.src = frameUrl(n);
  await img.decode();
  return img;
}

// Draw a frame scaled to cover the whole canvas, like CSS object-fit: cover.
function draw(index: number): void {
  const img = frames[index];
  const scale = Math.max(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
  const sw = canvas.width / scale;
  const sh = canvas.height / scale;
  const start = performance.now();
  ctx.drawImage(img, (img.naturalWidth - sw) / 2, (img.naturalHeight - sh) / 2, sw, sh, 0, 0, canvas.width, canvas.height);
  recordDraw?.(performance.now() - start);
}

function show(index: number): void {
  if (index === current) return;
  current = index;
  draw(index);
}

function resize(): void {
  const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
  const width = Math.round(canvas.clientWidth * dpr);
  const height = Math.round(canvas.clientHeight * dpr);
  if (width === canvas.width && height === canvas.height) return;
  canvas.width = width; // resizing clears the canvas, so redraw straight away
  canvas.height = height;
  if (current >= 0) draw(current);
}

async function start(): Promise<void> {
  const numbers = reducedMotion ? [STILL_FRAME] : Array.from({ length: FRAME_COUNT }, (_, i) => i + 1);
  let loaded = 0;
  let failed = false;
  try {
    frames = await Promise.all(
      numbers.map(async (n) => {
        const img = await loadFrame(n);
        loaded++;
        if (!failed) loader.textContent = `Loading ${Math.round((loaded / numbers.length) * 100)}%`;
        return img;
      }),
    );
  } catch {
    failed = true;
    loader.textContent = 'Could not load the frames. Please reload.';
    return;
  }
  loader.hidden = true;
  resize();
  window.addEventListener('resize', resize);

  if (reducedMotion) {
    show(0);
    return;
  }
  const trigger = ScrollTrigger.create({
    trigger: '#scrub',
    start: 'top top',
    end: 'bottom bottom',
    onUpdate: (self) => show(Math.round(self.progress * (FRAME_COUNT - 1))),
  });
  show(Math.round(trigger.progress * (FRAME_COUNT - 1)));
}

// ?stats: on-screen timing numbers, for measuring smoothness on a phone without dev tools.
// "draw" is the drawImage call itself; "frame gap" is the time between screen updates while
// scrubbing, which also catches decode work the browser defers until after drawImage returns.
function startStats(): (drawMs: number) => void {
  const box = document.createElement('div');
  box.id = 'stats';
  document.body.append(box);
  let drawLast = 0;
  let drawWorst = 0;
  let lastDrawAt = -Infinity;
  let gapWorst = 0;
  let slowFrames = 0;
  let previous = performance.now();

  const tick = (now: number) => {
    const gap = now - previous;
    previous = now;
    if (now - lastDrawAt < 250) {
      gapWorst = Math.max(gapWorst, gap);
      if (gap > 25) slowFrames++; // at least one missed frame on a 60 Hz screen
    }
    const text =
      `frame ${current + 1}/${FRAME_COUNT} | draw ${drawLast.toFixed(1)} ms (worst ${drawWorst.toFixed(1)}) | ` +
      `frame gap worst ${gapWorst.toFixed(0)} ms | slow frames ${slowFrames} | canvas ${canvas.width}x${canvas.height}`;
    if (text !== box.textContent) box.textContent = text;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  return (drawMs) => {
    drawLast = drawMs;
    drawWorst = Math.max(drawWorst, drawMs);
    lastDrawAt = performance.now();
  };
}

void start();
