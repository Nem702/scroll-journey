import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const FRAME_COUNT = 69;
const STILL_FRAME = 1; // k1, golden hour: the one frame shown when reduced motion is on

const params = new URLSearchParams(location.search);
const frameSet = pickSet(params.get('size'));
const blend = params.get('blend') !== '0'; // ?blend=0 shows the nearest frame only, for comparison

// Two stacked canvases. Even frames always draw on the bottom one and odd frames on the top one, so a step to the next
// frame redraws only one of them. Between two frames only the top one's opacity changes, which the browser composites
// without redrawing.
const canvases = [...document.querySelectorAll<HTMLCanvasElement>('#scene canvas')];
const contexts = canvases.map((canvas) => canvas.getContext('2d', { alpha: false })!);
const top = canvases[1];
const loader = document.querySelector<HTMLElement>('#loader')!;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

let frames: HTMLImageElement[] = [];
const held = [-1, -1]; // the frame each canvas holds (an index into frames)
let position = 0; // the scroll position in frames: 0 is the first, FRAME_COUNT - 1 the last
let topOpacity = -1;
const stats = params.has('stats') ? startStats() : null;

// 1440 frames only where they can show: the window's longer side is over 1920 device pixels, counting at most 2 per
// CSS pixel. Phones and 1080p screens get 1080.
function pickSet(asked: string | null): '1440' | '1080' {
  if (asked === '1440' || asked === '1080') return asked;
  const longer = Math.max(window.innerWidth, window.innerHeight) * Math.min(window.devicePixelRatio || 1, 2);
  return longer > 1920 ? '1440' : '1080';
}

function frameUrl(n: number): string {
  return `frames/time/${frameSet}/${String(n).padStart(4, '0')}.webp`;
}

async function loadFrame(n: number): Promise<HTMLImageElement> {
  const img = new Image();
  img.src = frameUrl(n);
  await img.decode();
  return img;
}

// Draw a frame scaled to cover the whole canvas, like CSS object-fit: cover.
function draw(layer: number, index: number): void {
  const canvas = canvases[layer];
  const img = frames[index];
  const scale = Math.max(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
  const sw = canvas.width / scale;
  const sh = canvas.height / scale;
  const start = performance.now();
  contexts[layer].drawImage(img, (img.naturalWidth - sw) / 2, (img.naturalHeight - sh) / 2, sw, sh, 0, 0, canvas.width, canvas.height);
  stats?.drew(performance.now() - start);
}

// Put a frame on its canvas, unless it's there already.
function hold(index: number): void {
  const layer = index % 2;
  if (held[layer] === index) return;
  held[layer] = index;
  draw(layer, index);
}

// Show a scroll position (in frames): the two nearest frames crossfaded, or with ?blend=0 the nearest one.
function show(at: number): void {
  position = at;
  let opacity: number;
  if (blend && frames.length > 1) {
    const i = Math.min(Math.floor(at), frames.length - 2);
    const t = Math.round((at - i) * 1000) / 1000; // 0 at frame i, 1 at frame i + 1
    hold(i);
    hold(i + 1);
    opacity = i % 2 === 0 ? t : 1 - t; // the top canvas holds the odd one of the two
  } else {
    const i = Math.round(at);
    hold(i);
    opacity = i % 2;
  }
  if (opacity !== topOpacity) {
    topOpacity = opacity;
    top.style.opacity = String(opacity);
  }
  stats?.moved();
}

// Size the canvases to the screen, but never with more pixels than the frame shows there (in place of a fixed cap on
// the pixel ratio). An upright phone shows about 540x1080 pixels of a 1920x1080 frame.
function resize(): void {
  const img = frames[0];
  const cssWidth = canvases[0].clientWidth;
  const cssHeight = canvases[0].clientHeight;
  const cover = Math.max(cssWidth / img.naturalWidth, cssHeight / img.naturalHeight); // CSS px per frame px
  const ratio = Math.min(window.devicePixelRatio || 1, 1 / cover);
  const width = Math.round(cssWidth * ratio);
  const height = Math.round(cssHeight * ratio);
  if (width === canvases[0].width && height === canvases[0].height) return;
  canvases.forEach((canvas, layer) => {
    canvas.width = width; // resizing clears the canvas, so redraw straight away
    canvas.height = height;
    if (held[layer] >= 0) draw(layer, held[layer]);
  });
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
    onUpdate: (self) => show(self.progress * (FRAME_COUNT - 1)),
  });
  show(trigger.progress * (FRAME_COUNT - 1));
}

// ?stats: on-screen timing numbers, for measuring smoothness on a phone without dev tools.
// "draw" is one drawImage call; "frame gap" is the time between screen updates while scrubbing, which also catches
// decode work the browser defers until after drawImage returns.
function startStats(): { drew: (ms: number) => void; moved: () => void } {
  const box = document.createElement('div');
  box.id = 'stats';
  document.body.append(box);
  let drawLast = 0;
  let drawWorst = 0;
  let lastMoveAt = -Infinity;
  let gapWorst = 0;
  let slowFrames = 0;
  let previous = performance.now();

  const tick = (now: number) => {
    const gap = now - previous;
    previous = now;
    if (now - lastMoveAt < 250) {
      gapWorst = Math.max(gapWorst, gap);
      if (gap > 25) slowFrames++; // at least one missed frame on a 60 Hz screen
    }
    const text =
      `frame ${(position + 1).toFixed(2)}/${FRAME_COUNT} | set ${frameSet} | blend ${blend ? 'on' : 'off'} | ` +
      `draw ${drawLast.toFixed(1)} ms (worst ${drawWorst.toFixed(1)}) | frame gap worst ${gapWorst.toFixed(0)} ms | ` +
      `slow frames ${slowFrames} | canvas ${canvases[0].width}x${canvases[0].height}`;
    if (text !== box.textContent) box.textContent = text;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  return {
    drew: (ms) => {
      drawLast = ms;
      drawWorst = Math.max(drawWorst, ms);
    },
    moved: () => {
      lastMoveAt = performance.now();
    },
  };
}

void start();
