# scroll-journey

Personal portfolio piece: one photoreal view from a cabin porch over a misty mountain valley. Scrolling scrubs pre-rendered Blender frames on a canvas and moves time from golden hour through dusk to stars; a car's headlights wind along the valley road at dusk and steam rises from a mug on the rail. Text and steam run live on top.

## Stack
- Blender 5.2 LTS, driven only by Python scripts run headless: `blender -b -P blender\<script>.py`. If `blender` isn't on PATH, call it by full path under `C:\Program Files\Blender Foundation\` (don't edit PATH). Cycles on the RTX 4080 (OptiX).
- Blender exits with code 0 even when a script fails: check the log for `Traceback`.
- Frames: numbered WebP. A ~1440p 16:9 desktop set plus a separate portrait phone set. Not 4K.
- Site: Vite + TypeScript + GSAP (ScrollTrigger), exact versions pinned. No framework and no live 3D (Three.js). No sound for now.
- Windows + PowerShell 7. Use `py`, not `python`.
- Priority: smoothness and visual quality first, then download size.

## Folders
- `assets-source/` raw downloads (car .glb, HDRIs, textures, models). Not committed.
- `assets-source/reference/` reference photos for comparison only. They never ship.
- `blender/` scene scripts, safe to re-run. `renders/` raw output, timing logs, review stills. Not committed.
- `public/frames/<scene>/` web-ready frames the site loads. Generated, not committed.
- `docs/` plan and handoff files. Local only, via `.git/info/exclude`.
- Attempt 1's 03 work and the 2D canvas fallback are parked on the git branch `parked-03-and-2d`. Don't merge it.

## Rules
- Approved sources: Poly Haven (CC0; official site and API only) and the Lionsharp 911 .glb. Any other asset, npm package, Blender add-on or service needs Nems's OK after a license, source and security screen.
- Never open or import `.blend` files from the internet; take Poly Haven models as glTF or FBX. `.blend` files our own scripts write (cached light trees under `assets-source/`, never downloaded or committed) are fine. Never enable "Auto Run Python Scripts" or pass `-y` / `--enable-autoexec`. No Blender MCP servers or AI bridge add-ons.
- Branding: no Porsche logos or wordmarks in the site UI; the car itself can appear as-is in renders. The footer keeps the credit below and "Not affiliated with Porsche. Personal portfolio project." Client or commercial reuse means swapping the car first.
- All JS bundled via npm: no CDN scripts, no analytics.
- Respect `prefers-reduced-motion`.
- Render budget: use the lightest model that looks the same at its size in the frame. Full-detail trees and props only right next to the camera; light versions (made once, saved beside the originals) everywhere else. Aim for peak RAM under 12 GB, and say in every report, in plain words, how much RAM and time a render took and what used most of it.

## Workflow
- Tasks arrive as `docs/handoff/NN-name.md`. Nems runs all git and the manual tests.
- Nems is the art director. Visual calls: save labelled stills to `renders/review/NN/`, list them in the report, and leave the pick to him. Technical calls: make them and explain why in the report.
- Reports in plain language for someone new to 3D and front-end animation.

## Credits
"FREE 1975 Porsche 911 (930) Turbo" by Lionsharp Studios (https://sketchfab.com/3d-models/free-1975-porsche-911-930-turbo-8568d9d14a994b9cae59499f0dbed21e), licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), modified. HDRIs, textures and models: Poly Haven (CC0).
