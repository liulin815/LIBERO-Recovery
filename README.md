# LIBERO-Recover — Project Page (v2)

Static showcase website for the ICLR 2027 submission
**"LIBERO-Recover: Beyond Task Success Towards Failure Recovery in Robotic
Manipulation Models"** (source: `../iclr2027_conference.tex`).

v2 restyles the page after [imitator-game.github.io](https://imitator-game.github.io/):
Apple-ish off-white canvas + white cards + blue `#0071e3` accent, Noto Sans,
a scrolling **video-reel hero**, a **tabbed L1–L4 level widget**, **flip
suite cards**, and a **figure-deck carousel** for the results figures.

## View locally

Fully static — no build step, no dependencies:

```bash
cd /H20_vepfs/liulin/scene_viz/project_page
python3 -m http.server 8399        # or any static server / just open index.html
# open http://localhost:8399
```

Total payload ≈ 9.5 MB (figures ≈ 4 MB + videos ≈ 2 MB with lazy metadata preload;
the hero reel videos start only when the hero is on screen).

## Publish at `https://<user>.github.io/<repo>/`

1. Create a GitHub repository, e.g. `libero-recover` (public).
2. Push the **contents** of this folder (not the folder itself) to `main`:
   ```bash
   cd /H20_vepfs/liulin/scene_viz/project_page
   git init -b main
   git add . && git commit -m "project page"
   git remote add origin git@github.com:<user>/libero-recover.git
   git push -u origin main
   ```
3. Repo → **Settings → Pages** → Source: *Deploy from a branch* →
   Branch: `main` / `/ (root)` → Save.
4. 1–2 min later the site is live at `https://<user>.github.io/libero-recover/`.
   - Prefer a root URL `https://<user>.github.io/`? Name the repo exactly
     `<user>.github.io` and push to `main` — no Settings change needed.
   - All asset paths are relative (`assets/…`, `css/…`, `js/…`), so the site
     works under both URL shapes unchanged.
5. Update the nav GitHub button / hero CTAs in `index.html` once the repo,
   PDF, and dataset links exist.

## Structure

```
project_page/
├── index.html          # single-page site, all content from the paper
├── css/style.css       # Apple-style tokens + layout (light theme, validated level palette)
├── js/main.js          # reel, tabs, flip cards, figdeck, videos, filter, nav, BibTeX
└── assets/
    ├── figures/        # copied from scene_viz/ (+ chunk sweep, benchmark gantt)
    └── videos/
        ├── levels/     # L1–L4 difficulty demos (H.264, from scene_viz/videos)
        └── demos/      # 12 sample recovery episodes from lerobot_mydata
```

## Section → source map

| Site section | Source |
|---|---|
| Hero stats / abstract / findings 1–8 | `../iclr2027_conference.tex` |
| Hero reel videos | `assets/videos/{demos,levels}` |
| Taxonomy figure, heatmap, radar, degradation, RC, action-density | `../*.png` |
| Chunk sweep figure | `/H20_vepfs/liulin/chunk_sweep_combined_1x4.png` |
| Benchmark timeline | `/H20_vepfs/liulin/benchmark_timeline/benchmark_gantt.png` |
| L1–L4 videos | `../videos/L{1..4}.mp4` (transcoded AV1→H.264) |
| Demo episodes + posters | `starVLA/datasets/LEROBOT_LIBERO_DATA/lerobot_mydata/` |
| Task-type montage | `lerobot_mydata/first_frames/task_overview.jpg` |

## Before publishing — edit these in `index.html`

1. **Authors** — hero block currently reads "Anonymous Author(s)".
2. **Buttons** — Paper / Code / Dataset hrefs are `#` with a "soon" tag.
3. **BibTeX** — `#bibtex` block uses the anonymous placeholder.

## Regenerating assets after data changes

```bash
# re-transcode a demo episode (chunk = episode_index // 1000)
ffmpeg -i <lerobot_mydata>/videos/chunk-000/observation.images.image/episode_000000.mp4 \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart -an assets/videos/demos/ep_000000.mp4
cp <lerobot_mydata>/first_frames/image/episode_000000.jpg assets/videos/demos/
```

## Design notes

- Light-only theme: every embedded paper figure renders on a white surface.
- Recovery-level colors (L1 blue `#3577c9`, L2 green `#2f9e6e`,
  L3 orange `#e08a1e`, L4 red `#b03030`) are a CVD-validated 4-slot palette;
  level identity is never carried by color alone — every badge keeps its
  text label ("L1 · Action Retry").
- Hero reel: two marquee rows (opposite directions), tiles doubled in JS for
  a seamless `-50%` loop; videos start only while the hero is on screen and
  pause below the fold. `prefers-reduced-motion` freezes the reel.
- Level widget: tabs set `--lvl-c` on the stage; badge, stat and dist-cells
  all inherit the level color; the four videos are stacked and toggled.
- Flip suite cards: hover flips (desktop), click flips (touch); the front
  video keeps hover-to-play / click-to-pin.
- Figure deck: 8 panels (findings 1–5, 8 + dataset scale + benchmark
  timeline) with dot rail, arrows, counter, and arrow-key support.
- Videos: muted, looped, hover-to-play (click to pin), pause automatically
  when scrolled out of view; `prefers-reduced-motion` disables hover autoplay.
