# bassalt

A lava lamp for Resolume Arena/Avenue, as an FFGL effect: a heat engine (bulb,
wax, salted water) with Cahn-Hilliard wax, advected heat and Hele-Shaw flow,
and the clip pushed through it. C++/GLSL, CMake MODULE -> universal `.bundle`
(macOS) + Windows `.dll`. MIT. ID `BS01`, display name `SW Bassalt`.

Read `AGENTS.md` before touching the circulation, the multigrid, the update or
anything marked `precise`.

## Commands (CMake)
- Configure: `cmake -B build -DCMAKE_BUILD_TYPE=Release`
- Fast dev build: add `-DCMAKE_OSX_ARCHITECTURES=arm64`
- Build: `cmake --build build`
- Install to Resolume: `cmake --install build` (NOT from a shared checkout)
- Render offline: `./build/bstest --out /tmp/frame.png --frames 600 --set "Speed=0.6"`
- Press buttons: `--warm N`, `--reset N`, `--pour N` press on frame N
- The card on its own: `./build/bstest --card /tmp/card.png`
- List parameters: `./build/bstest --list`
- Set anything by name: `./build/bstest --set "Clip=1" --set "Glass=1"`
- Watch the state: `./build/bstest --probe 60 --frames 600`
- Film: `./build/bstest --film 600 --size 960x540 --script docs/demo.cues | ffmpeg -f rawvideo -pix_fmt rgba -s 960x540 -r 60 -i - out.mp4`
- Film a clip through it: `ffmpeg -i in.mov -f rawvideo -pix_fmt rgba - | ./build/bstest --pipe --size WxH [--script cues] | ffmpeg ...`
- Dump what a check sets up: `--dump DIR` with a check (development aid)

## Verify
- Everything: `tools/verify.sh` (~3 min: fresh universal build, every check, the
  negative controls, the mutation, the sweep, lipo, plist, names, ad-hoc
  signature, `oxbow probe`, the bench). Do not edit it while it runs.
- **Rest**: `--still` (a level slab's psi exactly 0; a neutral blob's currents
  bounded and falling). **Conservation**: `--volume` (wax to six random walks
  of rounding; the local maximum principle for heat, every cell every step).
- **The laws**: `--crossover` (T* and its salt slope), `--darcy` (the
  depolarisation speed, and the GPU solve against an exact CPU solve of its own
  system), `--rt` (Rayleigh-Taylor growth and cutoff), `--diffusion`, `--heat`
  (the lumped law and the joule budget), `--bulb` (63% at tau; primed onset).
- **The solve**: `--multigrid`. **The optics**: `--glass` (two rasters),
  `--lens` (a round blob is a sphere's lens). **The host**: `--state`.
- **The checks can fail**: `--negative` (14 wrong models), `--mutate` (one
  character of the shipped GLSL).
- No dead controls: `python3 tools/sweep.py` (30 live; five need the context
  table: audio, a raised bass, a pour).
- Render cost: `./build/bstest --bench`.

## Notes
- **psi, not pressure**: the velocity is divergence-free by construction, the
  pressure's circulation is identically zero, and the walls are Dirichlet.
- **`precise` is load-bearing.** This GPU compiler reassociates float
  arithmetic; the circulation and every flux are `precise`, or a level slab
  moves and wax is not conserved to the bit.
- **Operator-dependent prolongation**, not bilinear: bilinear diverged on the
  cold slab's 10^5 jump.
- **The coarsest grid is at most 8 x 8 nodes**, solved whole in every fragment.
  Bigger costs a millisecond a call (private memory).
- **The step**: heat limit, capillary limit (kCapillary = 5, measured), Courant
  1/4 on last frame's fastest face x 1.5 (read back a frame late, no stall);
  if the flow outruns it, ALL of psi is scaled -- never one face.
- **Cahn-Hilliard is subcycled** in its own pass; its mobility is a numerical
  device sized so a resting blob's currents die.
- Van Leer on phi and T. Superbee made spurious currents permanent.
- All host parameters are 0..1 and mapped in `Controls.cpp`; option parameters
  hold the element value; events act on the rising edge.
- Names at most 16 characters (the FFGL field is not null-terminated);
  `verify.sh` checks.
- GLSL reserved words must not be identifiers; `verify.sh` greps for them.
- Override `SetTextParameter` to return `FF_SUCCESS` for the About block.
- `bassalt_core` is an OBJECT library: the plugin registers itself from a
  file-scope constructor nothing references.
- The macOS build must be universal. Check with `lipo`, never the build log.
- Local repo only: no GitHub remote, no tag, not registered on the website.

## Not done yet
- Never loaded into Resolume (oxbow probe only). No OFX port, browser demo,
  factory presets or rocket mask. Never built on Windows.
- `StoatworksAbout.h` and `ATTRIBUTIONS.md` are provisional hand copies.

## Diagnostics

`source/Diag.{h,cpp}` is a log file only, with no crash handler (this runs
inside Resolume): which shader failed to compile, the GL vendor/renderer, the
host clock's unit once voted on.

    ~/Library/Logs/bassalt/bassalt.YYYY-MM-DD.log
