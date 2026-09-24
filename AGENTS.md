# bassalt — for agents

The why behind the code. `CLAUDE.md` has the commands; this file has the
reasoning, the traps that were actually hit, and what is and is not known.
Built 2026-09-23 in one session, from Allan's request: "bassalt, a new
resolume plugin that emulates a lava lamp". The spec is
`~/Projects/resolume/specs/SPEC-bassalt.md`.

## The one idea

The frame is a lava lamp, and the lamp is a heat engine. A bulb heats wax that
is a little denser than the water when cold; wax expands about three times as
fast as the water does, so above a crossover temperature T* it is the lighter,
rises, cools at the cap, sinks. Nothing is animated. Four fields on a grid in
metres, with their constants in `Physics.h`:

- **phi**, the wax fraction: Cahn-Hilliard, in flux form.
- **T**, the temperature: advected in flux form, conducted, heated through the
  base by the bulb, cooled through the two glass faces and harder through the
  cap.
- **psi**, the streamfunction of Darcy flow in a Hele-Shaw gap of width b.
- **(U, V)**, the texture coordinate the wax carries, for Clip = Dyed.

## The flow, and why it is a streamfunction

Darcy in the gap: u = -(b^2 / 12 mu)(grad p - rho g - F), div u = 0. The spec
asks for a pressure solve. It is solved for the **streamfunction** instead
(u = d psi/dy, v = -d psi/dx), which is the same Darcy law with the pressure
eliminated, and still a variable-coefficient Poisson problem for the same
multigrid. Three reasons, each of which a check depends on:

1. **The velocity is divergence-free by construction**, whatever the solver's
   residual: each face's flux is the difference of two corner values of psi,
   and round a cell they cancel. A conservative, limited scheme only keeps its
   no-new-extrema property on a divergence-free field (`--volume`).
2. **The pressure is eliminated exactly.** Summing Darcy round the loop of four
   cell centres about a node, the pressure differences telescope to zero, and
   the right-hand side is the circulation of the body force. For a horizontally
   uniform lamp that is the difference of two bit-identical floats: a level
   slab's psi is exactly zero, not "zero to the solver's tolerance" (`--still`).
3. **Dirichlet walls.** psi = 0 on all four walls is "no flow through any of
   them"; there is no null space and no compatibility condition.

The operator: sum over the four node edges of a (psi_0 - psi_n) = C, with a the
resistivity 12 mu / b^2 averaged onto the cell face the edge coincides with
(resistances in series: the arithmetic mean of 1/K), times the face's aspect.
Written out in `Shaders.cpp` (`kCoefShader`, `kCirculationShader`) and
independently in the harness (`cpuSystem`), and `--darcy` checks the two agree.

**The force.** rho' g downward (the density anomaly about the liquid at 20 C, so
floats keep the anomaly's precision) and the Korteweg force in its potential
form, -phi grad(beta mu^), which is exactly balanced at the Cahn-Hilliard
equilibrium (mu^ uniform).

### Darcy rather than Stokes

A flat-panel lamp is a Hele-Shaw cell, and in one the flow is Darcy's law when
the gap Reynolds number Re_b = rho U b^2 / (mu L) is small. At this lamp's
typical speeds (U ~ 5 mm/s, b = 4 mm, mu = 2 mPa s, L ~ 3 cm) it is about 1.3:
marginal, not small. A real round lamp is neither case: its blob Reynolds
number rho U a / mu is 10 to 100, so it is not creeping flow either, and
Stokes would be no more right than Darcy. Darcy was chosen because it is one
scalar elliptic solve per step, because its blob speed has a closed form to
check against, and because it is exact for the thin flat lamp the frame
literally is. The **Brinkman** term (viscous shear in the plane, which rounds
blobs) was not added: it makes the streamfunction equation fourth order, which
this multigrid does not solve. Decided, and an open question below.

### The multigrid

Vertex-centred V(1,1): red-black Gauss-Seidel with both colours in one pass
(a black node computes its red neighbours' new values itself: exactly the
two-pass sweep, bit for bit, for one pass's cost), the residual computed on the
fly inside the full-weighting restriction, the coarsest grid (at most 8 x 8
nodes) solved whole in every fragment by 40 sweeps of SOR, and the correction
prolonged **by the operator** (flux-continuous between coarse nodes: Alcouffe,
Brandt, Dendy & Painter 1981) with the first post-smoothing sweep in the same
pass. Coarse coefficients: harmonic along an edge, (1/4, 1/2, 1/4) across it.
One V-cycle per substep, warm-started from the last psi.

Measured (`--multigrid`): 0.127 a cycle on constant coefficients (local Fourier
analysis gives 0.074 for two grids; stated 0.15), 0.129 across the cold slab's
10^5 jump, 0.247 through a convecting lamp; one warm-started cycle leaves psi
7.4e-4 of the flow's energy from the exact solve, against the 1% a step needs.

## The wax: Cahn-Hilliard, and why its mobility is what it is

f = phi^2 (1 - phi)^2, interface width xi = one cell (10-90% in 4.4 cells),
mu^ = f'(phi) - 2 xi^2 lap phi, and beta = 3 sigma / xi so that the interface
carries surface tension sigma. Degenerate mobility D max(4 phi (1 - phi), 0.02).

The mobility is a **numerical device**, not a property of wax: the interface
has to relax ahead of the flow deforming it, or the potential-form Korteweg
force is never at equilibrium and the spurious currents never die. At D = dx x
5e-5 m/s they sat at a floor (4e-4 m/s with superbee, 7e-5 with van Leer); at
ten times that they fall monotonically. Ten times makes Cahn-Hilliard's
explicit limit the stiffest in the lamp, so it has its own pass, subcycled
inside each flow step: two passes a subcycle where a flow step is about thirty.

## The heat

One volumetric heat capacity for both phases (water's, 4.18e6 J/m^3 K; paraffin's
is about 1.9e6). Face loss 2.5 W/m^2 K through each glass face, uniform, and a
cap of 60 W/m^2 K over the top edge's area. With uniform face loss and one heat
capacity, the lamp's mean temperature obeys the lumped law C dTm/dt = P - hA (Tm
- Ta) **exactly**, whatever the field looks like and with the flow on or off,
because every flux telescopes. That is why the capacity is shared; `--heat`
checks the law with the cap's extra loss switched off, and every joule with it
on. For a flat panel the faces are ~100 times the edges' area, so face loss
dominating is the physics; "strongest at the cap" is the cap's coefficient.

The bulb is a Gaussian across the base, normalised in double so its row sums
to exactly the bulb's watts; the coil adds conductivity (0 to 6 W/m K) to the
bottom 3% of the lamp's middle half. **No latent heat of fusion**: it did not
fit; melting is a viscosity law only. Wax viscosity: the melt's above the
melting point, rising e-fold every 2.5 K below it, capped at 10^4 times.

Densities (sources: paraffin melts expand 8e-4 to 1e-3 /K near melting;
water's volumetric expansion runs 2.1e-4 at 20 C to 5.1e-4 at 60 C; NaCl brine
is 1.0071 at 1% and 1.0286 at 4%, CRC Handbook): wax 1025 kg/m^3 at 20 C,
9e-4 /K; liquid 998.2 (1 + 0.0070 S%), 3e-4 /K. So T* = 20 + (rho_w0 - rho_l0)
/ (rho_w0 a_w - rho_l0 a_l), 63 C with no salt, falling 11.3 K per 1% of salt.
The default salt, 0.6%, puts T* at 56 C, just above the warm lamp's mean; at 1%
(T* = 52) every cell of the warm lamp was above T* and the wax pooled at the
cap, which is the "too hot" stall the spec names and the wrong default.

## The optics, and the one invented step

The slice is 2-D, so a blob has no thickness. **Invented**: each blob is
inflated into a lens by solving -lap h = phi with h pinned to zero on the
water's side of phi = 1/2; for a round blob h = (R^2 - r^2)/4, so 4 sqrt(h) is
2 sqrt(R^2 - r^2), a sphere's chord. It is the same multigrid with a penalty
diagonal, one warm-started cycle a frame, only when Lamp and Behind are shown.
`--lens` checks a round blob against the sphere. Refraction is the thin-prism
turn (n_wax - n_water) grad t, carried to a world Refraction metres behind;
absorption is Beer-Lambert, Wax Colour being what 3 cm of wax transmits and
Liquid Tint what the lamp's middle does.

**Cylinder**: a vertical cylinder of radius W/2 seen side on. Snell at the
front (air to water; a thin wall of uniform thickness offsets a ray without
turning it, and is neglected), straight across, Snell at the back, on to the
world. The wax is in the middle plane. `--glass` checks both mappings against
the closed form.

**Dyed**: the wax carries (U, V), advected semi-Lagrangian. A dye field
stretched past 4:1 either way relaxes back towards rest at (stretch - 4)/4 per
30 s of lamp: continuous, local, and it lets a pinched picture heal rather than
smear for ever. Reset and Pour re-seed it.

## The step

dt = min(what is left of the frame, the heat's explicit limit, the capillary
limit, Courant 1/4 on last frame's fastest face times 1.5). Cahn-Hilliard
subcycles inside. At most 32 substeps a frame; past that the lamp runs slower
than Speed asks (`--bench` prints the achieved rate).

- **The fastest face is read back a frame late** through a pixel-pack buffer
  and a fence, so the read never stalls. Waited on rather than polled, so every
  run takes the same steps.
- **Should the flow outrun that estimate, the whole psi is scaled down** to the
  step's Courant number on the GPU, from this substep's own reduced speed. Never
  a face on its own (see the traps).
- **The capillary limit**, kCapillary h^3 / (sigma K_max), with kCapillary = 5
  MEASURED (see the traps). It is why Surface Tension stops at 10 mN/m.

## The traps

Ordered by how much time they cost.

**The GPU compiler reassociates floating point.** A level slab's circulation
came out 2.6e-8, then (with the opposite faces differenced first) 9.5e-15:
Apple's GLSL compiler had distributed phi (mu_R - mu_L) into two products and
fused one into an FMA, cancelling it against a rounded copy of itself. The
fix is GLSL's `precise`, which this GL honours: the circulation and every flux
in the update and interface passes (conservation rests on the two cells that
share a face computing its flux bit for bit identically) are `precise`.

**`std::max` swallows NaN, and `--still` passed on a lamp full of it.** The
largest |psi| of an all-NaN field came out 0, "exactly zero, as required".
Every maximum in the harness now returns infinity for a non-finite value.

**Clamping a face's speed makes the velocity divergent.** The first safety net
for a flow outrunning its step clamped each face to the Courant limit. A
conservative scheme on a divergent field piles wax and heat into every
converging cell: T went to 0.01 and phi to infinity. The net now scales all of
psi, which stays divergence-free.

**The multigrid diverged on the cold slab**, eight times a cycle, with bilinear
prolongation: solid wax 10^5 times the water's resistivity (it diverged at 10^3
too). The exact solution was 4e-10. Operator-dependent interpolation fixed it
(0.11 a cycle there) and took the running lamp from 0.27 a cycle to 0.12.
Found only because the negative control for `--still` (a tilted slab) blew up.

**Darcy flow with surface tension has a capillary step limit.** A resting blob's
currents grew to mm/s as a grid-scale ripple along its edge at 0.2 s steps and
died away at 0.133 s (2 mN/m); at 8 mN/m, 1/30 s grew and 1/60 s did not. The
ripple's rate came out 0.10-0.14 sigma K / h^3, hence kCapillary = 5. The
constant is measured, not derived; a semi-implicit capillary term would lift it
and was not attempted.

**Superbee made the spurious currents permanent.** It was the first limiter on
phi, for its sharp edges; it is anti-diffusive and put back the interface
energy Cahn-Hilliard took out. Van Leer on both fields now.

**The coarsest solve cost a millisecond a call.** At up to 16 x 16 nodes the
local array (dynamically indexed, so private memory) times 60 sweeps in every
fragment made the default frame 5.4 ms; at 8 x 8 and 40 sweeps it is 1.9 ms.

**The inflation pinned h to zero inside the interface**, so round blobs came out
a fifth too narrow and Behind barely showed the wax. It is pinned only on the
water's side of phi = 1/2 now; `--lens` checks it.

**A 17-character name reaches the host as 16.** `Liquid Tint_Green` went out as
`Liquid Tint_Gree` (oxbow showed it). The swatch's hidden components are
`Tint_Green` and `Tint_Blue`, and `verify.sh` fails any name over 16 or any
duplicate.

**A worst-case rounding bound had no teeth.** Wax conservation against n x 2 ulps
(n = cells x passes) allowed 1.6e-2 of the wax, and a non-conservative scheme
passed it (it failed only by blowing up the heat). The bound is now six random
walks (Higham's sqrt(n) rule), 1.2e-5; the wrong model drifts 1.3e-4.

**A global extremum test passes an unlimited scheme.** "T never exceeds the
starting maximum" passed with the limiter replaced by a plain average, because
the glass cools the maximum faster than the overshoots grow. The check is now
the local maximum principle, every cell against its stencil's old range, every
step. Its first run then FAILED on the shipped scheme -- because switching the
bulb off only drops its target, and the one-pole bulb went on heating the base.
The test lets the bulb's lag die first. (Halving the Courant number changed
nothing, which is how the bulb was found.)

**A weak negative control is not a control.** Zeroing Cahn-Hilliard's mobility
was meant to stop a resting blob's currents dying; they fell 133x anyway (the
flow relaxes the interface too) and it failed only on its last mark. Surface
tension with the wrong sign is the control now: currents 27x over the bound.

**The heat check's Euler bound was 50 K wide.** (dt/2) t max|y''| is dominated by
the bulb's half-second ramp at t = 0; the one-face wrong model passed it. The
bound is (dt/2) integral |y''| dt, taken numerically; the wrong model fails by
300x.

**The glass loses heat in the same explicit step that conduction spreads it**, so
a hot spot's variance grows 2 kappa dt / (1 - r dt) a step, not 2 kappa dt:
0.075% at these steps, and `--diffusion` predicts that.

**Editing `verify.sh` while it runs** corrupts the run: bash reads scripts as it
goes, and the offsets moved. It reported an unmatched quote that `bash -n` never
found.

**This machine was shared.** Other sessions had ffmpeg at 700% and Arena up;
timings moved 50% between runs. The bench takes the best and the median of five
batches.

## Every numeric check, and where its tolerance comes from

| check | bound | why that number |
| --- | --- | --- |
| `--still` slab | exactly 0 | the circulation of a horizontally uniform field is a difference of bit-identical floats |
| `--still` blob | first mark <= K sigma / R^2; falling at every mark and 10x overall | the capillary Darcy speed: the whole Laplace pressure driving flow across the blob |
| `--volume` wax | 6 x 2 ulp(1.5) x sqrt(cells x passes) | each cell's own rounding, a random walk (Higham 2.8) |
| `--volume` heat | 5 ulps of the cell's T a step | the update's ulp plus T times the divergence's rounding at Courant 1/4 |
| `--crossover` | 1e-5 K + 2 ulps of T | bisection width, and where float(T) passes float(T*) |
| `--darcy` assembly | 1e-6 coefficients, 1e-5 right-hand side | six float operations; rho' is ~3x the anomaly it differences |
| `--darcy` solve | Cauchy-Schwarz from the GPU's residual | the energy-norm error is sqrt(r A^-1 r), exactly |
| `--darcy` law | |E1 - E2| + (g I)^2 + (R/L)^4 | Richardson over three resolutions (its own error, measured); the images' second order |
| `--multigrid` | 0.15 a cycle (water) | LFA 0.074 for two grids, plus the same for the V-cycle's inexact coarse solves |
| `--multigrid` step | 1e-2 of the flow's energy | a step moves nothing a quarter cell; 1% of that is 1/400 cell |
| `--diffusion` | steps x 2 ulp(T) x sum r^2 / sum E | each cell with excess rounds 2 ulps a step |
| `--rt` rates | k xi of the law | first order in k xi, after the interface's own response is measured and divided out |
| `--rt` cutoff | k_c xi | half the relative error of each of drho g and sigma |
| `--heat` recurrence | steps x 2 ulp(T) | each cell's step; the mean's error is at most the largest |
| `--heat` law | (dt/2) integral |y''| dt + that | forward Euler's global error for a stable linear ODE |
| `--heat` budget | 2 ulps a step | each step on its own |
| `--bulb` | 1e-12 | the one-pole is exact for any step; double rounding over the steps |
| `--glass` Flat | 1e-6 | pixel centres are exact; float |
| `--glass` Cylinder | 1/512 texel + 2e-5 | the filter's 8 sub-texel bits; GLSL gives asin, tan, sin, cos no accuracy bound |
| `--lens` | xi / R (4/3 of it at R/2) | the edge is at the interface's middle to half its width |
| `--state` | exact | every piece of GL state a host could care about |

Every one has a negative control in `--negative` (14 wrong models, all
detected), and `--mutate` changes one character of the shipped GLSL (gravity's
sign in the buoyancy) and requires `--crossover` to fail (three checks do).

## Would each check hold on another rasteriser, at another raster?

- **The physics checks run on a grid in metres**, so their raster is
  irrelevant by construction: `--still`, `--volume`, `--crossover`, `--darcy`,
  `--multigrid`, `--diffusion`, `--rt`, `--heat`, `--bulb`, `--lens` would give
  the same numbers at 4K.
- **The optics checks run at 480 x 270 and 1280 x 720** (`--glass`).
- **Another GPU.** The bounds that are rounding bounds (volume, heat, diffusion,
  crossover) assume IEEE float storage and `precise` being honoured. A driver
  that ignored `precise` would fail `--still` (the slab's psi would be ~1e-14,
  not 0) -- which is the right outcome, since the level lamp would then drift.
  The filter-precision bound in `--glass` is the D3D10 floor of 8 sub-texel
  bits, which every desktop GPU meets. The transcendental allowance (2e-5) is a
  guess at "any sane asin"; measured error here is 4e-6.
- **Measured-not-derived constants** would not transfer: `kCapillary` (5) is
  from this machine's thresholds, with a 1.5-3x margin.

## Shape of the code

    source/Physics.*     constants, the density/viscosity laws (//= mirrored),
                         T*, the blob speed and RT laws, the grid, the step limits
    source/Controls.*    0..1 host parameters to physical units
    source/Shaders.*     fourteen passes; ShaderSource() and the harness's overrides
    source/Bassalt.*     the plugin: clock, events, bulb, the multigrid, the passes
    source/Audio.*       millpond's analyser, plus the bass level and priming switch
    source/PassBuffer.*  FFGLFBO with the leak fixed
    source/GLState.h     put the host's state back (now the pack buffer too)
    tools/bstest/        the harness: the checks, the CPU Darcy system, --pipe/--film
    tools/sweep.py       no control is silently dead
    tools/verify.sh      all of it

## Decisions taken without asking

- **Name and id** as the spec: `bassalt`, `BS01`, `SW Bassalt`.
- **Streamfunction, not pressure**, above.
- **No Brinkman term; no latent heat; no rocket mask** (a stretch goal).
- **Surface Tension 0 to 10 mN/m** (the spec gave no range): the capillary limit
  goes as 1/sigma. Default 2 mN/m, a surfactant-laden lamp.
- **Default Speed 3x.** At 30x the capillary limit costs ~14 substeps a frame.
- **The lamp starts warm**, already moving: a lamp that does nothing for an
  hour reads as broken. Reset gives the cold slab.
- **Warm** sets every cell to the lumped steady temperature Ta + P/(hA_face +
  hA_cap) and the bulb to its target; the stratification then develops in a few
  minutes of lamp time rather than an hour.
- **Bass is not peak-normalised**: a normalised level re-scales a breakdown back
  to "1" within seconds, and a breakdown should let the lamp cool. Bass is the
  gain. Bass Band is 1 to 8 bins; Resolume's bin law is unmeasured fleet-wide.
- **Kick** is joules into the base, 0 to 300, scaled by the hit's strength.
- **Clip Heat** is watts at an all-white clip, spread as the clip's luma.
- **Pour** is the clip's luma through a smoothstep 0.08 wide about the
  threshold; poured wax takes the water's temperature where it lands.
- **The lamp is as wide as the frame**, so a wider frame is a bigger lamp with
  more glass, and needs more bulb for the same temperature.
- **Raw GL bindings** in the passes rather than the SDK's `Scoped*`: those clear
  to 0 on exit and cannot unwind three or more units (millpond's trap). Every
  unit is unbound once at the end of `ProcessOpenGL`; `--state` checks.
- **About is generated, `guide = ""`**: `StoatworksAbout.h` comes from
  sync-about.py since the v0.1.0 registration; with no user guide the About
  block is four parameters (38 in all). Writing a guide and setting
  `"guide": true` makes it five, which needs a rebuild and `verify.sh`.
  `ATTRIBUTIONS.md` is still hand-written.

## What is genuinely verified, and what is assumed

Verified, on this machine (Apple Silicon, macOS 26.4): every check in the
table, `tools/verify.sh` green (universal bundle, `lipo` shows both slices,
plist, ad-hoc signature, `oxbow probe` reads `SW Bassalt` / `BS01` / effect),
30 parameters live in the sweep.

Assumed, or not done:

- **Never loaded into Resolume.** How 38 parameters in seven groups present,
  whether the tint and wax colour show as swatches, whether the clock arrives in
  seconds (it is voted on), whether the events show as buttons.
- **Resolume's FFT bins**, and Bass and Kick against real music.
- **Windows** never built or run. The pack-buffer read, `precise` and RGBA32F
  render targets are standard on DX11-class parts but untested.
- **The look** is judged by eye: the lens rims streak where the lens's slope is
  steep, and the Cylinder's outermost few per cent band where rays graze.
- **Short waves are slow.** At 128 cells the diffuse interface passes only
  56-85% of a sharp interface's Rayleigh-Taylor growth (`--rt` measures it).
  The lamp's blobs are that model's, not a sharp-interface lamp's.
- **Performance**: Detail 256 is not real time at the default surface tension
  (the capillary limit is h^3), and Speed 300x runs at about 70x once the lamp
  convects.
- **The flow's rescue** (scaling psi when the flow outruns last frame's speed)
  is exercised by events but not measured on its own.
- **The dye** is not conserved and relaxes when stretched: it is a picture, not
  a physical quantity.

## Browser demo

`demo/` is the page at bassalt-demo.stoatworks-labs.com, built 2026-09-24 on the
shared kit (`stoatworks-backend/resolume-demo`, vendored by hand into
`demo/vendor/`, never edited here). What runs, and what does not:

- **The shaders run for real**: all fourteen passes plus the vertex shader,
  copied unedited into `demo/plugin.js`. `demo/tools/check_shaders.py`
  (galvo's shape, called from `verify.sh`) compares each with the C++ character
  for character, joins the update pass's two raw strings the way the compiler
  does, and refuses any backslash but the one escape a template literal needs
  (a backslash before a backtick, for the backticks round `precise` in an
  update-pass comment).
- **The orchestration is a port that only a reader checks**: `Physics.cpp`
  (constants, T*, ChooseGrid, the three step limits, the bulb's one-pole),
  `Controls.cpp` (each result rounded to float) and `BassaltPlugin`'s pass
  sequence -- ensureBuffers, the events, Simulate()'s substeps and Cahn-Hilliard
  subcycles, the V-cycle recursion, the inflation, the composite's uniforms.
- **Formats are the plugin's**: RGBA32F cells and coefficients, R32F levels.
  WebGL2 needs EXT_color_buffer_float to render to them (`needFloat`) and
  OES_texture_float_linear to filter them; without the latter a float texture
  with LINEAR filtering is incomplete and even texelFetch reads zero, so the page
  throws a GLError naming it rather than showing an empty lamp.
- **`precise` is dropped** by the kit's port() (ES 3.00 has none). This plugin
  relies on it (the traps above), so the page says a browser may reassociate the
  circulation and fluxes: a level slab need not stay exactly still, and wax need
  not be conserved to the bit.
- **No audio.** Decided: the whole Audio group (the FFT buffer, Bass, Bass Band,
  Kick) is ABSENT, as on readout, rosette and cadence, rather than present and
  dead. The removal is exact: on silence the analyser's Bass() is 0 and it never
  fires, so BulbTarget() is Bulb alone and no Kick is queued. The bulb runs on the
  Bulb control at its own default (30 W); nothing invented drives it. Said in the
  banner, the disclosure and here.
- **Warm, Reset, Pour** are booleans the renderer releases on the frame it takes
  the press (readout's precedent: the kit has no FF_TYPE_EVENT).
- **The speed read-back differs.** The plugin waits on its fence so every run
  takes the same steps; WebGL2 allows no client wait. The page collects the
  speed only once `getSyncParameter` says SIGNALED and makes no new request while
  one is in flight, so a GPU running behind sizes its steps from the last speed
  that arrived. The update shader's rescue (scale all of psi) is unchanged.
  Chrome trap, measured: re-using the pixel-pack buffer's storage warns "READ-usage
  buffer was written, then fenced, but written again before being read back" on
  EVERY frame even with the read in between, until the context stops reporting
  WebGL errors; `bufferData` fresh storage before each request silences it.
- **Clip = Behind** by default with the kit's geometry card first (the nearest
  thing to the harness's card); Lights on black is the clip to Pour from. The
  composite writes alpha 1, so there is no backdrop picker.
- **Detail stays at the plugin's default, 128.** The default lamp (224 x 128,
  2 substeps a frame at 3x) measured ~50 fps in Chrome on this Mac's Apple
  Silicon GPU; at 30x it takes ~16 substeps a frame. Headless SwiftShader runs
  it, slowly (Detail 64 at 300x: 4 lamp-minutes in four wall minutes, blobs
  visible, no console errors).
- **The About block is absent**, as on every demo in the suite.
- A line under the canvas reports lamp time, grid, substeps a frame, the bulb and
  T*, and says when the 32-substep cap makes the lamp run slower than Speed.

## Open design questions

- A semi-implicit capillary term would lift the step limit, allow clean
  paraffin's 50 mN/m, and make Speed 300x real.
- Brinkman (rounder blobs) needs a fourth-order solve or a split.
- Latent heat would give the real lamp's long plateau at the melting point.
- Should the lamp be a lamp-shaped mask in the frame (the spec's rocket), rather
  than filling it? The heat balance would then not depend on the aspect.
- Is 2 mN/m and a T* just above the mean the right default lamp? It cycles, but
  it is one point in a space nobody has looked at on a real screen.
