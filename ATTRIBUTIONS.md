# Attributions

bassalt is built on other people's work. This file lists what that work is,
who did it, and what it is doing here.

> **Provisional.** Across the fleet this file is generated from master lists in
> `stoatworks-backend` by `scripts/sync-attributions.py`. bassalt is in that
> script's `names.json` but not in its component lists, so this copy is still
> hand-written; v0.1.0 shipped that way. Finishing the registration and re-running
> the sync is the fix — and note that the script's `--only` flag truncates the
> file rather than filtering it.

## Third-party code this project uses

### Resolume FFGL SDK

<https://github.com/resolume/ffgl>
Licence: BSD-3-Clause
Copyright: FreeFrame

Vendored as a git submodule at `external/ffgl`, pinned to `b1afaf9`.

The plugin ABI itself. An FFGL effect is defined by this SDK's headers — there
is no other way to be loadable by Resolume Arena and Avenue.

### GLEW — the OpenGL Extension Wrangler Library

<https://github.com/nigels-com/glew>
Licence: BSD-3-Clause (with Mesa 3-D and Khronos components)
Copyright: Milan Ikits, Marcelo E. Magallon and Lev Povalahev

Windows only, from vcpkg, statically linked. The SDK's headers pull it in for
the OpenGL function pointers; macOS uses the system OpenGL framework instead.

### zlib

<https://zlib.net>
Licence: zlib
Copyright: Jean-loup Gailly and Mark Adler

Ships with macOS. The offline harness links it to deflate its PNG output.
Nothing in the shipped plugin uses it.

## Work from elsewhere in the fleet

### millpond — the analyser, the clock, the harness

<https://github.com/stoatworks-labs/millpond>
Licence: MIT
Copyright: Stoatworks Labs

`source/Audio.{h,cpp}` is millpond's analyser (rosette's, from macroblock's),
with its primed first frame; bassalt adds a bass level and a switch that turns
the priming off for `bstest`'s negative control. The host-clock vote in
`UpdateClock` is rosette's, unchanged. The harness's shape, `--pipe`/`--film`/
`--script`, the negative-control runner, `tools/sweep.py` and
`tools/verify.sh` are millpond's, adapted.

### vectrix, tinsel

<https://github.com/stoatworks-labs/vectrix>
Licence: MIT
Copyright: Stoatworks Labs

`GLState.h` (put the host's state back however the frame ends) is vectrix's by
way of millpond, now also restoring the pixel-pack buffer. `PassBuffer` is
tinsel's (`FFGLFBO` with the SDK's colour-texture leak fixed). `Diag` and the
CMake shape come from tinsel.

## Method

Textbook physics and numerics, described in books and papers rather than
copied from anyone's source:

- Darcy flow in a Hele-Shaw cell, and the translation speed of a circular
  inclusion (the 2-D depolarisation result) -- derived in `AGENTS.md`.
- Rayleigh-Taylor instability in a Hele-Shaw cell with surface tension and
  finite layers (Saffman & Taylor, *Proc. R. Soc. A* 245, 312 (1958), and the
  standard linear analysis).
- The Cahn-Hilliard equation (Cahn & Hilliard, *J. Chem. Phys.* 28, 258
  (1958)) and its Korteweg force in potential form (Jacqmin, *J. Comput. Phys.*
  155, 96 (1999)).
- MUSCL with van Leer's limiter (van Leer, *J. Comput. Phys.* 32, 101 (1979)).
- Multigrid: red-black Gauss-Seidel and local Fourier analysis (Trottenberg,
  Oosterlee & Schuller, *Multigrid*, 2001); operator-dependent interpolation
  (Alcouffe, Brandt, Dendy & Painter, *SIAM J. Sci. Stat. Comput.* 2, 430
  (1981)).
- Rounding error as a random walk (Higham, *Accuracy and Stability of
  Numerical Algorithms*, section 2.8).
- Brine densities and water's thermal expansion from the CRC Handbook's
  tables; paraffin's expansion from the melt range quoted in AGENTS.md.

## Getting this wrong

If your work is here and the description is inaccurate, the licence is wrong, or
you would rather not be listed — open an issue and it will be fixed.
