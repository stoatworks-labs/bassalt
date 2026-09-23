#pragma once

/**
    The passes, as GLSL source.

    ---------------------------------------------------------------- the grids

    Cells: nx x ny, one texel each, cell (i, j) centred at ((i + 1/2) dx,
    (j + 1/2) dy). The state (phi, T, U, V) and the per-cell properties live
    here.

    Nodes: (nx + 1) x (ny + 1), node (I, J) at (I dx, J dy) -- the cells'
    corners. The streamfunction psi lives here, and so does every multigrid
    level, each (nx / 2^l + 1) x (ny / 2^l + 1). Boundary nodes are the walls:
    psi = 0 on all four, which is "no flow through any of them".

    -------------------------------------------------------- one substep

        1. props      cells. The chemical potential mu^ = f'(phi) - 2 xi^2 lap
                      phi, the density anomaly rho', the resistivity
                      A = 12 mu / b^2, the conductivity k.
        2. coef       nodes. The Darcy operator's edge coefficients: A averaged
                      onto each cell face, times the face's aspect.
        3. circulation nodes. The right-hand side: the circulation of the body
                      force (buoyancy + Korteweg) round the node's dual cell.
                      The pressure's circulation is identically zero, which is
                      how it is eliminated.
        4. coarsen    each coarser level's coefficients (harmonic along an
                      edge, (1/4, 1/2, 1/4) across it).
        5. multigrid  V-cycles: smooth (red-black Gauss-Seidel, both colours in
                      one pass), restrict the residual (full weighting, the
                      residual computed on the fly), recurse, solve the coarsest
                      grid exactly in registers, prolong (bilinear), smooth.
        6. reduce     the fastest face speed, for the Courant step.
        7. update     cells. Advection in flux form (MUSCL with van Leer's
                      limiter), conduction, the bulb, the glass and the cap,
                      the clip's heat; the dye's texture coordinates by
                      semi-Lagrangian back-tracing.
        7b. interface cells, subcycled: props again, then one Cahn-Hilliard
                      step of phi in flux form.

    ---------------------------------------------------------- per frame

        8. inflate    nodes, the same multigrid with a different operator:
                      -lap h = phi, pinned to 0 in the water. A round blob's
                      h is ( R^2 - r^2 ) / 4, so 4 sqrt( h ) is its thickness
                      as a lens -- the one invented step (AGENTS.md).
        9. composite  the output: the glass, the lenses, Beer-Lambert, the
                      bulb's light, or one field on its own.

    ---------------------------------------------------------- the harness

    Every source is reached through ShaderSource(), so the harness can put a
    deliberately wrong one in its place (--negative) or change one character
    of the shipped one (--mutate) before InitGL compiles it. Nothing in the
    plugin sets an override.
*/

#include <string>

namespace bassalt
{

enum class ShaderId
{
	Vertex = 0,
	Props,
	Coef,
	Circulation,
	Coarsen,
	Smooth,
	Restrict,
	Coarsest,
	Prolong,
	Update,
	Interface,
	Reduce,
	Event,
	Inflate,
	Composite,

	Count
};

/// What InitGL compiles: the override if the harness set one, else shipped.
const char* ShaderSource( ShaderId id );

/// The shipped text, whatever is overridden.
const char* ShippedSource( ShaderId id );

/// For the harness only. Takes effect at the next InitGL.
void SetShaderOverride( ShaderId id, const std::string& source );
void ClearShaderOverrides();

const char* ShaderName( ShaderId id );

} // namespace bassalt
