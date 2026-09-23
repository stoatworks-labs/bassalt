#pragma once

/**
    The lamp's physics on the CPU side: its constants, the laws the shaders
    evaluate per cell, and the grid.

    Everything the harness PREDICTS from is here, in double precision, and the
    shaders are handed the same numbers as uniforms. Where a shader evaluates a
    law itself (the density and viscosity per cell), the law is written in both
    places and marked `//= mirrored` on both sides; the harness checks the
    GPU's answer against this one.

    ------------------------------------------------------------ the model

    A 2-D slice of the lamp, a Hele-Shaw cell of gap b. Four fields per cell:

      phi   wax fraction, 1 wax, 0 water. Cahn-Hilliard, in flux form.
      T     temperature, degrees C. Advected in flux form, conducted, heated
            by the bulb through the base, cooled through the glass faces and
            harder through the cap.
      U, V  the texture coordinate the wax carries, for Clip = Dyed.

    and the flow, from Darcy's law in the gap:

      u = -( b^2 / 12 mu ) ( grad p - rho g - F_st ),   div u = 0

    solved for the streamfunction psi (u = d psi/dy, v = -d psi/dx), which is
    the same Darcy law with the pressure eliminated. AGENTS.md says why.
*/

namespace bassalt
{
namespace physics
{
constexpr double kGravity = 9.81;

/// The temperature the linear density laws are written about, degrees C.
constexpr double kReferenceT = 20.0;

/// Wax, extrapolated as a liquid to 20 C: 1025 kg/m^3. A lamp's wax is
/// paraffin loaded with something denser (historically a chlorinated solvent)
/// until it sits just heavier than the water when cold.
constexpr double kWaxDensity = 1025.0;

/// Volumetric expansion of liquid paraffin, 1/K. Paraffin melts expand at
/// 8e-4 to 1e-3 /K near their melting points; 9e-4 is the middle.
constexpr double kWaxExpansion = 9.0e-4;

/// Pure water at 20 C, kg/m^3.
constexpr double kWaterDensity = 998.2;

/// The liquid's volumetric expansion, 1/K. Water's runs from 2.1e-4 at 20 C to
/// 5.1e-4 at 60 C; one linear law over the lamp's range takes 3e-4.
constexpr double kLiquidExpansion = 3.0e-4;

/// Brine's density rises by about 0.70% per 1% of dissolved salt by mass (NaCl
/// at 20 C: 1.0071 at 1%, 1.0286 at 4%).
constexpr double kSaltDensityPerPercent = 0.0070;

/// The liquid, water and glycol as a lamp's is: 2 mPa s. Held constant; its
/// temperature dependence is small against the wax's.
constexpr double kLiquidViscosity = 2.0e-3;

/// How far below its melting point wax's viscosity rises e-fold, K, and the
/// ceiling it rises to, as a multiple of the melt's. Paraffin goes from a
/// few mPa s to effectively solid over a few kelvin; 1e4 is "does not move on
/// the lamp's time scale" without making the solve's coefficients worse.
constexpr double kFreezeWidth   = 2.5;
constexpr double kSolidMultiple = 1.0e4;

/// One volumetric heat capacity for both phases, J/(m^3 K). Water's is 4.18e6
/// and paraffin's about 1.9e6; using water's for both is what makes the lamp's
/// mean temperature obey the lumped law exactly (AGENTS.md).
constexpr double kHeatCapacity = 4.18e6;

/// Thermal conductivities, W/(m K): water 0.60, liquid paraffin 0.24.
constexpr double kLiquidConductivity = 0.60;
constexpr double kWaxConductivity    = 0.24;

/// Heat loss through each of the two glass faces, W/(m^2 K). The lamp is as
/// wide as the frame, so a 16:9 slice has several times a real lamp's glass
/// for its bulb; 2.5 per face makes 30 W hold a 0.3 m lamp near 60 C.
constexpr double kFaceLoss = 2.5;

/// Heat loss through the metal cap -- the top edge -- W/(m^2 K) over the cap's
/// area (the frame width times the gap). A cap is a finned heat sink.
constexpr double kCapLoss = 60.0;

/// Where the coil sits: the bottom 3% of the height, the middle half of the
/// width. The bulb shines up through the base under it, a Gaussian across the
/// base 0.15 of the width wide.
constexpr double kCoilHeight   = 0.03;
constexpr double kCoilWidth    = 0.50;
constexpr double kBulbSpread   = 0.15;

/// Wax and water refractive indices.
constexpr double kWaxIndex   = 1.44;
constexpr double kWaterIndex = 1.34;

//---------------------------------------------------------------------------
// The numerics that are part of the model.
//---------------------------------------------------------------------------

/// The Cahn-Hilliard interface's width parameter xi, in cells: phi runs as
/// 1 / ( 1 + exp( -d / xi ) ) across it, 10% to 90% in 4.4 xi.
constexpr double kInterfaceCells = 1.0;

/// The Cahn-Hilliard mobility is a numerical device (AGENTS.md): D = dx * this,
/// m^2/s, times the degenerate factor 4 phi (1 - phi). Fast enough that the
/// interface relaxes (rate D / xi^2 = 0.2 /s at 128 cells) well ahead of the
/// flow deforming it, which is what lets the spurious currents die away:
/// at a tenth of this they sat at a floor (`bstest --still`, AGENTS.md).
/// Cahn-Hilliard is subcycled inside each flow step, so its explicit limit
/// costs two passes a subcycle rather than a whole solve.
constexpr double kInterfaceSpeed = 5.0e-4;

/// The floor on the degenerate mobility factor, so that an overshoot past 0 or
/// 1 can still relax.
constexpr double kMobilityFloor = 0.02;

/// The advective Courant number on the fastest face. MUSCL with a limiter
/// whose slope ratio reaches 2, under forward Euler, keeps every cell a convex
/// combination of its neighbours at a summed inflow Courant number of 1/2, and
/// a cell has at most two inflow faces at the fastest speed.
constexpr double kCourant = 0.25;

/// Safety on the explicit diffusion and Cahn-Hilliard limits.
constexpr double kDiffusionSafety = 0.8;

/// The capillary step's constant. MEASURED, not derived: the grid-scale
/// ripple's growth rate came out as 0.10-0.14 sigma K / h^3 from the neutral
/// blob (`bstest --still`) at 2 and 8 mN/m -- stable at 7.3 h^3 / ( sigma K )
/// in both, unstable at 15 -- so 5 is the last stable step with margin.
constexpr double kCapillary = 5.0;

/// Most substeps one host frame may take. Past this the lamp runs slower than
/// Speed asks; `bstest --bench` prints the achieved rate.
constexpr int kMaxSubsteps = 32;

/// Multigrid V-cycles per substep, warm-started from the last psi.
/// `bstest --multigrid` measures that one meets the step's tolerance.
constexpr int kCyclesPerStep = 1;

/// Sweeps of the in-register SOR solve on the coarsest grid, and its
/// over-relaxation (near the optimum 2 / ( 1 + sin( pi / 8 ) ) for the
/// widest coarsest grid).
constexpr int kCoarsestSweeps = 40;
constexpr double kCoarsestOmega = 1.6;

/// The margin on last frame's fastest face when choosing this frame's step:
/// the speed is read back a frame late so the read never stalls.
constexpr double kSpeedMargin = 1.5;

/// The coarsest grid's largest node count along either axis; the coarsest
/// shader holds the whole grid in a local array this big squared.
constexpr int kCoarsestMax = 8;

//---------------------------------------------------------------------------
// The lamp as the controls describe it, in SI units.
//---------------------------------------------------------------------------
struct Lamp
{
	double height       = 0.3; ///< m, the frame's height
	double width        = 0.533;///< m, the frame's width
	double gap          = 0.004;///< m, b
	double ambient      = 22.0; ///< C
	double salt         = 1.0;  ///< % by mass
	double tension      = 0.002;///< N/m
	double waxViscosity = 0.02; ///< Pa s, above the melting point
	double meltingPoint = 50.0; ///< C
	double coil         = 3.0;  ///< W/(m K) added in the coil layer
};

/// The liquid's density at the reference temperature, kg/m^3.
double LiquidDensity0( double salt );

/// The crossover: the temperature at which wax and liquid are equally dense.
/// Above it wax is the lighter and rises.
///   rho_w0 ( 1 - a_w ( T - Tr ) ) = rho_l0 ( 1 - a_l ( T - Tr ) )
///   T* = Tr + ( rho_w0 - rho_l0 ) / ( rho_w0 a_w - rho_l0 a_l )
double Crossover( double salt );

/// d( rho_wax - rho_liquid ) / dT, negated: rho_w0 a_w - rho_l0 a_l, kg/(m^3 K).
double CrossoverSlope( double salt );

/// The density anomaly the flow sees, rho - rho_l( Tr ), kg/m^3.   //= mirrored
///   phi * D1 * ( T* - T ) - rho_l0 a_l ( T - Tr )
double DensityAnomaly( double phi, double T, double salt );

/// Wax viscosity at T: the melt's above the melting point, rising e-fold every
/// kFreezeWidth below it, capped at kSolidMultiple times.            //= mirrored
double WaxViscosity( double T, const Lamp& lamp );

/// 1/K = 12 mu / b^2, with mu = phi mu_wax + (1 - phi) mu_liquid, phi clamped
/// to [0, 1] for the property.                                       //= mirrored
double Resistivity( double phi, double T, const Lamp& lamp );

/// The Hele-Shaw speed of an isolated circular blob: the 2-D depolarisation
/// result, U = K_in K_out drho g / ( K_in + K_out ). Positive is up for a
/// lighter blob (drho = rho_out - rho_in > 0). AGENTS.md derives it.
double BlobSpeed( double kIn, double kOut, double drho );

/// Rayleigh-Taylor in a Hele-Shaw cell with surface tension, a light layer
/// of depth h1 under a heavy one of depth h2, walls top and bottom:
///   s = k ( drho g - sigma k^2 ) / ( coth( k h1 ) / K1 + coth( k h2 ) / K2 )
/// Growth for k below k_c = sqrt( drho g / sigma ), decay above.
double RayleighTaylorRate( double k, double drho, double sigma, double k1, double h1, double k2, double h2 );

/// The glass: face loss rate 1/tau_f = 2 h_f / ( C b ), 1/s.
double FaceLossRate( double gap );

/// Lamp-wide conductance to the room through both faces, W/K: 2 h_f W H.
double FaceConductance( const Lamp& lamp );

/// Through the cap, W/K: h_cap W b.
double CapConductance( const Lamp& lamp );

/// The lamp's heat capacity, J/K: C W H b.
double HeatCapacityTotal( const Lamp& lamp );

//---------------------------------------------------------------------------
// The grid.
//---------------------------------------------------------------------------
struct Grid
{
	int nx     = 0;///< cells across
	int ny     = 0;///< cells up
	int levels = 0;///< multigrid levels, the finest included
	double dx  = 0;///< m
	double dy  = 0;///< m

	bool operator==( const Grid& other ) const
	{
		return nx == other.nx && ny == other.ny;
	}
	bool operator!=( const Grid& other ) const
	{
		return !( *this == other );
	}

	/// Nodes at level l along x and y: cells / 2^l + 1.
	int NodesX( int level ) const
	{
		return ( nx >> level ) + 1;
	}
	int NodesY( int level ) const
	{
		return ( ny >> level ) + 1;
	}
};

/// Cells up the height are `cellsUp`; across, the frame's aspect times that,
/// rounded to a multiple of 2^(levels - 1) so every level halves exactly. The
/// halving stops at the first level whose nodes fit 8 x 8, which the
/// coarsest shader solves whole. The cells are not quite square (the rounding
/// is at most half a coarsest cell), so the frame maps onto the domain
/// exactly and nothing is stretched.
Grid ChooseGrid( double width, double height, int cellsUp );

/// The largest stable substep for explicit heat conduction, with
/// kDiffusionSafety -- the flow step's limit besides the Courant number.
double DiffusionLimit( const Grid& grid, const Lamp& lamp );

/// The largest stable explicit Cahn-Hilliard step, with kDiffusionSafety:
/// the subcycle.
double InterfaceLimit( const Grid& grid );

/// The capillary limit on the flow step: kCapillary h^3 / ( sigma K_max ).
/// Darcy flow with surface tension relaxes an interface wiggle of wavenumber
/// k at a rate ~ sigma K k^3, fastest at the grid scale, and an explicit step
/// past 2 / rate grows a cell-scale ripple along every interface. Infinite
/// with no surface tension.
double CapillaryLimit( const Grid& grid, const Lamp& lamp );

/// The Cahn-Hilliard numbers: xi (m), the mobility D (m^2/s), and beta
/// (N/m^2, the scale of the chemical potential, 3 sigma / xi).
double InterfaceWidth( const Grid& grid );
double InterfaceMobility( const Grid& grid );
double ChemicalScale( const Grid& grid, double sigma );

/// The bulb's one-pole, exact for any step: approach the target by
/// 1 - exp( -dt / tau ).
double BulbStep( double current, double target, double dt, double tau );

} // namespace physics
} // namespace bassalt
