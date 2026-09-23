#include "Physics.h"

#include <algorithm>
#include <cmath>

namespace bassalt
{
namespace physics
{
double LiquidDensity0( double salt )
{
	return kWaterDensity * ( 1.0 + kSaltDensityPerPercent * salt );
}

double CrossoverSlope( double salt )
{
	return kWaxDensity * kWaxExpansion - LiquidDensity0( salt ) * kLiquidExpansion;
}

double Crossover( double salt )
{
	return kReferenceT + ( kWaxDensity - LiquidDensity0( salt ) ) / CrossoverSlope( salt );
}

//= mirrored in Shaders.cpp, kPropsShader
double DensityAnomaly( double phi, double T, double salt )
{
	return phi * CrossoverSlope( salt ) * ( Crossover( salt ) - T )
	       - LiquidDensity0( salt ) * kLiquidExpansion * ( T - kReferenceT );
}

//= mirrored in Shaders.cpp, kPropsShader
double WaxViscosity( double T, const Lamp& lamp )
{
	const double below = std::max( lamp.meltingPoint - T, 0.0 ) / kFreezeWidth;
	return lamp.waxViscosity * std::exp( std::min( below, std::log( kSolidMultiple ) ) );
}

//= mirrored in Shaders.cpp, kPropsShader
double Resistivity( double phi, double T, const Lamp& lamp )
{
	const double p  = std::clamp( phi, 0.0, 1.0 );
	const double mu = p * WaxViscosity( T, lamp ) + ( 1.0 - p ) * kLiquidViscosity;
	return 12.0 * mu / ( lamp.gap * lamp.gap );
}

double BlobSpeed( double kIn, double kOut, double drho )
{
	return kIn * kOut * drho * kGravity / ( kIn + kOut );
}

double RayleighTaylorRate( double k, double drho, double sigma, double k1, double h1, double k2, double h2 )
{
	const double drive = drho * kGravity - sigma * k * k;
	const double drag  = 1.0 / ( std::tanh( k * h1 ) * k1 ) + 1.0 / ( std::tanh( k * h2 ) * k2 );
	return k * drive / drag;
}

double FaceLossRate( double gap )
{
	return 2.0 * kFaceLoss / ( kHeatCapacity * gap );
}

double FaceConductance( const Lamp& lamp )
{
	return 2.0 * kFaceLoss * lamp.width * lamp.height;
}

double CapConductance( const Lamp& lamp )
{
	return kCapLoss * lamp.width * lamp.gap;
}

double HeatCapacityTotal( const Lamp& lamp )
{
	return kHeatCapacity * lamp.width * lamp.height * lamp.gap;
}

Grid ChooseGrid( double width, double height, int cellsUp )
{
	Grid grid;
	grid.ny = std::max( cellsUp, 8 );

	//The fewest halvings that bring the coarsest grid inside the coarsest
	//shader's 16 x 16 nodes, the width rounded to a whole number of coarsest
	//cells each time. More levels than that only add passes, and a pass is
	//what costs here.
	const double across = grid.ny * width / height;
	int halvings        = 0;
	grid.nx             = std::max( 2, static_cast< int >( std::lround( across ) ) );
	for( int h = 1; grid.ny % ( 1 << h ) == 0; ++h )
	{
		const int unit = 1 << h;
		const int nx   = std::max( 2, static_cast< int >( std::lround( across / unit ) ) ) * unit;
		halvings       = h;
		grid.nx        = nx;
		if( nx / unit + 1 <= kCoarsestMax && grid.ny / unit + 1 <= kCoarsestMax )
			break;
	}

	grid.levels = halvings + 1;
	grid.dx     = width / grid.nx;
	grid.dy     = height / grid.ny;
	return grid;
}

double InterfaceWidth( const Grid& grid )
{
	return kInterfaceCells * std::max( grid.dx, grid.dy );
}

double InterfaceMobility( const Grid& grid )
{
	return std::max( grid.dx, grid.dy ) * kInterfaceSpeed;
}

double ChemicalScale( const Grid& grid, double sigma )
{
	return 3.0 * sigma / InterfaceWidth( grid );
}

double DiffusionLimit( const Grid& grid, const Lamp& lamp )
{
	const double laplacian = 4.0 / ( grid.dx * grid.dx ) + 4.0 / ( grid.dy * grid.dy );

	//Heat: the fastest diffusivity is the coil layer's. Half the stability
	//limit, so that conduction alone uses at most half of the convex-
	//combination budget and advection (kCourant) the other half: that is what
	//keeps a new temperature inside the range of the old ones.
	const double kappa = ( std::max( kLiquidConductivity, kWaxConductivity ) + lamp.coil ) / kHeatCapacity;
	return kDiffusionSafety / ( kappa * laplacian );
}

double InterfaceLimit( const Grid& grid )
{
	//Cahn-Hilliard, linearised about a pure phase: rate D L ( 2 xi^2 L + f'' ),
	//f'' = 2 at phi = 0 and 1; forward Euler is stable below 2 / rate.
	const double laplacian = 4.0 / ( grid.dx * grid.dx ) + 4.0 / ( grid.dy * grid.dy );
	const double xi        = InterfaceWidth( grid );
	const double D         = InterfaceMobility( grid );
	return kDiffusionSafety * 2.0 / ( D * laplacian * ( 2.0 * xi * xi * laplacian + 2.0 ) );
}

double CapillaryLimit( const Grid& grid, const Lamp& lamp )
{
	if( lamp.tension <= 0.0 )
		return 1e30;
	const double h    = std::min( grid.dx, grid.dy );
	const double kMax = lamp.gap * lamp.gap / ( 12.0 * std::min( kLiquidViscosity, lamp.waxViscosity ) );
	return kCapillary * h * h * h / ( lamp.tension * kMax );
}

double BulbStep( double current, double target, double dt, double tau )
{
	if( tau <= 0.0 )
		return target;
	return current + ( target - current ) * ( 1.0 - std::exp( -dt / tau ) );
}

} // namespace physics
} // namespace bassalt
