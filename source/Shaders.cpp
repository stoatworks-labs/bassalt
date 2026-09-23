#include "Shaders.h"

#include <map>

namespace bassalt
{
namespace
{

//---------------------------------------------------------------------------
// The full-screen quad. Every pass is one.
//---------------------------------------------------------------------------
const char* const kVertexShader = R"(#version 410 core

layout( location = 0 ) in vec4 vPosition;
layout( location = 1 ) in vec2 vUV;

out vec2 uv;

void main()
{
	gl_Position = vPosition;
	uv          = vUV;
}
)";

//---------------------------------------------------------------------------
// 1. props: per cell, what the flow and the heat need to know about it.
//---------------------------------------------------------------------------
const char* const kPropsShader = R"(#version 410 core

uniform sampler2D State;     //(phi, T, U, V), read texel for texel
uniform ivec2 Cells;
uniform vec2 Spacing;        //dx, dy, metres
uniform float Xi2;           //2 xi^2, m^2
uniform float Slope;         //rho_w0 a_w - rho_l0 a_l, kg/(m^3 K)
uniform float Crossover;     //T*, C
uniform float LiquidBeta;    //rho_l0 a_l, kg/(m^3 K)
uniform float ReferenceT;    //C
uniform float MeltingPoint;  //C
uniform float FreezeWidth;   //K
uniform float SolidLog;      //ln( the solid's multiple )
uniform float WaxMu;         //Pa s, the melt
uniform float LiquidMu;      //Pa s
uniform float Resist;        //12 / b^2
uniform float WaxK;          //W/(m K)
uniform float LiquidK;
uniform float CoilK;         //added in the coil layer
uniform vec2 CoilBox;        //half-width and height, as fractions of the lamp

out vec4 fragColor;

//The walls are no-flux for phi: the cell beyond is the cell itself.
float phiAt( ivec2 c )
{
	return texelFetch( State, clamp( c, ivec2( 0 ), Cells - 1 ), 0 ).x;
}

void main()
{
	ivec2 c = ivec2( gl_FragCoord.xy );
	vec4 s  = texelFetch( State, c, 0 );
	float phi = s.x;
	float T   = s.y;

	//The Cahn-Hilliard chemical potential over beta. f = phi^2 ( 1 - phi )^2,
	//so its equilibrium profile is 1 / ( 1 + exp( -d / xi ) ).
	float lapX = ( phiAt( c + ivec2( 1, 0 ) ) - 2.0 * phi + phiAt( c - ivec2( 1, 0 ) ) ) / ( Spacing.x * Spacing.x );
	float lapY = ( phiAt( c + ivec2( 0, 1 ) ) - 2.0 * phi + phiAt( c - ivec2( 0, 1 ) ) ) / ( Spacing.y * Spacing.y );
	float potential = 2.0 * phi * ( 1.0 - phi ) * ( 1.0 - 2.0 * phi ) - Xi2 * ( lapX + lapY );

	//The properties see phi clamped: Cahn-Hilliard may overshoot 0 and 1 by a
	//little, and water cannot be less than no wax.
	float p = clamp( phi, 0.0, 1.0 );

	//= mirrored in Physics.cpp, DensityAnomaly: rho - rho_l( Tref ).
	float rho = p * Slope * ( Crossover - T ) - LiquidBeta * ( T - ReferenceT );

	//= mirrored in Physics.cpp, WaxViscosity and Resistivity.
	float below = max( MeltingPoint - T, 0.0 ) / FreezeWidth;
	float waxMu = WaxMu * exp( min( below, SolidLog ) );
	float resist = Resist * ( p * waxMu + ( 1.0 - p ) * LiquidMu );

	vec2 at = ( vec2( c ) + 0.5 ) / vec2( Cells );
	float k = p * WaxK + ( 1.0 - p ) * LiquidK;
	if( abs( at.x - 0.5 ) < CoilBox.x && at.y < CoilBox.y )
		k += CoilK;

	fragColor = vec4( potential, rho, resist, k );
}
)";

//---------------------------------------------------------------------------
// 2. coef: the Darcy operator's edge coefficients, at the finest nodes.
//
// The node edge from (I, J) to (I+1, J) IS the cell face between cells
// (I, J-1) and (I, J), which the vertical velocity crosses; the edge to
// (I, J+1) is the face between (I-1, J) and (I, J). The face's resistivity is
// the mean of its two cells' (resistances in series), times its aspect.
//---------------------------------------------------------------------------
const char* const kCoefShader = R"(#version 410 core

uniform sampler2D Props;
uniform ivec2 Cells;
uniform vec2 Spacing;

out vec4 fragColor;

float resistAt( ivec2 c )
{
	return texelFetch( Props, clamp( c, ivec2( 0 ), Cells - 1 ), 0 ).z;
}

void main()
{
	ivec2 n  = ivec2( gl_FragCoord.xy );
	float aE = ( Spacing.y / Spacing.x ) * 0.5 * ( resistAt( ivec2( n.x, n.y - 1 ) ) + resistAt( n ) );
	float aN = ( Spacing.x / Spacing.y ) * 0.5 * ( resistAt( ivec2( n.x - 1, n.y ) ) + resistAt( n ) );
	fragColor = vec4( aE, aN, 0.0, 0.0 );
}
)";

//---------------------------------------------------------------------------
// 3. circulation: the right-hand side at the finest nodes.
//
// Darcy on each face: u / K = -grad p + f. Summed round the loop through the
// four cell centres about a node, the pressure differences cancel exactly and
// what is left is the circulation of f:
//
//     dx f_x(S) + dy f_y(E) - dx f_x(N) - dy f_y(W)
//
// f = rho' g (down) - phi grad( beta mu^ ), the Korteweg force in its
// potential form. For a horizontally uniform lamp the two f_y terms are the
// same arithmetic on the same floats and cancel to exactly zero, and a still
// lamp's psi stays exactly zero: `bstest --still`.
//---------------------------------------------------------------------------
const char* const kCirculationShader = R"(#version 410 core

uniform sampler2D State;
uniform sampler2D Props;
uniform ivec2 Cells;
uniform ivec2 Nodes;
uniform vec2 Spacing;
uniform float Gravity;
uniform float Beta;          //3 sigma / xi: mu = Beta * mu^

out vec4 fragColor;

vec4 stateAt( ivec2 c )
{
	return texelFetch( State, clamp( c, ivec2( 0 ), Cells - 1 ), 0 );
}
vec4 propsAt( ivec2 c )
{
	return texelFetch( Props, clamp( c, ivec2( 0 ), Cells - 1 ), 0 );
}

//The x force on the face between cells (I-1, j) and (I, j).
float forceX( int I, int j )
{
	ivec2 l = ivec2( I - 1, j );
	ivec2 r = ivec2( I, j );
	precise float phiFace = 0.5 * ( stateAt( l ).x + stateAt( r ).x );
	precise float force   = -phiFace * Beta * ( propsAt( r ).x - propsAt( l ).x ) / Spacing.x;
	return force;
}

//The y force on the face between cells (i, J-1) and (i, J).
float forceY( int i, int J )
{
	ivec2 b = ivec2( i, J - 1 );
	ivec2 t = ivec2( i, J );
	precise float phiFace  = 0.5 * ( stateAt( b ).x + stateAt( t ).x );
	precise float buoyancy = -Gravity * 0.5 * ( propsAt( b ).y + propsAt( t ).y );
	precise float force    = buoyancy - phiFace * Beta * ( propsAt( t ).x - propsAt( b ).x ) / Spacing.y;
	return force;
}

void main()
{
	ivec2 n = ivec2( gl_FragCoord.xy );
	if( n.x <= 0 || n.y <= 0 || n.x >= Nodes.x - 1 || n.y >= Nodes.y - 1 )
	{
		fragColor = vec4( 0.0 );
		return;
	}

	//Each pair of opposite faces differenced BEFORE anything multiplies it.
	//Written as four products summed, the compiler fuses one product into an
	//FMA with the next subtraction, which cancels it against a ROUNDED copy
	//of itself and leaves one rounding (2.6e-8 here) where a level lamp needs
	//exactly zero. A difference of two identical floats is zero whatever is
	//fused around it.
	precise float circulation = Spacing.y * ( forceY( n.x, n.y ) - forceY( n.x - 1, n.y ) )
	                    - Spacing.x * ( forceX( n.x, n.y ) - forceX( n.x, n.y - 1 ) );
	fragColor = vec4( circulation, 0.0, 0.0, 0.0 );
}
)";

//---------------------------------------------------------------------------
// 4. coarsen: one level's coefficients from the level below.
//
// A coarse edge spans two fine edges end to end (harmonic mean: series) and
// three fine rows side by side, the outer two shared with the neighbouring
// coarse edges (weights 1/4, 1/2, 1/4: parallel). The diagonal term s scales
// with the cell's area, so it restricts like a right-hand side.
//---------------------------------------------------------------------------
const char* const kCoarsenShader = R"(#version 410 core

uniform sampler2D Coef;      //the finer level: (aE, aN, s, -)
uniform ivec2 FineNodes;

out vec4 fragColor;

vec4 fineAt( ivec2 n )
{
	return texelFetch( Coef, clamp( n, ivec2( 0 ), FineNodes - 1 ), 0 );
}

float series( float a, float b )
{
	return a + b > 0.0 ? 2.0 * a * b / ( a + b ) : 0.0;
}

void main()
{
	ivec2 f = 2 * ivec2( gl_FragCoord.xy );

	float aE = 0.25 * series( fineAt( f + ivec2( 0, -1 ) ).x, fineAt( f + ivec2( 1, -1 ) ).x )
	           + 0.5 * series( fineAt( f ).x, fineAt( f + ivec2( 1, 0 ) ).x )
	           + 0.25 * series( fineAt( f + ivec2( 0, 1 ) ).x, fineAt( f + ivec2( 1, 1 ) ).x );

	float aN = 0.25 * series( fineAt( f + ivec2( -1, 0 ) ).y, fineAt( f + ivec2( -1, 1 ) ).y )
	           + 0.5 * series( fineAt( f ).y, fineAt( f + ivec2( 0, 1 ) ).y )
	           + 0.25 * series( fineAt( f + ivec2( 1, 0 ) ).y, fineAt( f + ivec2( 1, 1 ) ).y );

	float s = fineAt( f ).z
	          + 0.5 * ( fineAt( f + ivec2( 1, 0 ) ).z + fineAt( f - ivec2( 1, 0 ) ).z + fineAt( f + ivec2( 0, 1 ) ).z
	                    + fineAt( f - ivec2( 0, 1 ) ).z )
	          + 0.25 * ( fineAt( f + ivec2( 1, 1 ) ).z + fineAt( f + ivec2( -1, 1 ) ).z + fineAt( f + ivec2( 1, -1 ) ).z
	                     + fineAt( f + ivec2( -1, -1 ) ).z );

	fragColor = vec4( aE, aN, s, 0.0 );
}
)";

//---------------------------------------------------------------------------
// 5a. smooth: one red-black Gauss-Seidel sweep in ONE pass.
//
// A red node (I + J even) relaxes against its black neighbours' old values. A
// black node needs its four red neighbours' NEW values, so it computes them
// itself from their own (black, old) neighbours. The result is exactly the
// two-pass red-black sweep, bit for bit, for one pass's overhead: these grids
// are small enough that the pass, not the arithmetic, is what costs.
//---------------------------------------------------------------------------
const char* const kSmoothShader = R"(#version 410 core

uniform sampler2D Psi;
uniform sampler2D Coef;
uniform sampler2D Rhs;
uniform ivec2 Nodes;
uniform int ZeroGuess;       //1: the old values are all zero (a fresh coarse level)

out vec4 fragColor;

bool onWall( ivec2 n )
{
	return n.x <= 0 || n.y <= 0 || n.x >= Nodes.x - 1 || n.y >= Nodes.y - 1;
}

float oldAt( ivec2 n )
{
	if( ZeroGuess != 0 || onWall( n ) )
		return 0.0;
	return texelFetch( Psi, n, 0 ).x;
}

//Gauss-Seidel at n, given its four neighbours' values.
float relax( ivec2 n, float east, float west, float north, float south )
{
	vec4 c   = texelFetch( Coef, n, 0 );
	float aW = texelFetch( Coef, n - ivec2( 1, 0 ), 0 ).x;
	float aS = texelFetch( Coef, n - ivec2( 0, 1 ), 0 ).y;
	float r  = texelFetch( Rhs, n, 0 ).x;
	return ( r + c.x * east + aW * west + c.y * north + aS * south ) / ( c.x + aW + c.y + aS + c.z );
}

float redNew( ivec2 n )
{
	if( onWall( n ) )
		return 0.0;
	return relax( n, oldAt( n + ivec2( 1, 0 ) ), oldAt( n - ivec2( 1, 0 ) ), oldAt( n + ivec2( 0, 1 ) ),
	              oldAt( n - ivec2( 0, 1 ) ) );
}

void main()
{
	ivec2 n = ivec2( gl_FragCoord.xy );
	if( onWall( n ) )
	{
		fragColor = vec4( 0.0 );
		return;
	}

	float value;
	if( ( ( n.x + n.y ) & 1 ) == 0 )
		value = redNew( n );
	else
		value = relax( n, redNew( n + ivec2( 1, 0 ) ), redNew( n - ivec2( 1, 0 ) ), redNew( n + ivec2( 0, 1 ) ),
		               redNew( n - ivec2( 0, 1 ) ) );

	fragColor = vec4( value, 0.0, 0.0, 0.0 );
}
)";

//---------------------------------------------------------------------------
// 5b. restrict: the residual on the fine level, full-weighted onto the coarse.
//
// Both sides of the fine equations carry the cell's area (the left as the
// stencil of a Laplacian that was never divided by h^2, the right as a
// circulation round an area), so the coarse right-hand side is FOUR times the
// full-weighting average: weights 1, 1/2, 1/4.
//---------------------------------------------------------------------------
const char* const kRestrictShader = R"(#version 410 core

uniform sampler2D Psi;       //fine
uniform sampler2D Coef;      //fine
uniform sampler2D Rhs;       //fine
uniform ivec2 FineNodes;
uniform ivec2 CoarseNodes;

out vec4 fragColor;

bool onFineWall( ivec2 n )
{
	return n.x <= 0 || n.y <= 0 || n.x >= FineNodes.x - 1 || n.y >= FineNodes.y - 1;
}

float psiAt( ivec2 n )
{
	return onFineWall( n ) ? 0.0 : texelFetch( Psi, n, 0 ).x;
}

float residual( ivec2 n )
{
	if( onFineWall( n ) )
		return 0.0;
	vec4 c    = texelFetch( Coef, n, 0 );
	float aW  = texelFetch( Coef, n - ivec2( 1, 0 ), 0 ).x;
	float aS  = texelFetch( Coef, n - ivec2( 0, 1 ), 0 ).y;
	float p   = psiAt( n );
	float lhs = c.x * ( p - psiAt( n + ivec2( 1, 0 ) ) ) + aW * ( p - psiAt( n - ivec2( 1, 0 ) ) )
	            + c.y * ( p - psiAt( n + ivec2( 0, 1 ) ) ) + aS * ( p - psiAt( n - ivec2( 0, 1 ) ) ) + c.z * p;
	return texelFetch( Rhs, n, 0 ).x - lhs;
}

void main()
{
	ivec2 n = ivec2( gl_FragCoord.xy );
	if( n.x <= 0 || n.y <= 0 || n.x >= CoarseNodes.x - 1 || n.y >= CoarseNodes.y - 1 )
	{
		fragColor = vec4( 0.0 );
		return;
	}

	ivec2 f = 2 * n;
	float r = residual( f )
	          + 0.5 * ( residual( f + ivec2( 1, 0 ) ) + residual( f - ivec2( 1, 0 ) ) + residual( f + ivec2( 0, 1 ) )
	                    + residual( f - ivec2( 0, 1 ) ) )
	          + 0.25 * ( residual( f + ivec2( 1, 1 ) ) + residual( f + ivec2( -1, 1 ) ) + residual( f + ivec2( 1, -1 ) )
	                     + residual( f + ivec2( -1, -1 ) ) );
	fragColor = vec4( r, 0.0, 0.0, 0.0 );
}
)";

//---------------------------------------------------------------------------
// 5c. coarsest: solve the whole coarsest grid in every fragment.
//
// At most 16 x 16 nodes. Each fragment runs the same lexicographic SOR on the
// same data in the same order, so every fragment holds the same answer, and
// writes out its own node. One pass instead of a hundred.
//---------------------------------------------------------------------------
const char* const kCoarsestShader = R"(#version 410 core

uniform sampler2D Coef;
uniform sampler2D Rhs;
uniform ivec2 Nodes;
uniform int Sweeps;
uniform float Omega;

out vec4 fragColor;

float x[ 256 ];

void main()
{
	ivec2 n = ivec2( gl_FragCoord.xy );
	if( n.x <= 0 || n.y <= 0 || n.x >= Nodes.x - 1 || n.y >= Nodes.y - 1 )
	{
		fragColor = vec4( 0.0 );
		return;
	}

	for( int k = 0; k < 256; ++k )
		x[ k ] = 0.0;

	for( int sweep = 0; sweep < Sweeps; ++sweep )
		for( int J = 1; J < Nodes.y - 1; ++J )
			for( int I = 1; I < Nodes.x - 1; ++I )
			{
				ivec2 m  = ivec2( I, J );
				vec4 c   = texelFetch( Coef, m, 0 );
				float aW = texelFetch( Coef, m - ivec2( 1, 0 ), 0 ).x;
				float aS = texelFetch( Coef, m - ivec2( 0, 1 ), 0 ).y;
				float r  = texelFetch( Rhs, m, 0 ).x;
				int k    = J * 16 + I;
				float gs = ( r + c.x * x[ k + 1 ] + aW * x[ k - 1 ] + c.y * x[ k + 16 ] + aS * x[ k - 16 ] )
				           / ( c.x + aW + c.y + aS + c.z );
				x[ k ] += Omega * ( gs - x[ k ] );
			}

	fragColor = vec4( x[ n.y * 16 + n.x ], 0.0, 0.0, 0.0 );
}
)";

//---------------------------------------------------------------------------
// 5d. prolong: add the coarse correction, interpolated bilinearly, and smooth
// once more -- in the same pass, the red-black sweep reading the corrected
// values it needs on the fly, exactly as the smoother does its red ones.
//---------------------------------------------------------------------------
const char* const kProlongShader = R"(#version 410 core

uniform sampler2D Psi;        //fine, before the correction
uniform sampler2D Correction; //coarse
uniform sampler2D Coef;       //fine
uniform sampler2D Rhs;        //fine
uniform ivec2 Nodes;          //fine
uniform float Weight;         //1: the coarse-grid correction, whole

out vec4 fragColor;

bool onWall( ivec2 n )
{
	return n.x <= 0 || n.y <= 0 || n.x >= Nodes.x - 1 || n.y >= Nodes.y - 1;
}

//psi with the coarse correction added: bilinear between coarse nodes, which
//for a fine node is one, two or four of them.
float corrected( ivec2 n )
{
	if( onWall( n ) )
		return 0.0;
	ivec2 h   = n / 2;
	ivec2 odd = n - 2 * h;
	float e   = 0.25 * ( texelFetch( Correction, h, 0 ).x + texelFetch( Correction, h + ivec2( odd.x, 0 ), 0 ).x
	                   + texelFetch( Correction, h + ivec2( 0, odd.y ), 0 ).x + texelFetch( Correction, h + odd, 0 ).x );
	return texelFetch( Psi, n, 0 ).x + Weight * e;
}

float relax( ivec2 n, float east, float west, float north, float south )
{
	vec4 c   = texelFetch( Coef, n, 0 );
	float aW = texelFetch( Coef, n - ivec2( 1, 0 ), 0 ).x;
	float aS = texelFetch( Coef, n - ivec2( 0, 1 ), 0 ).y;
	float r  = texelFetch( Rhs, n, 0 ).x;
	return ( r + c.x * east + aW * west + c.y * north + aS * south ) / ( c.x + aW + c.y + aS + c.z );
}

float redNew( ivec2 n )
{
	if( onWall( n ) )
		return 0.0;
	return relax( n, corrected( n + ivec2( 1, 0 ) ), corrected( n - ivec2( 1, 0 ) ), corrected( n + ivec2( 0, 1 ) ),
	              corrected( n - ivec2( 0, 1 ) ) );
}

void main()
{
	ivec2 n = ivec2( gl_FragCoord.xy );
	if( onWall( n ) )
	{
		fragColor = vec4( 0.0 );
		return;
	}

	float value;
	if( ( ( n.x + n.y ) & 1 ) == 0 )
		value = redNew( n );
	else
		value = relax( n, redNew( n + ivec2( 1, 0 ) ), redNew( n - ivec2( 1, 0 ) ), redNew( n + ivec2( 0, 1 ) ),
		               redNew( n - ivec2( 0, 1 ) ) );
	fragColor = vec4( value, 0.0, 0.0, 0.0 );
}
)";

//---------------------------------------------------------------------------
// 7. update: one substep of phi and T's transport, T's heat, and the dye.
// Cahn-Hilliard follows, subcycled, in kInterfaceShader.
//
// Every flux is a function of its face alone, and the two cells that share a
// face call it with the same arguments in the same order -- so they agree to
// the bit, the sum over the lamp telescopes, and wax and heat are conserved to
// the rounding of each cell's own sum (`bstest --volume`, `--heat`).
//
// The velocity comes from psi at the cell's corners, so it is divergence-free
// by construction whatever the solver's residual: each face's flux is a
// difference of two corner values, and round a cell they cancel.
//---------------------------------------------------------------------------
const char* const kUpdateShader = R"(#version 410 core

uniform sampler2D State;     //(phi, T, U, V)
uniform sampler2D Props;     //(mu^, rho', A, k)
uniform sampler2D Psi;       //finest nodes
uniform sampler2D Clip;      //the host's picture, for Clip Heat
uniform sampler2D Fastest;   //1 x 1: this substep's fastest face, m/s
uniform ivec2 Cells;
uniform vec2 Spacing;
uniform float Dt;            //seconds
uniform float Capacity;      //J/(m^3 K)
uniform float Ambient;       //C
uniform float FaceRate;      //2 h_f / ( C b ), 1/s
uniform float CapRate;       //h_cap / ( C dy ), 1/s, the top row only
uniform float BulbPower;     //W, through the base
uniform float BulbSigma;     //m
uniform float BulbNorm;      //1 / the bottom row's sum of the profile
uniform float Gap;           //m
uniform float ClipPower;     //W, when the clip is all white
uniform vec2 MaxUV;
uniform float UVRelax;       //1/s, for dye stretched past 4:1
uniform float SpeedCap;      //m/s: the face speed at the step's Courant number

out vec4 fragColor;

vec4 stateAt( ivec2 c )
{
	return texelFetch( State, clamp( c, ivec2( 0 ), Cells - 1 ), 0 );
}
vec4 propsAt( ivec2 c )
{
	return texelFetch( Props, clamp( c, ivec2( 0 ), Cells - 1 ), 0 );
}
//The step is chosen from last frame's fastest face with a margin. Should the
//flow have outrun the margin, the WHOLE streamfunction is scaled down to the
//step's Courant number -- never a face on its own, which would make the
//velocity divergent, and a conservative scheme then piles wax and heat into
//every converging cell without bound (it did: AGENTS.md). Scaled, the flow
//runs slow for one step and stays divergence-free.
float flowScale()
{
	float fastest = texelFetch( Fastest, ivec2( 0 ), 0 ).x;
	return fastest > SpeedCap ? SpeedCap / fastest : 1.0;
}

float psiAt( int I, int J )
{
	return texelFetch( Psi, ivec2( I, J ), 0 ).x;
}

//Limited slopes. Both keep the face value between the cell and its downwind
//neighbour, which with the Courant number held to 1/4 keeps every new value
//a convex combination of old ones: no new extrema.
float vanLeer( float back, float ahead )
{
	float p = back * ahead;
	return p > 0.0 ? 2.0 * p / ( back + ahead ) : 0.0;
}
float superbee( float back, float ahead )
{
	if( back * ahead <= 0.0 )
		return 0.0;
	float a = abs( back );
	float b = abs( ahead );
	return sign( back ) * max( min( 2.0 * a, b ), min( a, 2.0 * b ) );
}

//The advective flux of (phi, T) through a face with speed w, the face lying
//between q1 and q2, with q0 behind q1 and q3 ahead of q2. Superbee on the wax
//keeps its edge sharp; van Leer on the heat keeps it smooth.
vec2 advect( float w, vec4 q0, vec4 q1, vec4 q2, vec4 q3 )
{
	precise vec2 face;
	if( w >= 0.0 )
		face = vec2( q1.x + 0.5 * vanLeer( q1.x - q0.x, q2.x - q1.x ), q1.y + 0.5 * vanLeer( q1.y - q0.y, q2.y - q1.y ) );
	else
		face = vec2( q2.x - 0.5 * vanLeer( q2.x - q1.x, q3.x - q2.x ), q2.y - 0.5 * vanLeer( q2.y - q1.y, q3.y - q2.y ) );
	precise vec2 flux = w * face;
	return flux;
}

//Conduction through the face between cells a and b, as a temperature flux
//(W/m^2 over C). Cahn-Hilliard has its own pass: see kInterfaceShader.
vec2 diffuse( ivec2 a, ivec2 b, float h )
{
	vec4 pa = propsAt( a );
	vec4 pb = propsAt( b );
	precise float k    = pa.w + pb.w > 0.0 ? 2.0 * pa.w * pb.w / ( pa.w + pb.w ) : 0.0;
	precise float heat = -k * ( stateAt( b ).y - stateAt( a ).y ) / ( h * Capacity );
	return vec2( 0.0, heat );
}

//Everything through the x-face between cells (i, j) and (i+1, j), speed w.
vec2 fluxX( int i, int j, float w )
{
	vec4 q0 = stateAt( ivec2( i - 1, j ) );
	vec4 q1 = stateAt( ivec2( i, j ) );
	vec4 q2 = stateAt( ivec2( i + 1, j ) );
	vec4 q3 = stateAt( ivec2( i + 2, j ) );
	precise vec2 flux = advect( w, q0, q1, q2, q3 ) + diffuse( ivec2( i, j ), ivec2( i + 1, j ), Spacing.x );
	return flux;
}

//Everything through the y-face between cells (i, j) and (i, j+1), speed w.
vec2 fluxY( int i, int j, float w )
{
	vec4 q0 = stateAt( ivec2( i, j - 1 ) );
	vec4 q1 = stateAt( ivec2( i, j ) );
	vec4 q2 = stateAt( ivec2( i, j + 1 ) );
	vec4 q3 = stateAt( ivec2( i, j + 2 ) );
	precise vec2 flux = advect( w, q0, q1, q2, q3 ) + diffuse( ivec2( i, j ), ivec2( i, j + 1 ), Spacing.y );
	return flux;
}

//The dye's texture coordinate at a point in cell units, bilinearly.
vec2 dyeAt( vec2 p )
{
	p = clamp( p - 0.5, vec2( 0.0 ), vec2( Cells - 1 ) );
	ivec2 c = ivec2( floor( p ) );
	vec2 t  = p - vec2( c );
	vec2 a  = mix( stateAt( c ).zw, stateAt( c + ivec2( 1, 0 ) ).zw, t.x );
	vec2 b  = mix( stateAt( c + ivec2( 0, 1 ) ).zw, stateAt( c + ivec2( 1, 1 ) ).zw, t.x );
	return mix( a, b, t.y );
}
)"
R"(
void main()
{
	ivec2 c = ivec2( gl_FragCoord.xy );
	int i   = c.x;
	int j   = c.y;
	vec4 s  = stateAt( c );

	//The four face speeds, from psi at the corners. A wall face's two corners
	//are both on the wall, where psi is 0, so it passes nothing.
	//Every flux below is computed twice, once by each cell that shares its
	//face, and conservation rests on the two agreeing to the bit. This
	//compiler reassociates and fuses float arithmetic unless told not to
	//(AGENTS.md), so all of it is `precise`.
	precise float scale = flowScale();
	precise float uL = scale * ( ( psiAt( i, j + 1 ) - psiAt( i, j ) ) / Spacing.y );
	precise float uR = scale * ( ( psiAt( i + 1, j + 1 ) - psiAt( i + 1, j ) ) / Spacing.y );
	precise float vB = scale * ( -( psiAt( i + 1, j ) - psiAt( i, j ) ) / Spacing.x );
	precise float vT = scale * ( -( psiAt( i + 1, j + 1 ) - psiAt( i, j + 1 ) ) / Spacing.x );

	precise vec2 left   = fluxX( i - 1, j, uL );
	precise vec2 right  = fluxX( i, j, uR );
	precise vec2 bottom = fluxY( i, j - 1, vB );
	precise vec2 top    = fluxY( i, j, vT );

	precise vec2 change = -( ( right - left ) / Spacing.x + ( top - bottom ) / Spacing.y );

	//Heat in and out, degrees per second.
	float T      = s.y;
	float source = -FaceRate * ( T - Ambient );
	if( j == Cells.y - 1 )
		source -= CapRate * ( T - Ambient );
	if( j == 0 )
	{
		float x    = ( float( i ) + 0.5 ) * Spacing.x - 0.5 * float( Cells.x ) * Spacing.x;
		float lamp = exp( -0.5 * ( x / BulbSigma ) * ( x / BulbSigma ) ) * BulbNorm;
		source += BulbPower * lamp / ( Capacity * Gap * Spacing.x * Spacing.y );
	}
	vec2 at = ( vec2( c ) + 0.5 ) / vec2( Cells );
	if( ClipPower > 0.0 )
	{
		float luma = dot( texture( Clip, at * MaxUV ).rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
		source += ClipPower * luma / ( Capacity * Gap * float( Cells.x * Cells.y ) * Spacing.x * Spacing.y );
	}

	precise float phi = s.x + Dt * change.x;
	precise float Tn  = T + Dt * ( change.y + source );

	//The dye, back-traced along the cell's mean velocity.
	vec2 velocity = vec2( 0.5 * ( uL + uR ), 0.5 * ( vB + vT ) );
	vec2 from     = vec2( c ) + 0.5 - Dt * velocity / Spacing;
	vec2 dye      = dyeAt( from );

	//Stretched past 4:1 either way, the dye relaxes back towards where it would
	//be at rest, faster the further past; see AGENTS.md.
	ivec2 e = min( c + ivec2( 1, 0 ), Cells - 1 ), w = max( c - ivec2( 1, 0 ), ivec2( 0 ) );
	ivec2 n = min( c + ivec2( 0, 1 ), Cells - 1 ), so = max( c - ivec2( 0, 1 ), ivec2( 0 ) );
	vec2 dx = ( stateAt( e ).zw - stateAt( w ).zw ) * float( Cells.x ) / max( float( e.x - w.x ), 1.0 );
	vec2 dy = ( stateAt( n ).zw - stateAt( so ).zw ) * float( Cells.y ) / max( float( n.y - so.y ), 1.0 );
	float a  = dot( dx, dx ), b = dot( dx, dy ), d = dot( dy, dy );
	float mid = 0.5 * ( a + d ), spread = sqrt( max( mid * mid - ( a * d - b * b ), 0.0 ) );
	float big = sqrt( mid + spread ), small = sqrt( max( mid - spread, 1e-12 ) );
	float stretch = max( big, 1.0 / small );
	if( stretch > 4.0 )
		dye = mix( dye, at, min( 1.0, Dt * UVRelax * ( stretch - 4.0 ) / 4.0 ) );

	fragColor = vec4( phi, Tn, dye );
}
)";


//---------------------------------------------------------------------------
// 7b. interface: one Cahn-Hilliard subcycle, phi alone, in flux form:
//
//     d phi / dt = div( D M( phi ) grad mu^ ),   M = max( 4 phi ( 1 - phi ), floor )
//
// The same bit-identical-face construction as the update, so it conserves wax
// to rounding. Subcycled because its explicit limit is the stiffest in the
// lamp (fourth order), and a subcycle is two passes where a flow step is thirty.
//---------------------------------------------------------------------------
const char* const kInterfaceShader = R"(#version 410 core

uniform sampler2D State;     //(phi, T, U, V)
uniform sampler2D Props;     //(mu^, ...), from this state
uniform ivec2 Cells;
uniform vec2 Spacing;
uniform float Dt;
uniform float Mobility;      //D, m^2/s
uniform float MobilityFloor;

out vec4 fragColor;

vec4 stateAt( ivec2 c )
{
	return texelFetch( State, clamp( c, ivec2( 0 ), Cells - 1 ), 0 );
}
float potentialAt( ivec2 c )
{
	return texelFetch( Props, clamp( c, ivec2( 0 ), Cells - 1 ), 0 ).x;
}

//The flux from a to b, per unit length of face; a wall face has a and b the
//same cell, so nothing crosses it.
float flux( ivec2 a, ivec2 b, float h )
{
	precise float phiFace  = 0.5 * ( stateAt( a ).x + stateAt( b ).x );
	precise float mobility = Mobility * max( 4.0 * phiFace * ( 1.0 - phiFace ), MobilityFloor );
	precise float f        = -mobility * ( potentialAt( b ) - potentialAt( a ) ) / h;
	return f;
}

void main()
{
	ivec2 c = ivec2( gl_FragCoord.xy );
	vec4 s  = stateAt( c );
	precise float left   = flux( c - ivec2( 1, 0 ), c, Spacing.x );
	precise float right  = flux( c, c + ivec2( 1, 0 ), Spacing.x );
	precise float bottom = flux( c - ivec2( 0, 1 ), c, Spacing.y );
	precise float top    = flux( c, c + ivec2( 0, 1 ), Spacing.y );
	precise float phi    = s.x - Dt * ( ( right - left ) / Spacing.x + ( top - bottom ) / Spacing.y );
	fragColor = vec4( phi, s.yzw );
}
)";

//---------------------------------------------------------------------------
// 6. reduce: the fastest face speed, from psi. Mode 0 reads psi in blocks;
// mode 1 takes the max of what mode 0 wrote.
//---------------------------------------------------------------------------
const char* const kReduceShader = R"(#version 410 core

uniform sampler2D Source;
uniform ivec2 SourceSize;
uniform vec2 Spacing;
uniform int Mode;
uniform int Block;

out vec4 fragColor;

void main()
{
	ivec2 base  = ivec2( gl_FragCoord.xy ) * Block;
	float speed = 0.0;
	for( int y = 0; y < Block; ++y )
		for( int x = 0; x < Block; ++x )
		{
			ivec2 n = base + ivec2( x, y );
			if( n.x >= SourceSize.x || n.y >= SourceSize.y )
				continue;
			float here = texelFetch( Source, n, 0 ).x;
			if( Mode != 0 )
			{
				speed = max( speed, here );
				continue;
			}
			if( n.x + 1 < SourceSize.x )
				speed = max( speed, abs( texelFetch( Source, n + ivec2( 1, 0 ), 0 ).x - here ) / Spacing.x );
			if( n.y + 1 < SourceSize.y )
				speed = max( speed, abs( texelFetch( Source, n + ivec2( 0, 1 ), 0 ).x - here ) / Spacing.y );
		}
	fragColor = vec4( speed, 0.0, 0.0, 0.0 );
}
)";

//---------------------------------------------------------------------------
// The events: Reset (the cold slab), Warm (the lumped steady temperature),
// Pour (the clip's bright parts become wax).
//---------------------------------------------------------------------------
const char* const kEventShader = R"(#version 410 core

uniform sampler2D State;
uniform sampler2D Clip;
uniform int Mode;            //0 reset, 1 warm, 2 pour
uniform ivec2 Cells;
uniform vec2 Spacing;
uniform float Xi;            //m
uniform float WaxHeight;     //m, the slab's top on Reset
uniform float Temperature;   //C: Ambient on Reset, the warm lamp on Warm
uniform float Threshold;     //luma, on Pour
uniform vec2 MaxUV;

out vec4 fragColor;

void main()
{
	ivec2 c = ivec2( gl_FragCoord.xy );
	vec4 s  = texelFetch( State, c, 0 );
	vec2 at = ( vec2( c ) + 0.5 ) / vec2( Cells );

	if( Mode == 0 )
	{
		float y = at.y * float( Cells.y ) * Spacing.y;
		fragColor = vec4( 1.0 / ( 1.0 + exp( -( WaxHeight - y ) / Xi ) ), Temperature, at );
	}
	else if( Mode == 1 )
		fragColor = vec4( s.x, Temperature, s.zw );
	else
	{
		float luma = dot( texture( Clip, at * MaxUV ).rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
		float wax  = smoothstep( Threshold - 0.04, Threshold + 0.04, luma );
		fragColor  = vec4( max( s.x, wax ), s.y, wax > 0.5 ? at : s.zw );
	}
}
)";

//---------------------------------------------------------------------------
// 8. inflate: the lens-thickness problem's operator (Mode 0) and right-hand
// side (Mode 1), at the finest nodes. -lap h = phi in the wax; in the water a
// penalty pins h to zero.
//---------------------------------------------------------------------------
const char* const kInflateShader = R"(#version 410 core

uniform sampler2D State;
uniform ivec2 Cells;
uniform vec2 Spacing;
uniform int Mode;
uniform float Penalty;

out vec4 fragColor;

float phiAt( ivec2 c )
{
	return clamp( texelFetch( State, clamp( c, ivec2( 0 ), Cells - 1 ), 0 ).x, 0.0, 1.0 );
}

void main()
{
	ivec2 n   = ivec2( gl_FragCoord.xy );
	float wax = 0.25 * ( phiAt( n ) + phiAt( n - ivec2( 1, 0 ) ) + phiAt( n - ivec2( 0, 1 ) ) + phiAt( n - ivec2( 1, 1 ) ) );
	if( Mode == 0 )
		fragColor = vec4( Spacing.y / Spacing.x, Spacing.x / Spacing.y, Penalty * ( 1.0 - wax ) * ( 1.0 - wax ), 0.0 );
	else
		fragColor = vec4( wax * Spacing.x * Spacing.y, 0.0, 0.0, 0.0 );
}
)";

//---------------------------------------------------------------------------
// 9. composite.
//---------------------------------------------------------------------------
const char* const kCompositeShader = R"(#version 410 core

uniform sampler2D InputTexture;
uniform sampler2D StateTexture;  //cells, linear
uniform sampler2D ThickTexture;  //nodes, h: thickness is 4 sqrt( h )
uniform sampler2D PsiTexture;    //nodes
uniform vec2 MaxUV;
uniform ivec2 Cells;
uniform vec2 Spacing;
uniform vec2 LampSize;           //W, H, metres
uniform float Gap;
uniform int ClipMode;            //0 Behind, 1 Dyed
uniform int GlassMode;           //0 Flat, 1 Cylinder
uniform int ViewMode;            //0 Lamp, 1 Temperature, 2 Wax, 3 Velocity, 4 Density
uniform float MixAmount;
uniform vec3 WaxColour;          //seen through kWaxReference of wax
uniform vec3 Tint;               //seen straight through the lamp's middle
uniform float Glow;
uniform float BulbFraction;      //the bulb's power over 60 W
uniform float Distance;          //m, from the lamp's back to the world behind
uniform float WaxIndex;
uniform float WaterIndex;
uniform float Slope;
uniform float Crossover;
uniform float LiquidBeta;
uniform float ReferenceT;
uniform int UseThickness;

in vec2 uv;
out vec4 fragColor;

const float kWaxReference = 0.03;//m
const vec3 kWarm = vec3( 1.0, 0.62, 0.28 );

vec3 clipAt( vec2 p )
{
	return texture( InputTexture, clamp( p, vec2( 0.0 ), vec2( 1.0 ) ) * MaxUV ).rgb;
}

//The lamp's fields at a point on its middle plane, in metres.
vec4 stateAt( vec2 p )
{
	return texture( StateTexture, p / ( vec2( Cells ) * Spacing ) );
}
float nodeField( sampler2D field, vec2 p )
{
	vec2 nodes = vec2( Cells ) + 1.0;
	return texture( field, ( p / Spacing + 0.5 ) / nodes ).x;
}
float thicknessAt( vec2 p )
{
	return UseThickness != 0 ? 4.0 * sqrt( max( nodeField( ThickTexture, p ), 0.0 ) ) : 0.0;
}

vec3 ramp( float t )
{
	t = clamp( t, 0.0, 1.0 );
	return clamp( vec3( 1.6 * t - 0.1, 2.2 * t * t - 0.3 * t, 0.6 * sin( 3.14159 * t ) + max( 3.0 * t - 2.2, 0.0 ) ),
	              0.0, 1.0 );
}

void main()
{
	vec3 clip = clipAt( uv );
	float W   = LampSize.x;
	float H   = LampSize.y;

	//The glass. Where the view ray crosses the lamp's middle plane (slice),
	//where it lands on the world behind (world), and how much liquid it
	//crosses (path), all as fractions of the frame except path.
	float slice = uv.x;
	float world = uv.x;
	float path  = Gap;
	float pathRef = Gap;
	float behind  = Distance + 0.5 * Gap;
	float across  = 0.0;
	if( GlassMode == 1 )
	{
		//A vertical cylinder of radius W / 2, seen side on. Snell at the
		//front (air to water; a thin wall of uniform thickness offsets the ray
		//without turning it), straight across, Snell again at the back.
		float X   = clamp( 2.0 * uv.x - 1.0, -0.9999, 0.9999 );
		float ti  = asin( X );
		float tt  = asin( X / WaterIndex );
		float d1  = ti - tt;
		float zs  = sqrt( 1.0 - X * X );
		vec2 dir  = vec2( -sin( d1 ), -cos( d1 ) );
		vec2 into = vec2( X, zs );
		float chord = -2.0 * dot( into, dir );
		vec2 exit = into + chord * dir;
		float zb  = -1.0 - Distance / ( 0.5 * W );
		float run = ( exit.y - zb ) / cos( 2.0 * d1 );
		float wx  = exit.x - run * sin( 2.0 * d1 );
		slice   = 0.5 + 0.5 * ( X - zs * tan( d1 ) );
		world   = 0.5 + 0.5 * wx;
		path    = chord * 0.5 * W;
		pathRef = W;
		behind  = Distance + 0.5 * W;
		across  = X;
	}

	vec2 p   = vec2( slice * W, uv.y * H );
	vec4 s   = stateAt( p );
	float y  = p.y;
	vec3 col;

	if( ViewMode == 1 )
		col = ramp( ( s.y - 10.0 ) / 70.0 );
	else if( ViewMode == 2 )
		col = vec3( clamp( s.x, 0.0, 1.0 ) );
	else if( ViewMode == 3 )
	{
		float u = ( nodeField( PsiTexture, p + vec2( 0.0, 0.5 * Spacing.y ) ) - nodeField( PsiTexture, p - vec2( 0.0, 0.5 * Spacing.y ) ) ) / Spacing.y;
		float v = -( nodeField( PsiTexture, p + vec2( 0.5 * Spacing.x, 0.0 ) ) - nodeField( PsiTexture, p - vec2( 0.5 * Spacing.x, 0.0 ) ) ) / Spacing.x;
		float speed = length( vec2( u, v ) ) / 0.005;
		vec2 d = speed > 0.0 ? normalize( vec2( u, v ) ) : vec2( 0.0 );
		col = clamp( speed, 0.0, 1.0 ) * ( 0.55 + 0.45 * vec3( d.x, 0.5 * ( d.y - d.x ), -d.y ) );
	}
	else if( ViewMode == 4 )
	{
		float phi = clamp( s.x, 0.0, 1.0 );
		float rho = phi * Slope * ( Crossover - s.y ) - LiquidBeta * ( s.y - ReferenceT );
		float r   = clamp( rho / 10.0, -1.0, 1.0 );
		col = r > 0.0 ? mix( vec3( 0.5 ), vec3( 0.1, 0.3, 1.0 ), r ) : mix( vec3( 0.5 ), vec3( 1.0, 0.3, 0.1 ), -r );
	}
	else
	{
		//The wax's lens: thickness from the inflated distance field, and its
		//gradient by a centred difference a cell wide.
		float t  = thicknessAt( p );
		vec2 grad = vec2( thicknessAt( p + vec2( 0.5 * Spacing.x, 0.0 ) ) - thicknessAt( p - vec2( 0.5 * Spacing.x, 0.0 ) ),
		                  thicknessAt( p + vec2( 0.0, 0.5 * Spacing.y ) ) - thicknessAt( p - vec2( 0.0, 0.5 * Spacing.y ) ) )
		            / Spacing;
		//A thin prism turns a ray by ( n_wax - n_water ) times its wedge,
		//towards the thicker side; out through the flat back into air.
		vec2 turn  = clamp( ( WaxIndex - WaterIndex ) * grad, vec2( -0.5 ), vec2( 0.5 ) );
		vec2 shift = behind * turn / vec2( W, H );

		//The bulb's light, falling off up the lamp; the wax scatters it.
		float light = Glow * BulbFraction * exp( -y / ( 0.35 * H ) );
		float base  = Glow * BulbFraction * 0.8 * exp( -y / ( 0.04 * H ) )
		              * exp( -( ( p.x - 0.5 * W ) / ( 0.25 * W ) ) * ( ( p.x - 0.5 * W ) / ( 0.25 * W ) ) );

		vec3 waxTrans = pow( max( WaxColour, vec3( 1e-3 ) ), vec3( t / kWaxReference ) );
		vec3 liquid   = pow( max( Tint, vec3( 1e-3 ) ), vec3( max( path - t, 0.0 ) / pathRef ) );

		if( ClipMode == 0 )
		{
			vec3 seen = clipAt( vec2( world, uv.y ) + shift );
			col = seen * liquid * waxTrans + liquid * ( kWarm * base + WaxColour * kWarm * light * ( 1.0 - exp( -t / 0.01 ) ) );
		}
		else
		{
			//The wax is the picture, drawn with a crisp edge at phi = 1/2.
			float cover = clamp( 0.5 + ( s.x - 0.5 ) / max( fwidth( s.x ), 1e-4 ), 0.0, 1.0 );
			vec3 wax    = clipAt( s.zw ) * ( 1.0 + kWarm * light );
			vec3 water  = liquid * kWarm * ( base + 0.25 * light );
			col = mix( water, wax, cover );
		}

		//The cylinder's highlight: a soft window reflected in the glass.
		if( GlassMode == 1 )
			col += vec3( 0.35 * Glow * exp( -( ( across - 0.55 ) / 0.04 ) * ( ( across - 0.55 ) / 0.04 ) ) );
	}

	fragColor = vec4( mix( clip, col, MixAmount ), 1.0 );
}
)";

const char* const kShipped[] = {
	kVertexShader,   kPropsShader, kCoefShader,    kCirculationShader, kCoarsenShader, kSmoothShader,
	kRestrictShader, kCoarsestShader, kProlongShader, kUpdateShader,  kInterfaceShader, kReduceShader,
	kEventShader,    kInflateShader,  kCompositeShader,
};
static_assert( sizeof( kShipped ) / sizeof( kShipped[ 0 ] ) == static_cast< size_t >( ShaderId::Count ),
               "one shipped source per ShaderId" );

const char* const kNames[] = {
	"vertex", "props", "coef", "circulation", "coarsen", "smooth", "restrict",
	"coarsest", "prolong", "update", "interface", "reduce", "event", "inflate", "composite",
};

std::map< int, std::string >& overrides()
{
	static std::map< int, std::string > table;
	return table;
}
} // namespace

const char* ShaderSource( ShaderId id )
{
	const auto found = overrides().find( static_cast< int >( id ) );
	if( found != overrides().end() )
		return found->second.c_str();
	return ShippedSource( id );
}

const char* ShippedSource( ShaderId id )
{
	return kShipped[ static_cast< int >( id ) ];
}

void SetShaderOverride( ShaderId id, const std::string& source )
{
	overrides()[ static_cast< int >( id ) ] = source;
}

void ClearShaderOverrides()
{
	overrides().clear();
}

const char* ShaderName( ShaderId id )
{
	return kNames[ static_cast< int >( id ) ];
}

} // namespace bassalt
