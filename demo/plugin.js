/**
 * Bassalt — browser demo.
 *
 * The fifteen constants below (`VERTEX` … `COMPOSITE`) are every shader in
 * `source/Shaders.cpp`, copied across unedited — the update pass is two
 * adjacent raw strings in the C++, joined here exactly as the compiler joins
 * them. `demo/tools/check_shaders.py` compares them character for character
 * with the C++ and is called from `tools/verify.sh`, because two copies of a
 * shader is exactly the arrangement that drifts, invisibly, from both sides.
 * The one escape is galvo's: a comment in the update pass quotes `precise` in
 * backticks, and a backtick inside a JS template literal has to be written \`.
 * The check undoes that escape and refuses any other backslash.
 *
 * ------------------------------------------------------ what is ported, and how
 *
 * Bassalt's simulation is all GPU, so the port is the ORCHESTRATION: which pass
 * runs when, into which buffer, at which format, with which uniforms. Everything
 * below the shaders is a hand translation of the C++:
 *
 *   controls         source/Controls.cpp, one function per control
 *   physics          source/Physics.{h,cpp}: the constants, T*, the grid, the
 *                    heat, interface and capillary step limits, the bulb's
 *                    one-pole
 *   createRenderer   BassaltPlugin in source/Bassalt.cpp: the buffers, the
 *                    events, Simulate()'s substep loop, the V-cycle recursion,
 *                    the speed read back a frame late, the inflation and the
 *                    composite
 *
 * **Nothing checks that port but a reader.** When a mapping, a constant or the
 * pass order changes in the C++, it has to change here too.
 *
 * ------------------------------------------------------------- the differences
 *
 * **`precise` is dropped.** The kit's port() removes it because GLSL ES 3.00 has
 * no such keyword. Bassalt marks the circulation and every flux `precise`
 * because Apple's compiler reassociated the arithmetic and a level lamp's
 * streamfunction moved by 1e-14 until it was added (AGENTS.md). A browser's
 * compiler is free to do the same, so here a level slab need not stay exactly
 * still and wax need not be conserved to the bit, as the plugin's --still and
 * --volume checks require of the plugin. The page says so.
 *
 * **No audio.** The plugin's Audio group — the 64-bin FFT buffer, Bass, Bass
 * Band and Kick — is absent from the panel rather than present and dead. The
 * removal is exact: with silence in the buffer (which is all a browser could
 * offer without asking for a microphone) the plugin's analyser reports a bass
 * level of 0 and never fires an onset, so BulbTarget() is the Bulb control alone
 * and no Kick is ever queued. That is the state this page computes. The bulb is
 * driven by the plugin's own Bulb control at its own default, 30 W — nothing
 * invented stands in for the music.
 *
 * **Warm, Reset and Pour are toggles, not events.** FFGL has FF_TYPE_EVENT and
 * the kit's parameter model does not, so they are booleans that the renderer
 * releases the moment it has taken the press — what a host does with an event,
 * and why the buttons blink (readout's precedent).
 *
 * **Formats.** The same as the plugin: RGBA32F cells, R32F multigrid levels,
 * RGBA32F coefficients. WebGL2 renders to them only with EXT_color_buffer_float,
 * and the composite filters the state and psi linearly, which a float texture
 * only allows with OES_texture_float_linear — without it the texture is
 * incomplete and every fetch reads zero, so the page refuses to start rather
 * than show a plausible wrong lamp.
 *
 * **The fastest face** is read back a frame late through a pixel-pack buffer and
 * a fence, as the plugin does. The plugin WAITS on the fence, so every run takes
 * the same steps; WebGL2 allows no client wait, so here the read is collected
 * only once the fence has signalled, and a frame whose GPU is behind sizes its
 * steps from the last speed that did arrive. The step sequence therefore
 * depends on the GPU's timing here, where in the plugin it does not.
 *
 * **The About block is absent**: a text line and three buttons that open a
 * browser.
 */

import { mountDemo } from './vendor/demo.js';
import { Program, PassBuffer, GLError } from './vendor/gl.js';

//---------------------------------------------------------------------------
// Shaders — verbatim from source/Shaders.cpp. Do not edit here.
//---------------------------------------------------------------------------

const VERTEX = `#version 410 core

layout( location = 0 ) in vec4 vPosition;
layout( location = 1 ) in vec2 vUV;

out vec2 uv;

void main()
{
	gl_Position = vPosition;
	uv          = vUV;
}
`;

const PROPS = `#version 410 core

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
`;

const COEF = `#version 410 core

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
`;

const CIRCULATION = `#version 410 core

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
`;

const COARSEN = `#version 410 core

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
`;

const SMOOTH = `#version 410 core

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
`;

const RESTRICT = `#version 410 core

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
`;

const COARSEST = `#version 410 core

uniform sampler2D Coef;
uniform sampler2D Rhs;
uniform ivec2 Nodes;
uniform int Sweeps;
uniform float Omega;

out vec4 fragColor;

float x[ 64 ];//kCoarsestMax squared

void main()
{
	ivec2 n = ivec2( gl_FragCoord.xy );
	if( n.x <= 0 || n.y <= 0 || n.x >= Nodes.x - 1 || n.y >= Nodes.y - 1 )
	{
		fragColor = vec4( 0.0 );
		return;
	}

	for( int k = 0; k < 64; ++k )
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
				int k    = J * 8 + I;
				float gs = ( r + c.x * x[ k + 1 ] + aW * x[ k - 1 ] + c.y * x[ k + 8 ] + aS * x[ k - 8 ] )
				           / ( c.x + aW + c.y + aS + c.z );
				x[ k ] += Omega * ( gs - x[ k ] );
			}

	fragColor = vec4( x[ n.y * 8 + n.x ], 0.0, 0.0, 0.0 );
}
`;

const PROLONG = `#version 410 core

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

//The coarse correction at a fine node, interpolated by the operator rather
//than bilinearly. Between two coarse nodes the correction is taken
//flux-continuous through the fine node -- a_W ( e - e_W ) = a_E ( e_E - e) --
//so across a jump in the wax's resistivity it bends where the flux says it
//must. Bilinear interpolation does not, and on a cold slab (solid wax 10^5
//times stiffer than the water on it) the V-cycle it made DIVERGED, eight
//times a cycle (AGENTS.md). A node between four coarse nodes is the same rule
//applied to its four edge-midpoint neighbours. (Alcouffe, Brandt, Dendy &
//Painter, 1981, for five-point stencils.)
float coarseAt( ivec2 h )
{
	return texelFetch( Correction, h, 0 ).x;
}
float alongX( ivec2 n )//n.x odd, n.y even
{
	ivec2 h  = n / 2;
	float aW = texelFetch( Coef, n - ivec2( 1, 0 ), 0 ).x;
	float aE = texelFetch( Coef, n, 0 ).x;
	return ( aW * coarseAt( h ) + aE * coarseAt( h + ivec2( 1, 0 ) ) ) / ( aW + aE );
}
float alongY( ivec2 n )//n.x even, n.y odd
{
	ivec2 h  = n / 2;
	float aS = texelFetch( Coef, n - ivec2( 0, 1 ), 0 ).y;
	float aN = texelFetch( Coef, n, 0 ).y;
	return ( aS * coarseAt( h ) + aN * coarseAt( h + ivec2( 0, 1 ) ) ) / ( aS + aN );
}
float interpolated( ivec2 n )
{
	ivec2 odd = n - 2 * ( n / 2 );
	if( odd.x == 0 && odd.y == 0 )
		return coarseAt( n / 2 );
	if( odd.y == 0 )
		return alongX( n );
	if( odd.x == 0 )
		return alongY( n );
	float aW = texelFetch( Coef, n - ivec2( 1, 0 ), 0 ).x;
	float aE = texelFetch( Coef, n, 0 ).x;
	float aS = texelFetch( Coef, n - ivec2( 0, 1 ), 0 ).y;
	float aN = texelFetch( Coef, n, 0 ).y;
	return ( aW * alongY( n - ivec2( 1, 0 ) ) + aE * alongY( n + ivec2( 1, 0 ) ) + aS * alongX( n - ivec2( 0, 1 ) )
	         + aN * alongX( n + ivec2( 0, 1 ) ) )
	       / ( aW + aE + aS + aN );
}

//psi with the coarse correction added.
float corrected( ivec2 n )
{
	if( onWall( n ) )
		return 0.0;
	return texelFetch( Psi, n, 0 ).x + Weight * interpolated( n );
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
`;

const UPDATE = `#version 410 core

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

//The limited slope. It keeps the face value between the cell and its
//downwind neighbour, which with the Courant number held to 1/4 keeps every
//new value a convex combination of old ones: no new extrema.
float vanLeer( float back, float ahead )
{
	float p = back * ahead;
	return p > 0.0 ? 2.0 * p / ( back + ahead ) : 0.0;
}

//The advective flux of (phi, T) through a face with speed w, the face lying
//between q1 and q2, with q0 behind q1 and q3 ahead of q2. Van Leer on both.
//Superbee on the wax, the first choice for its sharp edges, is anti-diffusive:
//it put back the interface energy Cahn-Hilliard took out, and a resting blob's
//spurious currents never died (AGENTS.md).
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
	//(AGENTS.md), so all of it is \`precise\`.
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
`;

const INTERFACE = `#version 410 core

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
`;

const REDUCE = `#version 410 core

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
`;

const EVENT = `#version 410 core

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
`;

const INFLATE = `#version 410 core

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
		//Pinned on the water's side of phi = 1/2 only, so the lens's edge is
		//the interface's middle. Pinned wherever there was ANY water, the
		//first version held h to zero two cells inside the interface and a
		//round blob came out a fifth too narrow.
		fragColor = vec4( Spacing.y / Spacing.x, Spacing.x / Spacing.y, Penalty * max( 1.0 - 2.0 * wax, 0.0 ) * max( 1.0 - 2.0 * wax, 0.0 ), 0.0 );
	else
		fragColor = vec4( wax * Spacing.x * Spacing.y, 0.0, 0.0, 0.0 );
}
`;

const COMPOSITE = `#version 410 core

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
`;

//---------------------------------------------------------------------------
// source/Controls.cpp. The C++ works in float; each result is rounded to float
// here (Math.fround), so a value agrees with the plugin's to float rounding —
// std::pow and Math.pow need not round the same last bit.
//---------------------------------------------------------------------------

const f32 = Math.fround;
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

function geometric(value, low, high) {
  return f32(low * Math.pow(high / low, clamp(value, 0, 1)));
}
function linear(value, low, high) {
  return f32(low + (high - low) * clamp(value, 0, 1));
}

const kSpeedLow = 1;
const kSpeedHigh = 300;

const controls = {
  lampHeight: (v) => geometric(v, 0.1, 1.0),
  gap: (v) => geometric(v, 0.001, 0.02),
  speed: (v) => geometric(v, kSpeedLow, kSpeedHigh),
  paramFromSpeed: (speed) => f32(Math.log(clamp(speed, kSpeedLow, kSpeedHigh) / kSpeedLow) / Math.log(kSpeedHigh / kSpeedLow)),
  ambient: (v) => linear(v, 10, 40),
  bulb: (v) => linear(v, 0, 60),
  bulbLag: (v) => geometric(v, 0.5, 120),
  coil: (v) => linear(v, 0, 6),
  salt: (v) => linear(v, 0, 4),
  waxAmount: (v) => linear(v, 0.05, 0.4),
  tension: (v) => { const c = clamp(v, 0, 1); return f32(0.01 * c * c); },
  waxViscosity: (v) => geometric(v, 0.001, 1.0),
  meltingPoint: (v) => linear(v, 35, 70),
  clipHeat: (v) => linear(v, 0, 60),
  glow: (v) => linear(v, 0, 2),
  refraction: (v) => linear(v, 0, 1),
};

/// Controls.h: grid cells up the lamp's height, per Detail element.
const kDetailCells = [64, 96, 128, 192, 256];

//---------------------------------------------------------------------------
// source/Physics.{h,cpp}. Double precision in the C++, and JavaScript numbers
// are doubles, so these agree exactly.
//---------------------------------------------------------------------------

const physics = {
  kGravity: 9.81,
  kReferenceT: 20.0,
  kWaxDensity: 1025.0,
  kWaxExpansion: 9.0e-4,
  kWaterDensity: 998.2,
  kLiquidExpansion: 3.0e-4,
  kSaltDensityPerPercent: 0.007,
  kLiquidViscosity: 2.0e-3,
  kFreezeWidth: 2.5,
  kSolidMultiple: 1.0e4,
  kHeatCapacity: 4.18e6,
  kLiquidConductivity: 0.6,
  kWaxConductivity: 0.24,
  kFaceLoss: 2.5,
  kCapLoss: 60.0,
  kCoilHeight: 0.03,
  kCoilWidth: 0.5,
  kBulbSpread: 0.15,
  kWaxIndex: 1.44,
  kWaterIndex: 1.34,
  kInterfaceCells: 1.0,
  kInterfaceSpeed: 5.0e-4,
  kMobilityFloor: 0.02,
  kCourant: 0.25,
  kDiffusionSafety: 0.8,
  kCapillary: 5.0,
  kMaxSubsteps: 32,
  kCyclesPerStep: 1,
  kCoarsestSweeps: 40,
  kCoarsestOmega: 1.6,
  kSpeedMargin: 1.5,
  kCoarsestMax: 8,

  liquidDensity0(salt) {
    return this.kWaterDensity * (1.0 + this.kSaltDensityPerPercent * salt);
  },
  crossoverSlope(salt) {
    return this.kWaxDensity * this.kWaxExpansion - this.liquidDensity0(salt) * this.kLiquidExpansion;
  },
  crossover(salt) {
    return this.kReferenceT + (this.kWaxDensity - this.liquidDensity0(salt)) / this.crossoverSlope(salt);
  },
  faceLossRate(gap) {
    return (2.0 * this.kFaceLoss) / (this.kHeatCapacity * gap);
  },
  faceConductance(lamp) {
    return 2.0 * this.kFaceLoss * lamp.width * lamp.height;
  },
  capConductance(lamp) {
    return this.kCapLoss * lamp.width * lamp.gap;
  },
  interfaceWidth(grid) {
    return this.kInterfaceCells * Math.max(grid.dx, grid.dy);
  },
  interfaceMobility(grid) {
    return Math.max(grid.dx, grid.dy) * this.kInterfaceSpeed;
  },
  chemicalScale(grid, sigma) {
    return (3.0 * sigma) / this.interfaceWidth(grid);
  },
  diffusionLimit(grid, lamp) {
    const laplacian = 4.0 / (grid.dx * grid.dx) + 4.0 / (grid.dy * grid.dy);
    const kappa = (Math.max(this.kLiquidConductivity, this.kWaxConductivity) + lamp.coil) / this.kHeatCapacity;
    return this.kDiffusionSafety / (kappa * laplacian);
  },
  interfaceLimit(grid) {
    const laplacian = 4.0 / (grid.dx * grid.dx) + 4.0 / (grid.dy * grid.dy);
    const xi = this.interfaceWidth(grid);
    const D = this.interfaceMobility(grid);
    return (this.kDiffusionSafety * 2.0) / (D * laplacian * (2.0 * xi * xi * laplacian + 2.0));
  },
  capillaryLimit(grid, lamp) {
    if (lamp.tension <= 0.0) return 1e30;
    const h = Math.min(grid.dx, grid.dy);
    const kMax = (lamp.gap * lamp.gap) / (12.0 * Math.min(this.kLiquidViscosity, lamp.waxViscosity));
    return (this.kCapillary * h * h * h) / (lamp.tension * kMax);
  },
  bulbStep(current, target, dt, tau) {
    if (tau <= 0.0) return target;
    return current + (target - current) * (1.0 - Math.exp(-dt / tau));
  },
};

/// std::lround: halves away from zero. Every argument here is positive.
const lround = (x) => (x >= 0 ? Math.floor(x + 0.5) : -Math.floor(-x + 0.5));

/// Physics.cpp ChooseGrid.
function chooseGrid(width, height, cellsUp) {
  const grid = { nx: 0, ny: Math.max(cellsUp, 8), levels: 0, dx: 0, dy: 0 };
  const across = (grid.ny * width) / height;
  let halvings = 0;
  grid.nx = Math.max(2, lround(across));
  for (let h = 1; grid.ny % (1 << h) === 0; ++h) {
    const unit = 1 << h;
    const nx = Math.max(2, lround(across / unit)) * unit;
    halvings = h;
    grid.nx = nx;
    if (nx / unit + 1 <= physics.kCoarsestMax && grid.ny / unit + 1 <= physics.kCoarsestMax) break;
  }
  grid.levels = halvings + 1;
  grid.dx = width / grid.nx;
  grid.dy = height / grid.ny;
  return grid;
}

const nodesX = (grid, level) => (grid.nx >> level) + 1;
const nodesY = (grid, level) => (grid.ny >> level) + 1;

/// Bassalt.cpp: host seconds one frame may advance by, the dye's relaxation,
/// the inflation's pin, and the smoother's sweeps.
const kMaxFrameDelta = 0.25;
const kDyeRelax = f32(1.0 / 30.0);
const kInflatePenalty = 4000.0;
const kPreSweeps = 1;
const kPostSweeps = 1;

//---------------------------------------------------------------------------
// The renderer: BassaltPlugin, less the host.
//---------------------------------------------------------------------------

/// Read by the line under the canvas.
const telemetry = { grid: null, substeps: 0, shortfall: 0, simTime: 0, bulbPower: 0, crossover: 0, speed: 1 };

function createRenderer(gl, quad) {
  // Linear filtering of the RGBA32F state and the R32F psi and lens thickness
  // is what the composite does. Without this extension those textures are
  // incomplete in WebGL2 and every fetch — texelFetch included — reads zero.
  if (!gl.getExtension('OES_texture_float_linear')) {
    throw new GLError(
      'OES_texture_float_linear is missing. Bassalt keeps its lamp in 32-bit float textures and the composite filters them linearly, as the plugin does; without the extension those textures read as zero, and the page would show an empty lamp rather than say why.',
    );
  }

  const names = ['props', 'coef', 'circulation', 'coarsen', 'smooth', 'restrict', 'coarsest', 'prolong',
    'update', 'interface', 'reduce', 'event', 'inflate', 'composite'];
  const sources = [PROPS, COEF, CIRCULATION, COARSEN, SMOOTH, RESTRICT, COARSEST, PROLONG,
    UPDATE, INTERFACE, REDUCE, EVENT, INFLATE, COMPOSITE];
  const P = {};
  names.forEach((name, i) => { P[name] = new Program(gl, VERTEX, sources[i], name); });

  // Program.set() with two numbers is glUniform2f; the ivec2 uniforms need
  // glUniform2i, which the plugin's Program::SetInts is.
  const setInts = (p, name, a, b) => {
    const loc = p.location(name);
    if (loc === null) { p.missing.add(name); return; }
    gl.uniform2i(loc, a, b);
  };

  //--- the buffers (Bassalt.h) --------------------------------------------
  const state = [new PassBuffer(gl, { filter: 'linear' }), new PassBuffer(gl, { filter: 'linear' })];
  const props = new PassBuffer(gl, { filter: 'nearest' });
  const reduce = [new PassBuffer(gl, { filter: 'nearest' }), new PassBuffer(gl, { filter: 'nearest' })];

  function makeMultigrid() {
    return {
      levels: [],
      ensure(grid) {
        if (this.levels.length !== grid.levels) {
          this.destroy();
          for (let l = 0; l < grid.levels; ++l) {
            this.levels.push({
              coef: new PassBuffer(gl, { filter: 'nearest' }),
              rhs: new PassBuffer(gl, { filter: 'nearest' }),
              // Linear: the composite reads psi and the thickness between nodes.
              psi: [new PassBuffer(gl, { filter: 'linear' }), new PassBuffer(gl, { filter: 'linear' })],
              current: 0,
              nodesX: 0,
              nodesY: 0,
            });
          }
        }
        for (let l = 0; l < grid.levels; ++l) {
          const L = this.levels[l];
          L.nodesX = nodesX(grid, l);
          L.nodesY = nodesY(grid, l);
          L.coef.ensure(L.nodesX, L.nodesY, gl.RGBA32F);
          L.rhs.ensure(L.nodesX, L.nodesY, gl.R32F);
          L.psi[0].ensure(L.nodesX, L.nodesY, gl.R32F);
          L.psi[1].ensure(L.nodesX, L.nodesY, gl.R32F);
        }
      },
      destroy() {
        for (const L of this.levels) {
          L.coef.dispose();
          L.rhs.dispose();
          L.psi[0].dispose();
          L.psi[1].dispose();
        }
        this.levels = [];
      },
      clearSolution() {
        for (const L of this.levels) {
          L.psi[0].clearTo(0, 0, 0, 0);
          L.psi[1].clearTo(0, 0, 0, 0);
        }
      },
      solution() {
        return this.levels.length ? this.levels[0].psi[this.levels[0].current].texture : null;
      },
    };
  }
  const flow = makeMultigrid();
  const inflate = makeMultigrid();

  let stateIndex = 0;
  let haveThickness = false;
  let grid = { nx: 0, ny: 0, levels: 0, dx: 0, dy: 0 };
  let lamp = null;

  let speedBuffer = null;
  let speedFence = null;
  let speedEstimate = -1.0;
  const pixel = new Float32Array(4);

  let lastNow = -1.0;
  let simTime = 0.0;
  let bulbPower = 0.0;
  let pendingKick = 0.0; // never filled: no audio, so no onset (see the header)
  let fresh = true;
  let lastSubsteps = 0;
  let lastShortfall = 0;

  //--- ensureBuffers --------------------------------------------------------
  function ensureBuffers(wanted) {
    // A different grid is a different lamp: it starts again, warm.
    if (wanted.nx !== grid.nx || wanted.ny !== grid.ny) {
      state[0].dispose();
      state[1].dispose();
      props.dispose();
      reduce[0].dispose();
      reduce[1].dispose();
      flow.destroy();
      inflate.destroy();
      stateIndex = 0;
      fresh = true;
      speedEstimate = -1.0;
    }
    for (const buffer of state) buffer.ensure(wanted.nx, wanted.ny, gl.RGBA32F);
    props.ensure(wanted.nx, wanted.ny, gl.RGBA32F);
    reduce[0].ensure(Math.floor((wanted.nx + 1 + 15) / 16), Math.floor((wanted.ny + 1 + 15) / 16), gl.R32F);
    reduce[1].ensure(1, 1, gl.R32F);
    flow.ensure(wanted);
    inflate.ensure(wanted);
    grid = { ...wanted };
  }

  //--- the pass plumbing ----------------------------------------------------
  function begin(target, program) {
    target.bind();
    program.use();
  }
  function bindTexture(unit, texture) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
  }
  const draw = () => quad.draw();

  function propsPass() {
    const p = P.props;
    begin(props, p);
    bindTexture(0, state[stateIndex].texture);
    p.setInt('State', 0);
    setInts(p, 'Cells', grid.nx, grid.ny);
    p.set('Spacing', grid.dx, grid.dy);
    const xi = physics.interfaceWidth(grid);
    p.set('Xi2', 2.0 * xi * xi);
    p.set('Slope', physics.crossoverSlope(lamp.salt));
    p.set('Crossover', physics.crossover(lamp.salt));
    p.set('LiquidBeta', physics.liquidDensity0(lamp.salt) * physics.kLiquidExpansion);
    p.set('ReferenceT', physics.kReferenceT);
    p.set('MeltingPoint', lamp.meltingPoint);
    p.set('FreezeWidth', physics.kFreezeWidth);
    p.set('SolidLog', Math.log(physics.kSolidMultiple));
    p.set('WaxMu', lamp.waxViscosity);
    p.set('LiquidMu', physics.kLiquidViscosity);
    p.set('Resist', 12.0 / (lamp.gap * lamp.gap));
    p.set('WaxK', physics.kWaxConductivity);
    p.set('LiquidK', physics.kLiquidConductivity);
    p.set('CoilK', lamp.coil);
    p.set('CoilBox', 0.5 * physics.kCoilWidth, physics.kCoilHeight);
    draw();
  }

  function coarsen(system) {
    const p = P.coarsen;
    for (let l = 1; l < system.levels.length; ++l) {
      begin(system.levels[l].coef, p);
      bindTexture(0, system.levels[l - 1].coef.texture);
      p.setInt('Coef', 0);
      setInts(p, 'FineNodes', system.levels[l - 1].nodesX, system.levels[l - 1].nodesY);
      draw();
    }
  }

  function smooth(system, level, zeroGuess) {
    const L = system.levels[level];
    const p = P.smooth;
    begin(L.psi[1 - L.current], p);
    bindTexture(0, L.psi[L.current].texture);
    bindTexture(1, L.coef.texture);
    bindTexture(2, L.rhs.texture);
    p.setInt('Psi', 0);
    p.setInt('Coef', 1);
    p.setInt('Rhs', 2);
    setInts(p, 'Nodes', L.nodesX, L.nodesY);
    p.setInt('ZeroGuess', zeroGuess ? 1 : 0);
    draw();
    L.current = 1 - L.current;
  }

  function cycle(system, level) {
    const L = system.levels[level];

    if (level === system.levels.length - 1) {
      const p = P.coarsest;
      begin(L.psi[1 - L.current], p);
      bindTexture(0, L.coef.texture);
      bindTexture(1, L.rhs.texture);
      p.setInt('Coef', 0);
      p.setInt('Rhs', 1);
      setInts(p, 'Nodes', L.nodesX, L.nodesY);
      p.setInt('Sweeps', physics.kCoarsestSweeps);
      p.set('Omega', physics.kCoarsestOmega);
      draw();
      L.current = 1 - L.current;
      return;
    }

    // A coarse level solves for a correction, so it starts from zero.
    for (let sweep = 0; sweep < kPreSweeps; ++sweep) smooth(system, level, level > 0 && sweep === 0);

    const C = system.levels[level + 1];
    {
      const p = P.restrict;
      begin(C.rhs, p);
      bindTexture(0, L.psi[L.current].texture);
      bindTexture(1, L.coef.texture);
      bindTexture(2, L.rhs.texture);
      p.setInt('Psi', 0);
      p.setInt('Coef', 1);
      p.setInt('Rhs', 2);
      setInts(p, 'FineNodes', L.nodesX, L.nodesY);
      setInts(p, 'CoarseNodes', C.nodesX, C.nodesY);
      draw();
    }

    cycle(system, level + 1);

    // The correction, and the first post-smoothing sweep with it.
    {
      const p = P.prolong;
      begin(L.psi[1 - L.current], p);
      bindTexture(0, L.psi[L.current].texture);
      bindTexture(1, C.psi[C.current].texture);
      bindTexture(2, L.coef.texture);
      bindTexture(3, L.rhs.texture);
      p.setInt('Psi', 0);
      p.setInt('Correction', 1);
      p.setInt('Coef', 2);
      p.setInt('Rhs', 3);
      setInts(p, 'Nodes', L.nodesX, L.nodesY);
      p.set('Weight', 1.0);
      draw();
      L.current = 1 - L.current;
    }

    for (let sweep = 1; sweep < kPostSweeps; ++sweep) smooth(system, level, false);
  }

  function flowSolve(cycles, fromZero) {
    const fine = flow.levels[0];
    {
      const p = P.coef;
      begin(fine.coef, p);
      bindTexture(0, props.texture);
      p.setInt('Props', 0);
      setInts(p, 'Cells', grid.nx, grid.ny);
      p.set('Spacing', grid.dx, grid.dy);
      draw();
    }
    {
      const p = P.circulation;
      begin(fine.rhs, p);
      bindTexture(0, state[stateIndex].texture);
      bindTexture(1, props.texture);
      p.setInt('State', 0);
      p.setInt('Props', 1);
      setInts(p, 'Cells', grid.nx, grid.ny);
      setInts(p, 'Nodes', fine.nodesX, fine.nodesY);
      p.set('Spacing', grid.dx, grid.dy);
      p.set('Gravity', physics.kGravity);
      p.set('Beta', physics.chemicalScale(grid, lamp.tension));
      draw();
    }
    coarsen(flow);

    if (fromZero) flow.clearSolution();
    for (let c = 0; c < cycles; ++c) cycle(flow, 0);
  }

  function reduceSpeed() {
    const fine = flow.levels[0];
    {
      const p = P.reduce;
      begin(reduce[0], p);
      bindTexture(0, fine.psi[fine.current].texture);
      p.setInt('Source', 0);
      setInts(p, 'SourceSize', fine.nodesX, fine.nodesY);
      p.set('Spacing', grid.dx, grid.dy);
      p.setInt('Mode', 0);
      p.setInt('Block', 16);
      draw();
    }
    {
      const p = P.reduce;
      begin(reduce[1], p);
      bindTexture(0, reduce[0].texture);
      p.setInt('Source', 0);
      setInts(p, 'SourceSize', reduce[0].width, reduce[0].height);
      p.setInt('Mode', 1);
      p.setInt('Block', Math.max(reduce[0].width, reduce[0].height));
      draw();
    }
  }

  // An R32F target is read as RGBA/FLOAT: the one float read-back format WebGL2
  // guarantees with EXT_color_buffer_float. Only the red channel is used.
  function fastestFaceNow() {
    reduceSpeed();
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, pixel);
    const speed = pixel[0];
    return Number.isFinite(speed) ? speed : 0.0;
  }

  // The plugin requests the read, fences it, and next frame WAITS on the fence
  // (glClientWaitSync, up to a second) so that every run takes the same steps.
  // WebGL2 allows no client wait but zero, and a read of a fenced buffer
  // before its fence has signalled stalls the page. So here the read is taken
  // only once the fence HAS signalled, and no new request is made while one
  // is in flight:
  // when the GPU is more than a frame behind, a step is sized from the last
  // speed that did arrive. The update's own rescue (scaling psi when the flow
  // outruns the step, from this substep's reduced speed) is untouched by this.
  function requestSpeed() {
    if (speedFence !== null) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, reduce[1].fbo);
    if (speedBuffer === null) speedBuffer = gl.createBuffer();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, speedBuffer);
    // Fresh storage for every request. Re-using it draws Chrome's "READ-usage
    // buffer was written, then fenced, but written again before being read
    // back" on every frame even when it WAS read back in between, until the
    // context stops reporting WebGL errors at all. Measured 2026-09-24.
    gl.bufferData(gl.PIXEL_PACK_BUFFER, 16, gl.STREAM_READ);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    speedFence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
  }

  function collectSpeed() {
    if (speedFence === null) return;
    if (gl.getSyncParameter(speedFence, gl.SYNC_STATUS) !== gl.SIGNALED) return;
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, speedBuffer);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, pixel);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.deleteSync(speedFence);
    speedFence = null;
    const speed = pixel[0];
    speedEstimate = Number.isFinite(speed) ? speed : -1.0;
  }

  function updatePass(dt, power, clipPower, clipTexture) {
    const target = 1 - stateIndex;
    const p = P.update;
    begin(state[target], p);
    bindTexture(0, state[stateIndex].texture);
    bindTexture(1, props.texture);
    bindTexture(2, flow.solution());
    bindTexture(3, clipTexture);
    bindTexture(4, reduce[1].texture);
    p.setInt('State', 0);
    p.setInt('Props', 1);
    p.setInt('Psi', 2);
    p.setInt('Clip', 3);
    p.setInt('Fastest', 4);
    setInts(p, 'Cells', grid.nx, grid.ny);
    p.set('Spacing', grid.dx, grid.dy);
    p.set('Dt', dt);
    p.set('Capacity', physics.kHeatCapacity);
    p.set('Ambient', lamp.ambient);
    p.set('FaceRate', physics.faceLossRate(lamp.gap));
    p.set('CapRate', physics.kCapLoss / (physics.kHeatCapacity * grid.dy));
    p.set('BulbPower', power);
    const sigma = physics.kBulbSpread * lamp.width;
    let sum = 0.0;
    for (let i = 0; i < grid.nx; ++i) {
      const x = (i + 0.5) * grid.dx - 0.5 * grid.nx * grid.dx;
      sum += Math.exp(-0.5 * (x / sigma) * (x / sigma));
    }
    p.set('BulbSigma', sigma);
    p.set('BulbNorm', 1.0 / sum);
    p.set('Gap', lamp.gap);
    p.set('ClipPower', clipPower);
    p.set('MaxUV', 1.0, 1.0);
    p.set('UVRelax', kDyeRelax);
    p.set('SpeedCap', (physics.kCourant * Math.min(grid.dx, grid.dy)) / dt);
    draw();
    stateIndex = target;
  }

  function interfacePass(dt) {
    const target = 1 - stateIndex;
    const p = P.interface;
    begin(state[target], p);
    bindTexture(0, state[stateIndex].texture);
    bindTexture(1, props.texture);
    p.setInt('State', 0);
    p.setInt('Props', 1);
    setInts(p, 'Cells', grid.nx, grid.ny);
    p.set('Spacing', grid.dx, grid.dy);
    p.set('Dt', dt);
    p.set('Mobility', physics.interfaceMobility(grid));
    p.set('MobilityFloor', physics.kMobilityFloor);
    draw();
    stateIndex = target;
  }

  function eventPass(mode, temperature, clipTexture, params) {
    const target = 1 - stateIndex;
    const p = P.event;
    begin(state[target], p);
    bindTexture(0, state[stateIndex].texture);
    bindTexture(1, clipTexture);
    p.setInt('State', 0);
    p.setInt('Clip', 1);
    p.setInt('Mode', mode);
    setInts(p, 'Cells', grid.nx, grid.ny);
    p.set('Spacing', grid.dx, grid.dy);
    p.set('Xi', physics.interfaceWidth(grid));
    p.set('WaxHeight', controls.waxAmount(params.get('waxAmount')) * lamp.height);
    p.set('Temperature', temperature);
    p.set('Threshold', clamp(params.get('pourThreshold'), 0, 1));
    p.set('MaxUV', 1.0, 1.0);
    draw();
    stateIndex = target;
  }

  function inflatePass() {
    const fine = inflate.levels[0];
    const p = P.inflate;
    for (let mode = 0; mode < 2; ++mode) {
      begin(mode === 0 ? fine.coef : fine.rhs, p);
      bindTexture(0, state[stateIndex].texture);
      p.setInt('State', 0);
      setInts(p, 'Cells', grid.nx, grid.ny);
      p.set('Spacing', grid.dx, grid.dy);
      p.setInt('Mode', mode);
      p.set('Penalty', kInflatePenalty);
      draw();
    }
    coarsen(inflate);
    // Warm-started from last frame's lenses, which moved by a fraction of a cell.
    cycle(inflate, 0);
    haveThickness = true;
  }

  // Bass * analyser.Bass() is the other term in the plugin, and with no audio
  // the analyser's bass is 0: see the header.
  const bulbTarget = (params) => Math.max(0.0, controls.bulb(params.get('bulb')));

  function simulate(dtFrame, clipTexture, params) {
    const limit = Math.min(physics.diffusionLimit(grid, lamp), physics.capillaryLimit(grid, lamp));
    const interfaceLimit = physics.interfaceLimit(grid);
    const target = bulbTarget(params);
    const tau = controls.bulbLag(params.get('bulbLag'));
    const clipPower = controls.clipHeat(params.get('clipHeat'));
    const cell = Math.min(grid.dx, grid.dy);

    collectSpeed();

    let remaining = dtFrame;
    let substeps = 0;
    while (remaining > 1e-9 && substeps < physics.kMaxSubsteps) {
      propsPass();

      flowSolve(physics.kCyclesPerStep, false);
      // Last frame's fastest face with a margin, or -- after an event, or on
      // the first frame -- this solve's own, read now.
      if (speedEstimate < 0.0) speedEstimate = fastestFaceNow();
      else reduceSpeed();
      const speed = physics.kSpeedMargin * speedEstimate;

      let dt = Math.min(remaining, limit);
      if (speed > 0.0) dt = Math.min(dt, (physics.kCourant * cell) / speed);

      bulbPower = physics.bulbStep(bulbPower, target, dt, tau);
      const power = bulbPower + pendingKick / dt;
      pendingKick = 0.0;

      updatePass(dt, power, clipPower, clipTexture);

      // Cahn-Hilliard, in as many subcycles as its explicit limit needs.
      const subcycles = Math.max(1, Math.ceil(dt / interfaceLimit));
      for (let k = 0; k < subcycles; ++k) {
        propsPass();
        interfacePass(dt / subcycles);
      }

      remaining -= dt;
      simTime += dt;
      ++substeps;
    }
    lastSubsteps = substeps;
    lastShortfall = remaining;

    // The last solve's fastest face, already reduced, for the next frame.
    if (substeps > 0) requestSpeed();
  }

  return {
    render({ input, params, width, height, time }) {
      gl.disable(gl.BLEND);

      //--- time: frame-relative, clamped as the plugin clamps the host's ---
      const now = time;
      const hostDt = lastNow >= 0.0 ? clamp(now - lastNow, 0.0, kMaxFrameDelta) : 0.0;
      lastNow = now;

      //--- the events: a press, taken and released ------------------------
      const take = (id) => {
        if (params.get(id) > 0.5) {
          params.set(id, 0);
          return true;
        }
        return false;
      };
      const warmWanted = take('warm');
      const resetWanted = take('reset');
      const pourWanted = take('pour');

      //--- the lamp, in metres --------------------------------------------
      lamp = {
        height: controls.lampHeight(params.get('lampHeight')),
        width: 0,
        gap: controls.gap(params.get('gap')),
        ambient: controls.ambient(params.get('ambient')),
        salt: controls.salt(params.get('salt')),
        tension: controls.tension(params.get('tension')),
        waxViscosity: controls.waxViscosity(params.get('waxViscosity')),
        meltingPoint: controls.meltingPoint(params.get('meltingPoint')),
        coil: controls.coil(params.get('coil')),
      };
      lamp.width = (lamp.height * width) / height;

      const detail = clamp(Math.round(params.get('detail')), 0, kDetailCells.length - 1);
      const wanted = chooseGrid(lamp.width, lamp.height, kDetailCells[detail]);
      ensureBuffers(wanted);
      grid.dx = wanted.dx;
      grid.dy = wanted.dy;

      const clipTexture = input.texture;
      const warmT = lamp.ambient
        + bulbTarget(params) / (physics.faceConductance(lamp) + physics.capConductance(lamp));

      if (fresh || resetWanted) {
        eventPass(0, lamp.ambient, clipTexture, params);
        flow.clearSolution();
        inflate.clearSolution();
        bulbPower = 0.0;
        if (fresh) {
          // A lamp dropped on a layer starts warm.
          eventPass(1, warmT, clipTexture, params);
          bulbPower = bulbTarget(params);
        }
        fresh = false;
        speedEstimate = -1.0;
      }
      if (warmWanted) {
        eventPass(1, warmT, clipTexture, params);
        bulbPower = bulbTarget(params);
        speedEstimate = -1.0;
      }
      if (pourWanted) {
        eventPass(2, 0.0, clipTexture, params);
        speedEstimate = -1.0;
      }

      const speedFactor = controls.speed(params.get('speed'));
      const dt = hostDt * speedFactor;
      if (dt > 0.0) simulate(dt, clipTexture, params);
      else lastSubsteps = 0;

      const view = clamp(Math.round(params.get('view')), 0, 4);
      const clipMode = clamp(Math.round(params.get('clip')), 0, 1);
      const glass = clamp(Math.round(params.get('glass')), 0, 1);
      haveThickness = false;
      if (view === 0 && clipMode === 0) inflatePass();

      //--- composite, into the page's framebuffer -------------------------
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, width, height);
      {
        const p = P.composite.use();
        bindTexture(0, clipTexture);
        bindTexture(1, state[stateIndex].texture);
        bindTexture(2, inflate.solution());
        bindTexture(3, flow.solution());
        p.setInt('InputTexture', 0);
        p.setInt('StateTexture', 1);
        p.setInt('ThickTexture', 2);
        p.setInt('PsiTexture', 3);
        p.set('MaxUV', 1.0, 1.0);
        setInts(p, 'Cells', grid.nx, grid.ny);
        p.set('Spacing', grid.dx, grid.dy);
        p.set('LampSize', grid.nx * grid.dx, grid.ny * grid.dy);
        p.set('Gap', lamp.gap);
        p.setInt('ClipMode', clipMode);
        p.setInt('GlassMode', glass);
        p.setInt('ViewMode', view);
        p.set('MixAmount', clamp(params.get('mix'), 0, 1));
        p.set('WaxColour', params.get('waxR'), params.get('waxG'), params.get('waxB'));
        p.set('Tint', params.get('tintR'), params.get('tintG'), params.get('tintB'));
        p.set('Glow', controls.glow(params.get('glow')));
        p.set('BulbFraction', bulbPower / 60.0);
        p.set('Distance', controls.refraction(params.get('refraction')));
        p.set('WaxIndex', physics.kWaxIndex);
        p.set('WaterIndex', physics.kWaterIndex);
        p.set('Slope', physics.crossoverSlope(lamp.salt));
        p.set('Crossover', physics.crossover(lamp.salt));
        p.set('LiquidBeta', physics.liquidDensity0(lamp.salt) * physics.kLiquidExpansion);
        p.set('ReferenceT', physics.kReferenceT);
        p.setInt('UseThickness', haveThickness ? 1 : 0);
        draw();
      }

      // Everything this frame bound, unbound.
      for (let unit = 4; unit >= 0; --unit) bindTexture(unit, null);
      gl.useProgram(null);

      telemetry.grid = grid;
      telemetry.substeps = lastSubsteps;
      telemetry.shortfall = lastShortfall;
      telemetry.simTime = simTime;
      telemetry.bulbPower = bulbPower;
      telemetry.crossover = physics.crossover(lamp.salt);
      telemetry.speed = speedFactor;
    },
  };
}

//---------------------------------------------------------------------------
// The controls, read out of the plugin's constructor. Same names, same groups,
// same order, same defaults, same dropdown elements.
//
// Absent: the Audio group (the 64-bin FFT buffer, Bass, Bass Band and Kick),
// for the reason at the top of this file, and the About block.
//---------------------------------------------------------------------------

const fixed = (n, d) => n.toFixed(d);

mountDemo({
  name: 'Bassalt',
  pluginId: 'BS01',
  tagline:
    'A lava lamp, as a heat engine. A bulb heats wax that is a little denser than the salted water when cold and expands three times as fast, so above a crossover temperature it rises, cools at the cap and sinks. Cahn-Hilliard wax, advected heat and Hele-Shaw flow solved for its streamfunction on a GPU multigrid — nothing is animated. The clip is the world behind the lamp, bent by the wax, or the wax itself.',
  repo: 'https://github.com/stoatworks-labs/bassalt',

  params: [
    { id: 'lampHeight', name: 'Lamp Height', type: 'standard', default: 0.477, group: 'Lamp',
      display: (v) => `${fixed(controls.lampHeight(v) * 100, 1)} cm`,
      hint: 'The frame’s height in metres, 0.1 to 1 m. The lamp is simulated on a grid in metres, so a bigger lamp at the same Detail has coarser cells.' },
    { id: 'gap', name: 'Gap', type: 'standard', default: 0.463, group: 'Lamp',
      display: (v) => `${fixed(controls.gap(v) * 1000, 2)} mm`,
      hint: 'The Hele-Shaw gap, front glass to back, 1 to 20 mm. The flow goes as its square.' },
    { id: 'detail', name: 'Detail', type: 'option', default: 2, group: 'Lamp',
      elements: ['64', '96', '128', '192', '256'],
      hint: 'Grid cells up the lamp. A new Detail is a new lamp: it starts again, warm. 256 is not real time at the default surface tension in the plugin either.' },
    { id: 'speed', name: 'Speed', type: 'standard', default: controls.paramFromSpeed(3.0), group: 'Lamp',
      display: (v) => `${fixed(controls.speed(v), 1)}×`,
      hint: 'Lamp seconds per second, 1× to 300×. A real lamp takes an hour to warm. At most 32 substeps a frame: past that the lamp runs slower than asked, and the line under the picture says so.' },
    { id: 'ambient', name: 'Ambient', type: 'standard', default: 0.4, group: 'Lamp',
      display: (v) => `${fixed(controls.ambient(v), 1)} °C`,
      hint: 'The room, 10 to 40 °C.' },
    { id: 'warm', name: 'Warm', type: 'boolean', default: 0, group: 'Lamp',
      hint: 'FF_TYPE_EVENT in the plugin; a toggle the page releases itself here. Sets every cell to the lamp’s lumped steady temperature and the bulb to its target, keeping the wax where it is.' },
    { id: 'reset', name: 'Reset', type: 'boolean', default: 0, group: 'Lamp',
      hint: 'FF_TYPE_EVENT in the plugin; a toggle here. The cold slab: Wax Amount of wax at the bottom, everything at Ambient, bulb off and warming.' },

    { id: 'bulb', name: 'Bulb', type: 'standard', default: 0.5, group: 'Bulb',
      display: (v) => `${fixed(controls.bulb(v), 1)} W`,
      hint: 'The bulb’s power, 0 to 60 W. In the plugin Bass adds to this from the routed audio; there is no audio here, so this is the whole of it.' },
    { id: 'bulbLag', name: 'Bulb Lag', type: 'standard', default: 0.55, group: 'Bulb',
      display: (v) => `${fixed(controls.bulbLag(v), 1)} s`,
      hint: 'The bulb’s thermal time constant in lamp seconds, 0.5 to 120.' },
    { id: 'coil', name: 'Coil', type: 'standard', default: 0.5, group: 'Bulb',
      display: (v) => `${fixed(controls.coil(v), 2)} W/m K`,
      hint: 'Conductivity the coil adds in the bottom 3% of the lamp’s middle half.' },

    { id: 'salt', name: 'Salt', type: 'standard', default: 0.15, group: 'Fluids',
      display: (v) => `${fixed(controls.salt(v), 2)}%`,
      hint: 'Salt in the water, 0 to 4% by mass. It sets the crossover temperature T* above which wax is lighter than the water: 63 °C with none, 11.3 K lower per 1%. The line under the picture shows T* as it stands.' },
    { id: 'waxAmount', name: 'Wax Amount', type: 'standard', default: 0.286, group: 'Fluids',
      display: (v) => `${fixed(controls.waxAmount(v) * 100, 1)}%`,
      hint: 'Wax as a fraction of the lamp, 5 to 40%. Applied on Reset.' },
    { id: 'tension', name: 'Surface Tension', type: 'standard', default: 0.447, group: 'Fluids',
      display: (v) => `${fixed(controls.tension(v) * 1000, 2)} mN/m`,
      hint: 'Wax-water interfacial tension, 0 to 10 mN/m. The flow’s capillary step limit goes as 1/σ, which is why it stops at 10.' },
    { id: 'waxViscosity', name: 'Wax Viscosity', type: 'standard', default: 0.434, group: 'Fluids',
      display: (v) => `${fixed(controls.waxViscosity(v) * 1000, 1)} mPa s`,
      hint: 'Melted wax, 1 mPa s to 1 Pa s. Below the melting point it rises e-fold every 2.5 K.' },
    { id: 'meltingPoint', name: 'Melting Point', type: 'standard', default: 0.43, group: 'Fluids',
      display: (v) => `${fixed(controls.meltingPoint(v), 1)} °C`,
      hint: 'Wax melting point, 35 to 70 °C.' },

    { id: 'clip', name: 'Clip', type: 'option', default: 0, group: 'Clip',
      elements: ['Behind', 'Dyed'],
      hint: 'Behind: the clip is the world behind the lamp, seen through the glass and bent by each blob as a lens. Dyed: the clip is the wax, carried by the flow.' },
    { id: 'pour', name: 'Pour', type: 'boolean', default: 0, group: 'Clip',
      hint: 'FF_TYPE_EVENT in the plugin; a toggle here. The clip’s bright parts become wax at the water’s temperature where they land.' },
    { id: 'pourThreshold', name: 'Pour Threshold', type: 'standard', default: 0.5, group: 'Clip',
      hint: 'The luma above which Pour makes wax.' },
    { id: 'clipHeat', name: 'Clip Heat', type: 'standard', default: 0, group: 'Clip',
      display: (v) => `${fixed(controls.clipHeat(v), 1)} W`,
      hint: 'Watts the clip adds at all white, spread as its luma.' },

    { id: 'waxR', name: 'Wax Colour', type: 'colour', default: 1.0, group: 'Look' },
    { id: 'waxG', name: 'Wax Colour_Green', type: 'colour', default: 0.3, group: 'Look' },
    { id: 'waxB', name: 'Wax Colour_Blue', type: 'colour', default: 0.12, group: 'Look' },
    { id: 'tintR', name: 'Liquid Tint', type: 'colour', default: 0.85, group: 'Look' },
    { id: 'tintG', name: 'Tint_Green', type: 'colour', default: 0.55, group: 'Look' },
    { id: 'tintB', name: 'Tint_Blue', type: 'colour', default: 0.75, group: 'Look' },
    { id: 'glow', name: 'Glow', type: 'standard', default: 0.5, group: 'Look',
      display: (v) => `${fixed(controls.glow(v), 2)}`,
      hint: 'The bulb’s light in the lamp, 0 to 2.' },
    { id: 'glass', name: 'Glass', type: 'option', default: 0, group: 'Look',
      elements: ['Flat', 'Cylinder'],
      hint: 'Flat: the frame is a flat-panel lamp. Cylinder: a vertical glass cylinder seen side on, refracting the world behind it.' },
    { id: 'refraction', name: 'Refraction', type: 'standard', default: 0.3, group: 'Look',
      display: (v) => `${fixed(controls.refraction(v) * 100, 0)} cm`,
      hint: 'Distance from the lamp’s back to the world behind it, 0 to 1 m. At 0 nothing is displaced.' },

    { id: 'view', name: 'View', type: 'option', default: 0, group: 'Output',
      elements: ['Lamp', 'Temperature', 'Wax', 'Velocity', 'Density'],
      hint: 'Lamp is the effect; the others show one field of the model on its own.' },
    { id: 'mix', name: 'Mix', type: 'standard', default: 1.0, group: 'Output',
      display: (v) => `${fixed(v * 100, 0)}%`,
      hint: 'Against the clip as it came in.' },
  ],

  // The geometry card first: straight lines show each blob's lens and the
  // cylinder's squeeze, the way the plugin's own hero card does. Lights on
  // black is the one to Pour from — only the bright spots become wax.
  sources: ['grid', 'scene', 'spot', 'bars', 'ramp', 'detail'],

  needFloat: true,

  blurb:
    "It is Bassalt's own GLSL — all fourteen passes: the wax, the heat, the streamfunction's multigrid, the lens and the composite — ported from the repository to WebGL2 and running on generated clips in this page, with the plugin's pass orchestration ported to JavaScript alongside. There is no audio here, so the bulb that Bass drives in the plugin runs on its Bulb control alone.",

  differences: [
    'No audio. In the plugin, Bass adds watts to the bulb from the routed audio and Kick drops heat into the base on each onset, through a 64-bin spectrum Resolume hands it as an FFT parameter. A browser has no equivalent, and asking for your microphone to demonstrate a video effect is not a trade worth making — so the Audio group (the buffer, Bass, Bass Band and Kick) is absent from the panel rather than present and dead. The removal is exact: with silence the plugin’s analyser reports a bass of 0 and never fires, so its bulb is the Bulb control alone, which is what runs here. Nothing else drives the bulb.',
    '`precise` is gone. The plugin marks the circulation and every flux `precise`, because Apple’s GLSL compiler reassociated that arithmetic and a level lamp moved by 1e-14 until it was added. GLSL ES 3.00 has no `precise`, so the kit removes the word — and a browser’s compiler is then free to reassociate those sums. A level slab here need not stay exactly still, and wax need not be conserved to the bit, as the plugin’s `--still` and `--volume` checks require of the plugin.',
    'Only the shaders are the plugin’s own text, and `demo/tools/check_shaders.py` proves that. The orchestration — which of the fourteen passes runs when, the multigrid’s V-cycle, the substep and subcycle limits, the frame-late speed read-back, the events, the physical constants and every control’s conversion — is a hand port of `source/Bassalt.cpp`, `source/Physics.cpp` and `source/Controls.cpp` to JavaScript. Nothing checks that port but a reader.',
    'The buffers are the plugin’s formats — RGBA32F cells, R32F multigrid levels — which WebGL2 can render to only with EXT_color_buffer_float and filter only with OES_texture_float_linear. A browser without either gets an error rather than an 8-bit lamp.',
    'Warm, Reset and Pour are toggles, not events. FFGL has FF_TYPE_EVENT and the kit’s parameter model does not, so the page releases each one the moment it has taken the press, which is what a host does with an event and why they blink.',
    'The fastest-face read-back is a frame late through a pixel-pack buffer and a fence, as in the plugin — but the plugin waits on the fence, so every run of it takes the same steps, and WebGL2 allows no wait. Here the speed is collected once the fence has signalled, and a frame whose GPU is behind sizes its steps from the last speed that arrived, so the step sequence depends on your GPU’s timing. The plugin’s rescue for a flow that outruns its step is in the update shader and runs here unchanged.',
    'JavaScript has no float. The control conversions are 32-bit in the plugin and rounded to 32 bits here, so they agree to the last bit or so rather than exactly; the physics is double in both.',
    'The About block is absent: a text line and three buttons that open a browser.',
    'Nothing on this page is measured. The plugin’s claims are numbers — the GPU flow solve against an exact CPU solve to 1e-9, a blob’s Hele-Shaw speed to 0.1–0.5%, T* to 4e-6 K, wax conserved to its own rounding — each with a negative control. That is `tools/verify.sh` in the repository, not this page.',
  ],

  createRenderer,
});

//---------------------------------------------------------------------------
// The line under the picture. The lamp is slow on purpose — a real one takes an
// hour to warm — so without the lamp's own clock a visitor cannot tell a lamp
// that is warming from a page that is stuck. Skipped in embed mode.
//---------------------------------------------------------------------------
if (!new URLSearchParams(window.location.search).has('embed')) {
  const stage = document.querySelector('.stage');
  if (stage) {
    const line = document.createElement('p');
    line.className = 'stage__status';
    stage.append(line);

    const clock = (s) => {
      const m = Math.floor(s / 60);
      return m > 0 ? `${m} min ${fixed(s - 60 * m, 0)} s` : `${fixed(s, 1)} s`;
    };

    setInterval(() => {
      const t = telemetry;
      if (!t.grid) return;
      const slow = t.shortfall > 1e-9
        ? ` At 32 substeps a frame the lamp is running slower than ${fixed(t.speed, 1)}× asks.`
        : '';
      line.textContent =
        `Lamp time ${clock(t.simTime)} on a ${t.grid.nx} × ${t.grid.ny} grid, `
        + `${t.substeps} substep${t.substeps === 1 ? '' : 's'} a frame. `
        + `Bulb ${fixed(t.bulbPower, 1)} W; wax rises above T* = ${fixed(t.crossover, 1)} °C.${slow}`;
    }, 250);
  }
}
