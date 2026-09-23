#include "Bassalt.h"

#include "Diag.h"
#include "GLState.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

using namespace ffglex;

namespace bassalt
{
namespace
{
/// glGetString returns nullptr when there is no current context, and feeding
/// that to std::string is undefined behaviour.
std::string glStringOrUnknown( GLenum name )
{
	const GLubyte* value = glGetString( name );
	return value ? reinterpret_cast< const char* >( value ) : "unknown";
}

const char* const kDetailNames[] = { "64", "96", "128", "192", "256" };
const char* const kClipNames[]   = { "Behind", "Dyed" };
const char* const kGlassNames[]  = { "Flat", "Cylinder" };
const char* const kViewNames[]   = { "Lamp", "Temperature", "Wax", "Velocity", "Density" };

constexpr int kClockVotes = 4;

/// Host seconds one frame may advance by. The host's clock jumps when the
/// composition is scrubbed or the machine sleeps.
constexpr double kMaxFrameDelta = 0.25;

/// The dye relaxes back at this rate (1/s of lamp time) per unit of stretch
/// past 4:1.
constexpr float kDyeRelax = 1.0f / 30.0f;

/// The inflation's pin in the water, against a Laplacian whose diagonal is 4:
/// h falls a thousandfold per cell into the water.
constexpr float kInflatePenalty = 4000.0f;

/// Sweeps of the smoother before and after each coarse correction.
constexpr int kPreSweeps  = 1;
constexpr int kPostSweeps = 1;

double wallSeconds()
{
	using namespace std::chrono;
	static const steady_clock::time_point start = steady_clock::now();
	return duration_cast< duration< double > >( steady_clock::now() - start ).count();
}

int optionIndex( float value, int count )
{
	return std::clamp( static_cast< int >( std::lround( value ) ), 0, count - 1 );
}
} // namespace

static_assert( PT_COUNT - PT_ABOUT_TEXT == stoatworks::about::kParamCount,
               "the About run no longer matches StoatworksAbout.h -- "
               "add or remove a PT_ABOUT_BUTTON_n to match" );

//---------------------------------------------------------------------------
// Program
//---------------------------------------------------------------------------
GLint Program::Loc( const char* name )
{
	const auto found = cache.find( name );
	if( found != cache.end() )
		return found->second;
	const GLint location = glGetUniformLocation( Id(), name );
	cache.emplace( name, location );
	return location;
}
void Program::Set( const char* name, float v )
{
	glUniform1f( Loc( name ), v );
}
void Program::Set( const char* name, float a, float b )
{
	glUniform2f( Loc( name ), a, b );
}
void Program::Set( const char* name, float a, float b, float c )
{
	glUniform3f( Loc( name ), a, b, c );
}
void Program::SetInt( const char* name, int v )
{
	glUniform1i( Loc( name ), v );
}
void Program::SetInts( const char* name, int a, int b )
{
	glUniform2i( Loc( name ), a, b );
}
void Program::Release()
{
	shader.FreeGLResources();
	cache.clear();
}

//---------------------------------------------------------------------------
// Multigrid storage
//---------------------------------------------------------------------------
bool Multigrid::Ensure( const physics::Grid& grid )
{
	if( static_cast< int >( levels.size() ) != grid.levels )
	{
		Destroy();
		levels.resize( static_cast< size_t >( grid.levels ) );
	}

	bool ok = true;
	for( int l = 0; l < grid.levels; ++l )
	{
		Level& level = levels[ static_cast< size_t >( l ) ];
		level.nodesX = grid.NodesX( l );
		level.nodesY = grid.NodesY( l );
		ok = ok && level.coef.Ensure( level.nodesX, level.nodesY, GL_RGBA32F, PassBuffer::Sampling::Nearest );
		ok = ok && level.rhs.Ensure( level.nodesX, level.nodesY, GL_R32F, PassBuffer::Sampling::Nearest );
		//Linear: the composite reads psi and the thickness between nodes.
		ok = ok && level.psi[ 0 ].Ensure( level.nodesX, level.nodesY, GL_R32F, PassBuffer::Sampling::Linear );
		ok = ok && level.psi[ 1 ].Ensure( level.nodesX, level.nodesY, GL_R32F, PassBuffer::Sampling::Linear );
	}
	return ok;
}

void Multigrid::Destroy()
{
	for( Level& level : levels )
	{
		level.coef.Destroy();
		level.rhs.Destroy();
		level.psi[ 0 ].Destroy();
		level.psi[ 1 ].Destroy();
		level.current = 0;
	}
	levels.clear();
}

void Multigrid::ClearSolution()
{
	for( Level& level : levels )
	{
		level.psi[ 0 ].Clear();
		level.psi[ 1 ].Clear();
	}
}

//---------------------------------------------------------------------------
BassaltPlugin::BassaltPlugin()
{
	SetMinInputs( 1 );
	SetMaxInputs( 1 );

	//The lamp runs on the host's clock: it warms in so many seconds.
	SetTimeSupported( true );

	//-------------------------------------------------------------------
	// Defaults. A 30 cm lamp on a 30 W bulb, already warm, at 3x: dropping
	// the effect on a layer shows wax moving straight away rather than an hour
	// of a lamp warming up.
	//-------------------------------------------------------------------
	params[ PT_LAMP_HEIGHT ] = 0.477f;//0.30 m
	params[ PT_GAP ]         = 0.463f;//4 mm
	params[ PT_DETAIL ]      = 2.0f;  //128 cells up
	params[ PT_SPEED ]       = ParamFromSpeed( 3.0f );
	params[ PT_AMBIENT ]     = 0.4f;  //22 C
	params[ PT_WARM ]        = 0.0f;
	params[ PT_RESET ]       = 0.0f;

	params[ PT_BULB ]     = 0.5f; //30 W
	params[ PT_BULB_LAG ] = 0.55f;//10 s
	params[ PT_COIL ]     = 0.5f; //3 W/(m K)

	params[ PT_SALT ]          = 0.15f;//0.6%: T* = 56 C, just above the warm lamp's mean
	params[ PT_WAX_AMOUNT ]    = 0.286f;//15%
	params[ PT_TENSION ]       = 0.447f;//2 mN/m
	params[ PT_WAX_VISCOSITY ] = 0.434f;//20 mPa s
	params[ PT_MELTING_POINT ] = 0.43f; //50 C

	params[ PT_BASS ]      = 0.0f;
	params[ PT_BASS_BAND ] = 1.0f / 7.0f;//2 bins
	params[ PT_KICK ]      = 0.0f;

	params[ PT_CLIP ]           = static_cast< float >( ClipMode::Behind );
	params[ PT_POUR ]           = 0.0f;
	params[ PT_POUR_THRESHOLD ] = 0.5f;
	params[ PT_CLIP_HEAT ]      = 0.0f;

	params[ PT_WAX_R ]      = 1.0f;
	params[ PT_WAX_G ]      = 0.30f;
	params[ PT_WAX_B ]      = 0.12f;
	params[ PT_TINT_R ]     = 0.85f;
	params[ PT_TINT_G ]     = 0.55f;
	params[ PT_TINT_B ]     = 0.75f;
	params[ PT_GLOW ]       = 0.5f;
	params[ PT_GLASS ]      = static_cast< float >( Glass::Flat );
	params[ PT_REFRACTION ] = 0.3f;//30 cm to the world behind

	params[ PT_VIEW ] = static_cast< float >( View::Lamp );
	params[ PT_MIX ]  = 1.0f;

	//-------------------------------------------------------------------
	// Declaration. Every numeric parameter is a plain 0..1 float: the ranges
	// live in Controls.cpp and nowhere else.
	//-------------------------------------------------------------------
	SetParamInfof( PT_LAMP_HEIGHT, "Lamp Height", FF_TYPE_STANDARD );
	SetParamInfof( PT_GAP, "Gap", FF_TYPE_STANDARD );
	SetOptionParamInfo( PT_DETAIL, "Detail", kDetailCount, params[ PT_DETAIL ] );
	for( int i = 0; i < kDetailCount; ++i )
		SetParamElementInfo( PT_DETAIL, i, kDetailNames[ i ], static_cast< float >( i ) );
	SetParamInfof( PT_SPEED, "Speed", FF_TYPE_STANDARD );
	SetParamInfof( PT_AMBIENT, "Ambient", FF_TYPE_STANDARD );
	SetParamInfo( PT_WARM, "Warm", FF_TYPE_EVENT, false );
	SetParamInfo( PT_RESET, "Reset", FF_TYPE_EVENT, false );

	SetParamInfof( PT_BULB, "Bulb", FF_TYPE_STANDARD );
	SetParamInfof( PT_BULB_LAG, "Bulb Lag", FF_TYPE_STANDARD );
	SetParamInfof( PT_COIL, "Coil", FF_TYPE_STANDARD );

	SetParamInfof( PT_SALT, "Salt", FF_TYPE_STANDARD );
	SetParamInfof( PT_WAX_AMOUNT, "Wax Amount", FF_TYPE_STANDARD );
	SetParamInfof( PT_TENSION, "Surface Tension", FF_TYPE_STANDARD );
	SetParamInfof( PT_WAX_VISCOSITY, "Wax Viscosity", FF_TYPE_STANDARD );
	SetParamInfof( PT_MELTING_POINT, "Melting Point", FF_TYPE_STANDARD );

	//An FFT buffer: Resolume shows it as an audio-source picker and writes the
	//spectrum into it every frame. With no audio routed Bass and Kick do
	//nothing, rather than the bulb reacting to silence.
	SetBufferParamInfo( PT_AUDIO, "Audio", audio::kBins, FF_USAGE_FFT );
	for( int i = 0; i < audio::kBins; ++i )
		SetParamElementInfo( PT_AUDIO, i, "", 0.0f );
	SetParamInfof( PT_BASS, "Bass", FF_TYPE_STANDARD );
	SetParamInfof( PT_BASS_BAND, "Bass Band", FF_TYPE_STANDARD );
	SetParamInfof( PT_KICK, "Kick", FF_TYPE_STANDARD );

	SetOptionParamInfo( PT_CLIP, "Clip", static_cast< int >( ClipMode::Count ), params[ PT_CLIP ] );
	for( int i = 0; i < static_cast< int >( ClipMode::Count ); ++i )
		SetParamElementInfo( PT_CLIP, i, kClipNames[ i ], static_cast< float >( i ) );
	SetParamInfo( PT_POUR, "Pour", FF_TYPE_EVENT, false );
	SetParamInfof( PT_POUR_THRESHOLD, "Pour Threshold", FF_TYPE_STANDARD );
	SetParamInfof( PT_CLIP_HEAT, "Clip Heat", FF_TYPE_STANDARD );

	//Consecutive red/green/blue parameters are what a host needs to show a
	//swatch rather than three sliders.
	SetParamInfof( PT_WAX_R, "Wax Colour", FF_TYPE_RED );
	SetParamInfof( PT_WAX_G, "Wax Colour_Green", FF_TYPE_GREEN );
	SetParamInfof( PT_WAX_B, "Wax Colour_Blue", FF_TYPE_BLUE );
	SetParamInfof( PT_TINT_R, "Liquid Tint", FF_TYPE_RED );
	SetParamInfof( PT_TINT_G, "Tint_Green", FF_TYPE_GREEN );
	SetParamInfof( PT_TINT_B, "Tint_Blue", FF_TYPE_BLUE );
	SetParamInfof( PT_GLOW, "Glow", FF_TYPE_STANDARD );
	SetOptionParamInfo( PT_GLASS, "Glass", static_cast< int >( Glass::Count ), params[ PT_GLASS ] );
	for( int i = 0; i < static_cast< int >( Glass::Count ); ++i )
		SetParamElementInfo( PT_GLASS, i, kGlassNames[ i ], static_cast< float >( i ) );
	SetParamInfof( PT_REFRACTION, "Refraction", FF_TYPE_STANDARD );

	SetOptionParamInfo( PT_VIEW, "View", static_cast< int >( View::Count ), params[ PT_VIEW ] );
	for( int i = 0; i < static_cast< int >( View::Count ); ++i )
		SetParamElementInfo( PT_VIEW, i, kViewNames[ i ], static_cast< float >( i ) );
	SetParamInfof( PT_MIX, "Mix", FF_TYPE_STANDARD );

	//-------------------------------------------------------------------
	// Groups. SetParamGroup collapses runs of consecutive same-group ids, so
	// this depends entirely on the id order in Controls.h.
	//-------------------------------------------------------------------
	for( unsigned int id = PT_LAMP_HEIGHT; id <= PT_RESET; ++id )
		SetParamGroup( id, "Lamp" );
	for( unsigned int id = PT_BULB; id <= PT_COIL; ++id )
		SetParamGroup( id, "Bulb" );
	for( unsigned int id = PT_SALT; id <= PT_MELTING_POINT; ++id )
		SetParamGroup( id, "Fluids" );
	for( unsigned int id = PT_AUDIO; id <= PT_KICK; ++id )
		SetParamGroup( id, "Audio" );
	for( unsigned int id = PT_CLIP; id <= PT_CLIP_HEAT; ++id )
		SetParamGroup( id, "Clip" );
	for( unsigned int id = PT_WAX_R; id <= PT_REFRACTION; ++id )
		SetParamGroup( id, "Look" );
	for( unsigned int id = PT_VIEW; id <= PT_MIX; ++id )
		SetParamGroup( id, "Output" );

	SetParamInfo( PT_ABOUT_TEXT, "About", FF_TYPE_TEXT, stoatworks::about::defaultText() );
	{
		FFUInt32 aboutId = PT_ABOUT_TEXT + 1;
		for( const auto& b : stoatworks::about::buttons() )
			SetParamInfo( aboutId++, b.label, FF_TYPE_EVENT, false );
	}
	for( unsigned int id = PT_ABOUT_TEXT; id < PT_COUNT; ++id )
		SetParamGroup( id, "About" );
}

//---------------------------------------------------------------------------
FFResult BassaltPlugin::InitGL( const FFGLViewportStruct* vp )
{
	diag::init();
	diag::info( std::string( "GL vendor=" ) + glStringOrUnknown( GL_VENDOR ) + " renderer="
	            + glStringOrUnknown( GL_RENDERER ) + " version=" + glStringOrUnknown( GL_VERSION ) );

	for( int id = 1; id < static_cast< int >( ShaderId::Count ); ++id )
	{
		const ShaderId shader = static_cast< ShaderId >( id );
		if( programs[ id ].shader.Compile( ShaderSource( ShaderId::Vertex ), ShaderSource( shader ) ) )
			continue;

		//Returning FF_FAIL here is invisible to the operator: the plugin simply
		//does nothing in Resolume. These lines are the only record of which pass.
		diag::error( std::string( "the " ) + ShaderName( shader ) + " shader failed to compile - the plugin will do nothing" );
		FFGLLog::LogToHost( "Bassalt: shader failed to compile" );
		DeInitGL();
		return FF_FAIL;
	}

	if( !quad.Initialise() )
	{
		diag::error( "quad geometry failed to initialise" );
		DeInitGL();
		return FF_FAIL;
	}

	diag::info( "initialised" );
	return CFFGLPlugin::InitGL( vp );
}

//---------------------------------------------------------------------------
bool BassaltPlugin::ensureBuffers( const physics::Grid& wanted, GLsizei width, GLsizei height )
{
	//A different grid is a different lamp: the state cannot be carried across,
	//so it starts again, warm.
	if( wanted != grid )
	{
		state[ 0 ].Destroy();
		state[ 1 ].Destroy();
		props.Destroy();
		reduce[ 0 ].Destroy();
		reduce[ 1 ].Destroy();
		flow.Destroy();
		inflate.Destroy();
		stateIndex    = 0;
		fresh         = true;
		speedEstimate = -1.0;
	}

	//32-bit throughout: this GPU stores to half floats by truncating, and the
	//state is fed back for as long as the lamp runs.
	bool ok = true;
	for( PassBuffer& buffer : state )
		ok = ok && buffer.Ensure( wanted.nx, wanted.ny, GL_RGBA32F, PassBuffer::Sampling::Linear );
	ok = ok && props.Ensure( wanted.nx, wanted.ny, GL_RGBA32F, PassBuffer::Sampling::Nearest );
	ok = ok && reduce[ 0 ].Ensure( ( wanted.nx + 1 + 15 ) / 16, ( wanted.ny + 1 + 15 ) / 16, GL_R32F, PassBuffer::Sampling::Nearest );
	ok = ok && reduce[ 1 ].Ensure( 1, 1, GL_R32F, PassBuffer::Sampling::Nearest );
	ok = ok && flow.Ensure( wanted );
	ok = ok && inflate.Ensure( wanted );
	if( !ok )
		return false;

	grid         = wanted;
	bufferWidth  = width;
	bufferHeight = height;
	return true;
}

//---------------------------------------------------------------------------
// The pass plumbing. Raw GL rather than the SDK's Scoped* bindings: those
// clear to 0 on exit instead of restoring, on whichever unit is active then,
// and a pass here binds up to four units -- the unwinding is wrong from three
// (millpond, AGENTS.md). Every unit this file binds is unbound once at the
// end of ProcessOpenGL instead, and `bstest --state` checks.
//---------------------------------------------------------------------------
void BassaltPlugin::Begin( PassBuffer& target, Program& program )
{
	glBindFramebuffer( GL_FRAMEBUFFER, target.GetGLID() );
	glViewport( 0, 0, static_cast< GLsizei >( target.GetWidth() ), static_cast< GLsizei >( target.GetHeight() ) );
	glUseProgram( program.Id() );
}

void BassaltPlugin::BindTexture( int unit, GLuint texture )
{
	glActiveTexture( static_cast< GLenum >( GL_TEXTURE0 + unit ) );
	glBindTexture( GL_TEXTURE_2D, texture );
}

void BassaltPlugin::Draw()
{
	quad.Draw();
}

//---------------------------------------------------------------------------
void BassaltPlugin::PropsPass()
{
	Program& p = P( ShaderId::Props );
	Begin( props, p );
	BindTexture( 0, state[ stateIndex ].TextureID() );
	p.SetInt( "State", 0 );
	p.SetInts( "Cells", grid.nx, grid.ny );
	p.Set( "Spacing", static_cast< float >( grid.dx ), static_cast< float >( grid.dy ) );
	const double xi = physics::InterfaceWidth( grid );
	p.Set( "Xi2", static_cast< float >( 2.0 * xi * xi ) );
	p.Set( "Slope", static_cast< float >( physics::CrossoverSlope( lamp.salt ) ) );
	p.Set( "Crossover", static_cast< float >( physics::Crossover( lamp.salt ) ) );
	p.Set( "LiquidBeta", static_cast< float >( physics::LiquidDensity0( lamp.salt ) * physics::kLiquidExpansion ) );
	p.Set( "ReferenceT", static_cast< float >( physics::kReferenceT ) );
	p.Set( "MeltingPoint", static_cast< float >( lamp.meltingPoint ) );
	p.Set( "FreezeWidth", static_cast< float >( physics::kFreezeWidth ) );
	p.Set( "SolidLog", static_cast< float >( std::log( physics::kSolidMultiple ) ) );
	p.Set( "WaxMu", static_cast< float >( lamp.waxViscosity ) );
	p.Set( "LiquidMu", static_cast< float >( physics::kLiquidViscosity ) );
	p.Set( "Resist", static_cast< float >( 12.0 / ( lamp.gap * lamp.gap ) ) );
	p.Set( "WaxK", static_cast< float >( physics::kWaxConductivity ) );
	p.Set( "LiquidK", static_cast< float >( physics::kLiquidConductivity ) );
	p.Set( "CoilK", static_cast< float >( lamp.coil ) );
	p.Set( "CoilBox", static_cast< float >( 0.5 * physics::kCoilWidth ), static_cast< float >( physics::kCoilHeight ) );
	Draw();
}

//---------------------------------------------------------------------------
void BassaltPlugin::Coarsen( Multigrid& system )
{
	Program& p = P( ShaderId::Coarsen );
	for( size_t l = 1; l < system.levels.size(); ++l )
	{
		Begin( system.levels[ l ].coef, p );
		BindTexture( 0, system.levels[ l - 1 ].coef.TextureID() );
		p.SetInt( "Coef", 0 );
		p.SetInts( "FineNodes", system.levels[ l - 1 ].nodesX, system.levels[ l - 1 ].nodesY );
		Draw();
	}
}

void BassaltPlugin::Smooth( Multigrid& system, int level, bool zeroGuess )
{
	Multigrid::Level& L = system.levels[ static_cast< size_t >( level ) ];
	Program& p          = P( ShaderId::Smooth );
	Begin( L.psi[ 1 - L.current ], p );
	BindTexture( 0, L.psi[ L.current ].TextureID() );
	BindTexture( 1, L.coef.TextureID() );
	BindTexture( 2, L.rhs.TextureID() );
	p.SetInt( "Psi", 0 );
	p.SetInt( "Coef", 1 );
	p.SetInt( "Rhs", 2 );
	p.SetInts( "Nodes", L.nodesX, L.nodesY );
	p.SetInt( "ZeroGuess", zeroGuess ? 1 : 0 );
	Draw();
	L.current = 1 - L.current;
}

void BassaltPlugin::Cycle( Multigrid& system, int level )
{
	Multigrid::Level& L = system.levels[ static_cast< size_t >( level ) ];

	if( level == static_cast< int >( system.levels.size() ) - 1 )
	{
		Program& p = P( ShaderId::Coarsest );
		Begin( L.psi[ 1 - L.current ], p );
		BindTexture( 0, L.coef.TextureID() );
		BindTexture( 1, L.rhs.TextureID() );
		p.SetInt( "Coef", 0 );
		p.SetInt( "Rhs", 1 );
		p.SetInts( "Nodes", L.nodesX, L.nodesY );
		p.SetInt( "Sweeps", physics::kCoarsestSweeps );
		p.Set( "Omega", static_cast< float >( physics::kCoarsestOmega ) );
		Draw();
		L.current = 1 - L.current;
		return;
	}

	//A coarse level solves for a correction, so it starts from zero.
	for( int sweep = 0; sweep < kPreSweeps; ++sweep )
		Smooth( system, level, level > 0 && sweep == 0 );

	Multigrid::Level& C = system.levels[ static_cast< size_t >( level + 1 ) ];
	{
		Program& p = P( ShaderId::Restrict );
		Begin( C.rhs, p );
		BindTexture( 0, L.psi[ L.current ].TextureID() );
		BindTexture( 1, L.coef.TextureID() );
		BindTexture( 2, L.rhs.TextureID() );
		p.SetInt( "Psi", 0 );
		p.SetInt( "Coef", 1 );
		p.SetInt( "Rhs", 2 );
		p.SetInts( "FineNodes", L.nodesX, L.nodesY );
		p.SetInts( "CoarseNodes", C.nodesX, C.nodesY );
		Draw();
	}

	Cycle( system, level + 1 );

	//The correction, and the first post-smoothing sweep with it.
	{
		Program& p = P( ShaderId::Prolong );
		Begin( L.psi[ 1 - L.current ], p );
		BindTexture( 0, L.psi[ L.current ].TextureID() );
		BindTexture( 1, C.psi[ C.current ].TextureID() );
		BindTexture( 2, L.coef.TextureID() );
		BindTexture( 3, L.rhs.TextureID() );
		p.SetInt( "Psi", 0 );
		p.SetInt( "Correction", 1 );
		p.SetInt( "Coef", 2 );
		p.SetInt( "Rhs", 3 );
		p.SetInts( "Nodes", L.nodesX, L.nodesY );
		p.Set( "Weight", 1.0f );
		Draw();
		L.current = 1 - L.current;
	}

	for( int sweep = 1; sweep < kPostSweeps; ++sweep )
		Smooth( system, level, false );
}

//---------------------------------------------------------------------------
void BassaltPlugin::FlowSolve( int cycles, bool fromZero )
{
	Multigrid::Level& fine = flow.levels[ 0 ];
	{
		Program& p = P( ShaderId::Coef );
		Begin( fine.coef, p );
		BindTexture( 0, props.TextureID() );
		p.SetInt( "Props", 0 );
		p.SetInts( "Cells", grid.nx, grid.ny );
		p.Set( "Spacing", static_cast< float >( grid.dx ), static_cast< float >( grid.dy ) );
		Draw();
	}
	{
		Program& p = P( ShaderId::Circulation );
		Begin( fine.rhs, p );
		BindTexture( 0, state[ stateIndex ].TextureID() );
		BindTexture( 1, props.TextureID() );
		p.SetInt( "State", 0 );
		p.SetInt( "Props", 1 );
		p.SetInts( "Cells", grid.nx, grid.ny );
		p.SetInts( "Nodes", fine.nodesX, fine.nodesY );
		p.Set( "Spacing", static_cast< float >( grid.dx ), static_cast< float >( grid.dy ) );
		p.Set( "Gravity", static_cast< float >( physics::kGravity ) );
		p.Set( "Beta", static_cast< float >( physics::ChemicalScale( grid, lamp.tension ) ) );
		Draw();
	}
	Coarsen( flow );

	if( fromZero )
		flow.ClearSolution();
	for( int c = 0; c < cycles; ++c )
		Cycle( flow, 0 );
}

//---------------------------------------------------------------------------
void BassaltPlugin::ReduceSpeed()
{
	const Multigrid::Level& fine = flow.levels[ 0 ];
	{
		Program& p = P( ShaderId::Reduce );
		Begin( reduce[ 0 ], p );
		BindTexture( 0, fine.psi[ fine.current ].TextureID() );
		p.SetInt( "Source", 0 );
		p.SetInts( "SourceSize", fine.nodesX, fine.nodesY );
		p.Set( "Spacing", static_cast< float >( grid.dx ), static_cast< float >( grid.dy ) );
		p.SetInt( "Mode", 0 );
		p.SetInt( "Block", 16 );
		Draw();
	}
	{
		Program& p = P( ShaderId::Reduce );
		Begin( reduce[ 1 ], p );
		BindTexture( 0, reduce[ 0 ].TextureID() );
		p.SetInt( "Source", 0 );
		p.SetInts( "SourceSize", static_cast< int >( reduce[ 0 ].GetWidth() ), static_cast< int >( reduce[ 0 ].GetHeight() ) );
		p.SetInt( "Mode", 1 );
		p.SetInt( "Block", static_cast< int >( std::max( reduce[ 0 ].GetWidth(), reduce[ 0 ].GetHeight() ) ) );
		Draw();
	}
}

double BassaltPlugin::FastestFaceNow()
{
	ReduceSpeed();
	float speed = 0.0f;
	glReadPixels( 0, 0, 1, 1, GL_RED, GL_FLOAT, &speed );
	return std::isfinite( speed ) ? static_cast< double >( speed ) : 0.0;
}

void BassaltPlugin::RequestSpeed()
{
	glBindFramebuffer( GL_FRAMEBUFFER, reduce[ 1 ].GetGLID() );
	if( speedBuffer == 0 )
	{
		glGenBuffers( 1, &speedBuffer );
		glBindBuffer( GL_PIXEL_PACK_BUFFER, speedBuffer );
		glBufferData( GL_PIXEL_PACK_BUFFER, sizeof( float ), nullptr, GL_STREAM_READ );
	}
	glBindBuffer( GL_PIXEL_PACK_BUFFER, speedBuffer );
	glReadPixels( 0, 0, 1, 1, GL_RED, GL_FLOAT, nullptr );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
	if( speedFence != nullptr )
		glDeleteSync( speedFence );
	speedFence = glFenceSync( GL_SYNC_GPU_COMMANDS_COMPLETE, 0 );
}

void BassaltPlugin::CollectSpeed()
{
	if( speedFence == nullptr )
		return;
	//A frame has passed since the request, so the wait is over before it
	//starts; it is a wait rather than a poll so that the step sequence -- and so
	//every render -- is the same from run to run.
	glClientWaitSync( speedFence, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull );
	glDeleteSync( speedFence );
	speedFence = nullptr;
	float speed = 0.0f;
	glBindBuffer( GL_PIXEL_PACK_BUFFER, speedBuffer );
	glGetBufferSubData( GL_PIXEL_PACK_BUFFER, 0, sizeof( float ), &speed );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
	speedEstimate = std::isfinite( speed ) ? static_cast< double >( speed ) : -1.0;
}

//---------------------------------------------------------------------------
void BassaltPlugin::UpdatePass( double dt, double power, double clipPower, GLuint clipTexture, const FFGLTexCoords& maxUV )
{
	const int target = 1 - stateIndex;
	Program& p       = P( ShaderId::Update );
	Begin( state[ target ], p );
	BindTexture( 0, state[ stateIndex ].TextureID() );
	BindTexture( 1, props.TextureID() );
	BindTexture( 2, flow.Solution() );
	BindTexture( 3, clipTexture );
	BindTexture( 4, reduce[ 1 ].TextureID() );
	p.SetInt( "State", 0 );
	p.SetInt( "Props", 1 );
	p.SetInt( "Psi", 2 );
	p.SetInt( "Clip", 3 );
	p.SetInt( "Fastest", 4 );
	p.SetInts( "Cells", grid.nx, grid.ny );
	p.Set( "Spacing", static_cast< float >( grid.dx ), static_cast< float >( grid.dy ) );
	p.Set( "Dt", static_cast< float >( dt ) );
	p.Set( "Capacity", static_cast< float >( physics::kHeatCapacity ) );
	p.Set( "Ambient", static_cast< float >( lamp.ambient ) );
	p.Set( "FaceRate", static_cast< float >( physics::FaceLossRate( lamp.gap ) ) );
	p.Set( "CapRate", capOn ? static_cast< float >( physics::kCapLoss / ( physics::kHeatCapacity * grid.dy ) ) : 0.0f );
	p.Set( "BulbPower", static_cast< float >( power ) );
	const double sigma = physics::kBulbSpread * lamp.width;
	double sum         = 0.0;
	for( int i = 0; i < grid.nx; ++i )
	{
		const double x = ( i + 0.5 ) * grid.dx - 0.5 * grid.nx * grid.dx;
		sum += std::exp( -0.5 * ( x / sigma ) * ( x / sigma ) );
	}
	p.Set( "BulbSigma", static_cast< float >( sigma ) );
	p.Set( "BulbNorm", static_cast< float >( 1.0 / sum ) );
	p.Set( "Gap", static_cast< float >( lamp.gap ) );
	p.Set( "ClipPower", static_cast< float >( clipPower ) );
	p.Set( "MaxUV", maxUV.s, maxUV.t );
	p.Set( "UVRelax", kDyeRelax );
	p.Set( "SpeedCap", static_cast< float >( physics::kCourant * std::min( grid.dx, grid.dy ) / dt ) );
	Draw();
	stateIndex = target;
}

//---------------------------------------------------------------------------
void BassaltPlugin::InterfacePass( double dt )
{
	const int target = 1 - stateIndex;
	Program& p       = P( ShaderId::Interface );
	Begin( state[ target ], p );
	BindTexture( 0, state[ stateIndex ].TextureID() );
	BindTexture( 1, props.TextureID() );
	p.SetInt( "State", 0 );
	p.SetInt( "Props", 1 );
	p.SetInts( "Cells", grid.nx, grid.ny );
	p.Set( "Spacing", static_cast< float >( grid.dx ), static_cast< float >( grid.dy ) );
	p.Set( "Dt", static_cast< float >( dt ) );
	p.Set( "Mobility", static_cast< float >( physics::InterfaceMobility( grid ) ) );
	p.Set( "MobilityFloor", static_cast< float >( physics::kMobilityFloor ) );
	Draw();
	stateIndex = target;
}

//---------------------------------------------------------------------------
void BassaltPlugin::EventPass( int mode, float temperature, GLuint clipTexture, const FFGLTexCoords& maxUV )
{
	const int target = 1 - stateIndex;
	Program& p       = P( ShaderId::Event );
	Begin( state[ target ], p );
	BindTexture( 0, state[ stateIndex ].TextureID() );
	BindTexture( 1, clipTexture );
	p.SetInt( "State", 0 );
	p.SetInt( "Clip", 1 );
	p.SetInt( "Mode", mode );
	p.SetInts( "Cells", grid.nx, grid.ny );
	p.Set( "Spacing", static_cast< float >( grid.dx ), static_cast< float >( grid.dy ) );
	p.Set( "Xi", static_cast< float >( physics::InterfaceWidth( grid ) ) );
	p.Set( "WaxHeight", static_cast< float >( WaxAmountFromParam( params[ PT_WAX_AMOUNT ] ) * lamp.height ) );
	p.Set( "Temperature", temperature );
	p.Set( "Threshold", std::clamp( params[ PT_POUR_THRESHOLD ], 0.0f, 1.0f ) );
	p.Set( "MaxUV", maxUV.s, maxUV.t );
	Draw();
	stateIndex = target;
}

//---------------------------------------------------------------------------
void BassaltPlugin::InflatePass()
{
	Multigrid::Level& fine = inflate.levels[ 0 ];
	Program& p             = P( ShaderId::Inflate );
	for( int mode = 0; mode < 2; ++mode )
	{
		Begin( mode == 0 ? fine.coef : fine.rhs, p );
		BindTexture( 0, state[ stateIndex ].TextureID() );
		p.SetInt( "State", 0 );
		p.SetInts( "Cells", grid.nx, grid.ny );
		p.Set( "Spacing", static_cast< float >( grid.dx ), static_cast< float >( grid.dy ) );
		p.SetInt( "Mode", mode );
		p.Set( "Penalty", kInflatePenalty );
		Draw();
	}
	Coarsen( inflate );
	//Warm-started from last frame's lenses, which moved by a fraction of a
	//cell.
	Cycle( inflate, 0 );
	haveThickness = true;
}

//---------------------------------------------------------------------------
double BassaltPlugin::BulbTarget() const
{
	return std::max( 0.0, static_cast< double >( BulbFromParam( params[ PT_BULB ] ) )
	                          + static_cast< double >( BassFromParam( params[ PT_BASS ] ) ) * analyser.Bass() );
}

void BassaltPlugin::Simulate( double dtFrame, GLuint clipTexture, const FFGLTexCoords& maxUV )
{
	const double limit          = std::min( physics::DiffusionLimit( grid, lamp ), physics::CapillaryLimit( grid, lamp ) );
	const double interfaceLimit = physics::InterfaceLimit( grid );
	const double target    = BulbTarget();
	const double tau       = BulbLagFromParam( params[ PT_BULB_LAG ] );
	const double clipPower = ClipHeatFromParam( params[ PT_CLIP_HEAT ] );
	const double cell      = std::min( grid.dx, grid.dy );

	CollectSpeed();

	double remaining = dtFrame;
	int substeps     = 0;
	while( remaining > 1e-9 && substeps < physics::kMaxSubsteps )
	{
		PropsPass();

		double speed = 0.0;
		if( flowFrozen )
			flow.ClearSolution();
		else
		{
			FlowSolve( physics::kCyclesPerStep, false );
			//The step's speed: last frame's fastest face with a margin, or --
			//after an event, or on the first frame, when there is no last
			//frame worth the name -- this solve's own, read now.
			if( speedEstimate < 0.0 )
				speedEstimate = FastestFaceNow();
			else
				ReduceSpeed();
			speed = physics::kSpeedMargin * speedEstimate;
		}

		double dt = std::min( remaining, limit );
		if( speed > 0.0 )
			dt = std::min( dt, physics::kCourant * cell / speed );

		bulbPower          = physics::BulbStep( bulbPower, target, dt, tau );
		const double power = bulbPower + pendingKick / dt;
		pendingKick        = 0.0;

		UpdatePass( dt, power, clipPower, clipTexture, maxUV );

		//Cahn-Hilliard, in as many subcycles as its explicit limit needs.
		const int subcycles = std::max( 1, static_cast< int >( std::ceil( dt / interfaceLimit ) ) );
		for( int k = 0; k < subcycles; ++k )
		{
			PropsPass();
			InterfacePass( dt / subcycles );
		}

		if( keepLog )
			stepLog.push_back( StepRecord { dt, power, clipPower, speed } );

		remaining -= dt;
		simTime += dt;
		++substeps;
	}
	lastSubsteps  = substeps;
	lastShortfall = remaining;

	//The last solve's fastest face, already reduced, for the next frame.
	if( !flowFrozen && substeps > 0 )
		RequestSpeed();
}

//---------------------------------------------------------------------------
void BassaltPlugin::UpdateClock()
{
	const double wallNow = wallSeconds();
	if( wallStart < 0.0 )
		wallStart = wallNow;

	const double raw = hostTime;

	//Resolume has been seen sending both seconds and milliseconds through
	//SetTime. Vote on it against the wall clock, then stop asking.
	if( clockScale == 0.0 && raw >= 0.0 && lastRawTime >= 0.0 && lastWallTime >= 0.0 )
	{
		const double hostDelta = raw - lastRawTime;
		const double wallDelta = wallNow - lastWallTime;
		if( hostDelta > 0.0 && wallDelta >= 0.0005 )
		{
			const double ratio = hostDelta / wallDelta;
			if( ratio > 0.1 && ratio < 10.0 )
				++secondsVotes;
			else if( ratio > 100.0 && ratio < 10000.0 )
				++millisVotes;

			if( secondsVotes >= kClockVotes || millisVotes >= kClockVotes )
			{
				clockScale = millisVotes > secondsVotes ? 0.001 : 1.0;
				//The clock is about to jump from wall time to the host's own;
				//the frame that settles the vote takes no time.
				settledJump = true;
			}
		}
	}

	if( raw >= 0.0 )
		lastRawTime = raw;
	lastWallTime = wallNow;

	now = ( raw >= 0.0 && clockScale != 0.0 ) ? raw * clockScale : wallNow - wallStart;

	if( ++clockFrames == 60 )
		diag::info( "host clock at frame 60: raw=" + std::to_string( raw ) + " scale=" + std::to_string( clockScale ) );
}

void BassaltPlugin::UpdateAudio( double dt )
{
	float bins[ audio::kBins ] = {};
	int binCount              = 0;
	if( const ParamInfo* info = FindParamInfo( PT_AUDIO ) )
	{
		binCount = static_cast< int >( std::min< size_t >( info->elements.size(), audio::kBins ) );
		for( int i = 0; i < binCount; ++i )
			bins[ i ] = info->elements[ static_cast< size_t >( i ) ].value;
	}

	audio::Settings settings;
	settings.sensitivity = 0.6f;
	settings.bassBins    = BassBandFromParam( params[ PT_BASS_BAND ] );
	settings.prime       = audioPriming;
	analyser.Update( bins, binCount, static_cast< float >( dt ), settings );

	//A hit's heat, its size following how hard it was.
	if( analyser.Fired() )
		pendingKick += KickFromParam( params[ PT_KICK ] ) * ( 0.35 + 0.65 * analyser.Kick() );
}

//---------------------------------------------------------------------------
FFResult BassaltPlugin::ProcessOpenGL( ProcessOpenGLStruct* pgl )
{
	if( pgl == nullptr || pgl->numInputTextures < 1 || pgl->inputTextures[ 0 ] == nullptr )
		return FF_FAIL;

	const FFGLTextureStruct& source = *( pgl->inputTextures[ 0 ] );
	const GLsizei width             = static_cast< GLsizei >( source.Width );
	const GLsizei height            = static_cast< GLsizei >( source.Height );
	if( width <= 0 || height <= 0 )
		return FF_FAIL;

	ScopedGLState restore;
	const GLint* hostViewport = restore.saved.viewport;
	glDisable( GL_BLEND );

	//-------------------------------------------------------------------
	// Time. Frame-relative throughout: the host's clock overflows a float
	// within hours, so only differences of it are ever taken, in double.
	//-------------------------------------------------------------------
	UpdateClock();
	if( settledJump )
	{
		lastNow     = -1.0;
		settledJump = false;
	}
	const double hostDt = lastNow >= 0.0 ? std::clamp( now - lastNow, 0.0, kMaxFrameDelta ) : 0.0;
	lastNow             = now;
	UpdateAudio( hostDt );

	//-------------------------------------------------------------------
	// The lamp, in metres.
	//-------------------------------------------------------------------
	lamp = CurrentLamp();
	lamp.width = lamp.height * static_cast< double >( width ) / static_cast< double >( height );

	const int detail              = optionIndex( params[ PT_DETAIL ], kDetailCount );
	const physics::Grid wanted    = physics::ChooseGrid( lamp.width, lamp.height, kDetailCells[ detail ] );
	if( !ensureBuffers( wanted, width, height ) )
	{
		diag::error( "could not allocate the pass buffers" );
		return FF_FAIL;
	}
	//The metres may have moved without the cell counts doing so.
	grid.dx = wanted.dx;
	grid.dy = wanted.dy;

	const FFGLTexCoords maxUV = GetMaxGLTexCoords( source );
	const float warmT = static_cast< float >(
		lamp.ambient + BulbTarget() / ( physics::FaceConductance( lamp ) + physics::CapConductance( lamp ) ) );

	if( fresh || resetWanted )
	{
		EventPass( 0, static_cast< float >( lamp.ambient ), source.Handle, maxUV );
		flow.ClearSolution();
		inflate.ClearSolution();
		bulbPower = 0.0;
		if( fresh )
		{
			//A lamp dropped on a layer starts warm: see the constructor.
			EventPass( 1, warmT, source.Handle, maxUV );
			bulbPower = BulbTarget();
		}
		fresh         = false;
		resetWanted   = false;
		speedEstimate = -1.0;
	}
	if( warmWanted )
	{
		EventPass( 1, warmT, source.Handle, maxUV );
		bulbPower     = BulbTarget();
		warmWanted    = false;
		speedEstimate = -1.0;
	}
	if( pourWanted )
	{
		EventPass( 2, 0.0f, source.Handle, maxUV );
		pourWanted    = false;
		speedEstimate = -1.0;
	}

	const double dt = hostDt * SpeedFromParam( params[ PT_SPEED ] );
	if( dt > 0.0 )
		Simulate( dt, source.Handle, maxUV );
	else
		lastSubsteps = 0;

	const int view     = optionIndex( params[ PT_VIEW ], static_cast< int >( View::Count ) );
	const int clipMode = optionIndex( params[ PT_CLIP ], static_cast< int >( ClipMode::Count ) );
	const int glass    = optionIndex( params[ PT_GLASS ], static_cast< int >( Glass::Count ) );
	haveThickness      = false;
	if( view == static_cast< int >( View::Lamp ) && clipMode == static_cast< int >( ClipMode::Behind ) )
		InflatePass();

	//-------------------------------------------------------------------
	// Composite, into the host's framebuffer and viewport.
	//-------------------------------------------------------------------
	glBindFramebuffer( GL_FRAMEBUFFER, pgl->HostFBO );
	glViewport( hostViewport[ 0 ], hostViewport[ 1 ], hostViewport[ 2 ], hostViewport[ 3 ] );
	{
		Program& p = P( ShaderId::Composite );
		glUseProgram( p.Id() );
		BindTexture( 0, source.Handle );
		BindTexture( 1, state[ stateIndex ].TextureID() );
		BindTexture( 2, inflate.Solution() );
		BindTexture( 3, flow.Solution() );
		p.SetInt( "InputTexture", 0 );
		p.SetInt( "StateTexture", 1 );
		p.SetInt( "ThickTexture", 2 );
		p.SetInt( "PsiTexture", 3 );
		p.Set( "MaxUV", maxUV.s, maxUV.t );
		p.SetInts( "Cells", grid.nx, grid.ny );
		p.Set( "Spacing", static_cast< float >( grid.dx ), static_cast< float >( grid.dy ) );
		p.Set( "LampSize", static_cast< float >( grid.nx * grid.dx ), static_cast< float >( grid.ny * grid.dy ) );
		p.Set( "Gap", static_cast< float >( lamp.gap ) );
		p.SetInt( "ClipMode", clipMode );
		p.SetInt( "GlassMode", glass );
		p.SetInt( "ViewMode", view );
		p.Set( "MixAmount", std::clamp( params[ PT_MIX ], 0.0f, 1.0f ) );
		p.Set( "WaxColour", params[ PT_WAX_R ], params[ PT_WAX_G ], params[ PT_WAX_B ] );
		p.Set( "Tint", params[ PT_TINT_R ], params[ PT_TINT_G ], params[ PT_TINT_B ] );
		p.Set( "Glow", GlowFromParam( params[ PT_GLOW ] ) );
		p.Set( "BulbFraction", static_cast< float >( bulbPower / 60.0 ) );
		p.Set( "Distance", RefractionFromParam( params[ PT_REFRACTION ] ) );
		p.Set( "WaxIndex", static_cast< float >( physics::kWaxIndex ) );
		p.Set( "WaterIndex", static_cast< float >( physics::kWaterIndex ) );
		p.Set( "Slope", static_cast< float >( physics::CrossoverSlope( lamp.salt ) ) );
		p.Set( "Crossover", static_cast< float >( physics::Crossover( lamp.salt ) ) );
		p.Set( "LiquidBeta", static_cast< float >( physics::LiquidDensity0( lamp.salt ) * physics::kLiquidExpansion ) );
		p.Set( "ReferenceT", static_cast< float >( physics::kReferenceT ) );
		p.SetInt( "UseThickness", haveThickness ? 1 : 0 );
		Draw();
	}

	//Everything this frame bound, unbound: see Begin().
	for( int unit = 4; unit >= 0; --unit )
		BindTexture( unit, 0 );
	glUseProgram( 0 );

	return FF_SUCCESS;
}

//---------------------------------------------------------------------------
void BassaltPlugin::SolveForTest( int cycles, bool fromZero )
{
	if( grid.nx <= 0 )
		return;
	ScopedGLState restore;
	GLint previousFBO = 0;
	glGetIntegerv( GL_FRAMEBUFFER_BINDING, &previousFBO );
	lamp = CurrentLamp();
	lamp.width = grid.nx * grid.dx;
	PropsPass();
	FlowSolve( cycles, fromZero );
	speedEstimate = -1.0;
	for( int unit = 4; unit >= 0; --unit )
		BindTexture( unit, 0 );
	glUseProgram( 0 );
	glBindFramebuffer( GL_FRAMEBUFFER, static_cast< GLuint >( previousFBO ) );
}

//---------------------------------------------------------------------------
FFResult BassaltPlugin::DeInitGL()
{
	for( Program& program : programs )
		program.Release();
	quad.Release();

	state[ 0 ].Destroy();
	state[ 1 ].Destroy();
	props.Destroy();
	reduce[ 0 ].Destroy();
	reduce[ 1 ].Destroy();
	flow.Destroy();
	inflate.Destroy();
	if( speedFence != nullptr )
		glDeleteSync( speedFence );
	speedFence = nullptr;
	if( speedBuffer != 0 )
		glDeleteBuffers( 1, &speedBuffer );
	speedBuffer   = 0;
	speedEstimate = -1.0;

	grid       = physics::Grid {};
	stateIndex = 0;
	fresh      = true;
	return FF_SUCCESS;
}

//---------------------------------------------------------------------------
physics::Lamp BassaltPlugin::CurrentLamp() const
{
	physics::Lamp l;
	l.height       = LampHeightFromParam( params[ PT_LAMP_HEIGHT ] );
	l.width        = grid.nx > 0 ? grid.nx * grid.dx : l.height * 16.0 / 9.0;
	l.gap          = GapFromParam( params[ PT_GAP ] );
	l.ambient      = AmbientFromParam( params[ PT_AMBIENT ] );
	l.salt         = SaltFromParam( params[ PT_SALT ] );
	l.tension      = TensionFromParam( params[ PT_TENSION ] );
	l.waxViscosity = WaxViscosityFromParam( params[ PT_WAX_VISCOSITY ] );
	l.meltingPoint = MeltingPointFromParam( params[ PT_MELTING_POINT ] );
	l.coil         = CoilFromParam( params[ PT_COIL ] );
	return l;
}

GLuint BassaltPlugin::StateTextureID() const
{
	return state[ stateIndex ].TextureID();
}

void BassaltPlugin::LoadStateForTest( const std::vector< float >& rgba )
{
	if( grid.nx <= 0 || rgba.size() != static_cast< size_t >( grid.nx ) * grid.ny * 4 )
		return;
	glBindTexture( GL_TEXTURE_2D, state[ stateIndex ].TextureID() );
	glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, grid.nx, grid.ny, GL_RGBA, GL_FLOAT, rgba.data() );
	glBindTexture( GL_TEXTURE_2D, 0 );
	flow.ClearSolution();
	inflate.ClearSolution();
	speedEstimate = -1.0;
}

void BassaltPlugin::SetClockScaleForTest( double scale )
{
	clockScale = scale;
}

//---------------------------------------------------------------------------
FFResult BassaltPlugin::SetTime( double time )
{
	hostTime = time;
	return FF_SUCCESS;
}

char* BassaltPlugin::GetTextParameter( unsigned int index )
{
	if( index == PT_ABOUT_TEXT )
	{
		static const std::string text = stoatworks::about::textParam( 0 );
		return const_cast< char* >( text.c_str() );
	}
	return CFFGLPlugin::GetTextParameter( index );
}

FFResult BassaltPlugin::SetTextParameter( unsigned int index, const char* value )
{
	// The base class fails, and a failed default deletes the instance. The
	// About line is display-only, so there is nothing to store -- but it has to
	// say so successfully.
	if( index == PT_ABOUT_TEXT )
		return FF_SUCCESS;
	return CFFGLPlugin::SetTextParameter( index, value );
}

FFResult BassaltPlugin::SetFloatParameter( unsigned int index, float value )
{
	if( index >= PT_COUNT )
		return FF_FAIL;

	if( index >= PT_ABOUT_TEXT )
		return stoatworks::about::handleParam( index - PT_ABOUT_TEXT, value ) ? FF_SUCCESS : FF_FAIL;

	//The buttons act on the press, once: an event parameter arrives as 1 on
	//press and 0 on release, and a host is free to restate either.
	const bool down = value >= 0.5f;
	if( index == PT_WARM )
	{
		if( down && !warmHeld )
			warmWanted = true;
		warmHeld = down;
	}
	else if( index == PT_RESET )
	{
		if( down && !resetHeld )
			resetWanted = true;
		resetHeld = down;
	}
	else if( index == PT_POUR )
	{
		if( down && !pourHeld )
			pourWanted = true;
		pourHeld = down;
	}

	params[ index ] = value;
	return FF_SUCCESS;
}

float BassaltPlugin::GetFloatParameter( unsigned int index )
{
	if( index >= PT_COUNT )
		return 0.0f;
	return params[ index ];
}

} // namespace bassalt
