/**
    bstest -- render Bassalt offline, and measure what its lamp is doing.

    It drives the REAL plugin class, through the same ProcessOpenGL a host
    calls, on a synthetic 60 fps clock. Where a check needs a field, it reads
    back the texture the shipping passes wrote. A test that exercises a
    reimplementation tests the reimplementation.

        bstest --out /tmp/frame.png     the card, through the lamp
        bstest --card /tmp/card.png     the card on its own
        bstest --list                   every parameter and its default
        bstest --pipe                   raw frames in, raw frames out
        bstest --film N                 N frames of the card, raw frames out

    `--script` is the fleet's cue format: `frame  Parameter Name  value`, held
    before the first key and after the last, interpolated between. A button
    press is three keys (`29 Warm 0`, `30 Warm 1`, `31 Warm 0`).

    The claims, one flag each, in the order the README makes them:

        --still        a level, cold lamp does not move: |u| is exactly 0; a
                       neutral blob's spurious currents are bounded and fall
        --volume       wax is conserved through convection; heat makes no new
                       extrema
        --crossover    a blob rises iff T > T*, and T* moves with Salt as the
                       linear density laws say
        --darcy        an isolated blob moves at the Hele-Shaw speed, and the
                       GPU's solve is the CPU's solve of the same system
        --multigrid    the residual falls by the stated factor per cycle and
                       meets the step's tolerance
        --diffusion    a hot spot spreads as sigma^2 = sigma0^2 + 2 kappa t
        --rt           a light layer under a heavy one: modes below the
                       capillary cutoff grow at the Hele-Shaw rate, above it
                       decay
        --heat         the mean temperature follows the lumped law, and every
                       joule is accounted for
        --bulb         the bulb's lag is 63% at its time constant; a primed
                       onset detector fires on frame 1
        --glass        Flat is the identity; Cylinder is Snell, at two rasters
        --state        the host's GL state comes back as it went in
        --negative     every check above, against a wrong model, must fail
        --mutate       one character of the shipped GLSL changed must fail
        --bench        time a frame at 720p through 4K

    ----------------------------------------------------------- rasters

    The lamp is simulated on a grid sized in METRES, so the physics checks are
    raster-independent by construction and run at one small raster. The optics
    checks read pixels, so they run at two rasters.
*/

#include "Bassalt.h"
#include "Controls.h"
#include "Physics.h"
#include "Shaders.h"

#include <OpenGL/OpenGL.h>
#include <OpenGL/gl3.h>
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace bassalt;
namespace ph = bassalt::physics;

namespace
{
constexpr double kPi = 3.14159265358979323846;

using Floats  = std::vector< float >;
using Doubles = std::vector< double >;
using Bytes   = std::vector< unsigned char >;
//---------------------------------------------------------------------------
// Reporting.
//---------------------------------------------------------------------------
int g_failures = 0;

std::string fmt( const char* format, ... )
{
	char buffer[ 1024 ];
	va_list args;
	va_start( args, format );
	std::vsnprintf( buffer, sizeof( buffer ), format, args );
	va_end( args );
	return buffer;
}

void Check( bool condition, const std::string& message )
{
	std::printf( "  %s  %s\n", condition ? "ok  " : "FAIL", message.c_str() );
	if( !condition )
		++g_failures;
}

int Verdict()
{
	std::printf( "\n  %s\n", g_failures == 0 ? "PASS" : "FAIL" );
	return g_failures == 0 ? 0 : 1;
}

//---------------------------------------------------------------------------
// A PNG writer. zlib ships with the OS, so this is a few chunk headers and a
// CRC rather than a dependency.
//---------------------------------------------------------------------------
void putU32( Bytes& out, uint32_t value )
{
	out.push_back( static_cast< unsigned char >( value >> 24 ) );
	out.push_back( static_cast< unsigned char >( value >> 16 ) );
	out.push_back( static_cast< unsigned char >( value >> 8 ) );
	out.push_back( static_cast< unsigned char >( value ) );
}

void putChunk( Bytes& out, const char* type, const Bytes& data )
{
	putU32( out, static_cast< uint32_t >( data.size() ) );
	const size_t start = out.size();
	out.insert( out.end(), type, type + 4 );
	out.insert( out.end(), data.begin(), data.end() );
	uLong crc = crc32( 0L, Z_NULL, 0 );
	crc       = crc32( crc, out.data() + start, static_cast< uInt >( 4 + data.size() ) );
	putU32( out, static_cast< uint32_t >( crc ) );
}

/// `rgba` is floats, row 0 at the BOTTOM (GL's order); the file is written top
/// row first, which is the only place anything here flips.
bool writePng( const std::string& path, int width, int height, const Floats& rgba )
{
	Bytes raw;
	raw.reserve( static_cast< size_t >( height ) * ( 1 + static_cast< size_t >( width ) * 4 ) );
	for( int y = height - 1; y >= 0; --y )
	{
		raw.push_back( 0 );
		for( int x = 0; x < width; ++x )
			for( int c = 0; c < 4; ++c )
			{
				const float v = rgba[ ( static_cast< size_t >( y ) * width + x ) * 4 + c ];
				raw.push_back( static_cast< unsigned char >( std::lround( std::clamp( v, 0.0f, 1.0f ) * 255.0f ) ) );
			}
	}

	uLongf compressedSize = compressBound( static_cast< uLong >( raw.size() ) );
	Bytes compressed( compressedSize );
	if( compress2( compressed.data(), &compressedSize, raw.data(), static_cast< uLong >( raw.size() ), 6 ) != Z_OK )
		return false;
	compressed.resize( compressedSize );

	Bytes png = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n' };
	Bytes ihdr;
	putU32( ihdr, static_cast< uint32_t >( width ) );
	putU32( ihdr, static_cast< uint32_t >( height ) );
	ihdr.insert( ihdr.end(), { 8, 6, 0, 0, 0 } );
	putChunk( png, "IHDR", ihdr );
	putChunk( png, "IDAT", compressed );
	putChunk( png, "IEND", {} );

	FILE* file = fopen( path.c_str(), "wb" );
	if( file == nullptr )
		return false;
	const size_t written = fwrite( png.data(), 1, png.size(), file );
	fclose( file );
	return written == png.size();
}

//---------------------------------------------------------------------------
// Integer hashing, for the card. Never fract( sin( x ) * 43758.5453 ): that is
// the driver's answer, and two machines disagree about it.
//---------------------------------------------------------------------------
uint32_t lowbias32( uint32_t x )
{
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

double hash01( uint32_t a, uint32_t b = 0 )
{
	return static_cast< double >( lowbias32( a ^ lowbias32( b + 0x9e3779b9U ) ) ) / 4294967296.0;
}

//---------------------------------------------------------------------------
// GL plumbing.
//---------------------------------------------------------------------------
CGLContextObj createContext()
{
	//Accelerated first; fall back so the harness still runs somewhere without
	//a GPU, where it will at least prove the shaders compile.
	const CGLPixelFormatAttribute accelerated[] = {
		kCGLPFAOpenGLProfile, static_cast< CGLPixelFormatAttribute >( kCGLOGLPVersion_GL4_Core ),
		kCGLPFAAccelerated,
		kCGLPFAColorSize, static_cast< CGLPixelFormatAttribute >( 24 ),
		kCGLPFAAlphaSize, static_cast< CGLPixelFormatAttribute >( 8 ),
		static_cast< CGLPixelFormatAttribute >( 0 )
	};
	const CGLPixelFormatAttribute software[] = {
		kCGLPFAOpenGLProfile, static_cast< CGLPixelFormatAttribute >( kCGLOGLPVersion_GL4_Core ),
		kCGLPFAColorSize, static_cast< CGLPixelFormatAttribute >( 24 ),
		kCGLPFAAlphaSize, static_cast< CGLPixelFormatAttribute >( 8 ),
		static_cast< CGLPixelFormatAttribute >( 0 )
	};

	CGLPixelFormatObj format = nullptr;
	GLint formatCount        = 0;
	if( CGLChoosePixelFormat( accelerated, &format, &formatCount ) != kCGLNoError || format == nullptr )
	{
		if( CGLChoosePixelFormat( software, &format, &formatCount ) != kCGLNoError || format == nullptr )
			return nullptr;
	}

	CGLContextObj context = nullptr;
	const CGLError error  = CGLCreateContext( format, nullptr, &context );
	CGLDestroyPixelFormat( format );
	if( error != kCGLNoError )
		return nullptr;

	CGLSetCurrentContext( context );
	return context;
}

GLuint makeTexture( int width, int height, const float* pixels )
{
	GLuint texture = 0;
	glGenTextures( 1, &texture );
	glBindTexture( GL_TEXTURE_2D, texture );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, pixels );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glBindTexture( GL_TEXTURE_2D, 0 );
	return texture;
}

/// The host's texture can be bigger than its picture (HardwareWidth > Width),
/// with the picture in the corner and MaxUV saying how much of it is real.
/// A padded copy of `picture`: magenta everywhere outside it.
GLuint paddedTexture( const Floats& picture, int width, int height, int hw, int hh )
{
	Floats padded( static_cast< size_t >( hw ) * hh * 4 );
	for( int y = 0; y < hh; ++y )
		for( int x = 0; x < hw; ++x )
		{
			float* o = &padded[ ( static_cast< size_t >( y ) * hw + x ) * 4 ];
			if( x < width && y < height )
				std::memcpy( o, &picture[ ( static_cast< size_t >( y ) * width + x ) * 4 ], 4 * sizeof( float ) );
			else
			{
				o[ 0 ] = 1.0f;
				o[ 1 ] = 0.0f;
				o[ 2 ] = 1.0f;
				o[ 3 ] = 1.0f;
			}
		}
	return makeTexture( hw, hh, padded.data() );
}

//---------------------------------------------------------------------------
// Parameters by display name, so the automation reads as English.
//---------------------------------------------------------------------------
struct NamedParameter
{
	std::string name;
	unsigned int index;
	float value;
	std::string kind;
};

const char* kindName( unsigned int type )
{
	switch( type )
	{
	case FF_TYPE_BOOLEAN: return "bool";
	case FF_TYPE_EVENT: return "event";
	case FF_TYPE_RED: return "red";
	case FF_TYPE_GREEN: return "green";
	case FF_TYPE_BLUE: return "blue";
	case FF_TYPE_XPOS: return "xpos";
	case FF_TYPE_YPOS: return "ypos";
	case FF_TYPE_OPTION: return "option";
	case FF_TYPE_BUFFER: return "buffer";
	case FF_TYPE_STANDARD: return "standard";
	case FF_TYPE_TEXT: return "text";
	default: return "other";
	}
}

std::vector< NamedParameter > listParameters( BassaltPlugin& plugin )
{
	std::vector< NamedParameter > list;
	for( unsigned int i = 0; i < PT_COUNT; ++i )
	{
		const char* const name = plugin.GetParamName( i );
		list.push_back( NamedParameter { name ? name : "?", i, plugin.GetFloatParameter( i ),
		                                 kindName( plugin.GetParamType( i ) ) } );
	}
	return list;
}

bool applySetting( BassaltPlugin& plugin, const std::string& assignment, std::string& error )
{
	const size_t equals = assignment.find( '=' );
	if( equals == std::string::npos )
	{
		error = "expected Name=Value";
		return false;
	}

	const std::string name  = assignment.substr( 0, equals );
	const std::string value = assignment.substr( equals + 1 );

	for( const NamedParameter& parameter : listParameters( plugin ) )
	{
		if( parameter.name != name )
			continue;
		plugin.SetFloatParameter( parameter.index, std::strtof( value.c_str(), nullptr ) );
		return true;
	}

	error = "no parameter called '" + name + "'";
	return false;
}

//---------------------------------------------------------------------------
// --script: one 'frame Parameter Name value' per line. Same format as the
// rest of the fleet, so one filming script drives any of them.
//---------------------------------------------------------------------------
using Track = std::vector< std::pair< int, float > >;

std::map< std::string, Track > loadScript( const std::string& path, std::string& error )
{
	std::map< std::string, Track > tracks;
	std::ifstream file( path );
	if( !file )
	{
		error = "cannot open " + path;
		return tracks;
	}

	std::string line;
	int lineNumber = 0;
	while( std::getline( file, line ) )
	{
		++lineNumber;
		const size_t hash = line.find( '#' );
		if( hash != std::string::npos )
			line.erase( hash );
		std::istringstream in( line );

		int frame = 0;
		if( !( in >> frame ) )
			continue;

		//The name is everything up to the last token: parameters have spaces
		//in them ("Pebble Size") and the value never does.
		std::vector< std::string > words;
		std::string word;
		while( in >> word )
			words.push_back( word );
		if( words.size() < 2 )
		{
			error = path + ":" + std::to_string( lineNumber ) + ": expected `frame Parameter Name value`";
			return {};
		}

		const float value = std::strtof( words.back().c_str(), nullptr );
		words.pop_back();
		std::string name = words.front();
		for( size_t i = 1; i < words.size(); ++i )
			name += " " + words[ i ];

		tracks[ name ].emplace_back( frame, value );
	}

	for( auto& entry : tracks )
		std::sort( entry.second.begin(), entry.second.end() );
	return tracks;
}

float valueAt( const Track& track, int frame )
{
	if( track.empty() )
		return 0.0f;
	if( frame <= track.front().first )
		return track.front().second;
	if( frame >= track.back().first )
		return track.back().second;
	for( size_t i = 0; i + 1 < track.size(); ++i )
	{
		const auto& a = track[ i ];
		const auto& b = track[ i + 1 ];
		if( frame >= a.first && frame <= b.first )
		{
			if( b.first == a.first )
				return b.second;
			const float t = static_cast< float >( frame - a.first ) / static_cast< float >( b.first - a.first );
			return a.second + ( b.second - a.second ) * t;
		}
	}
	return track.back().second;
}
//---------------------------------------------------------------------------
// The card: what the lamp is put in front of. Stripes of colour and a grid of
// dots behind (so refraction has something to bend everywhere, which the
// dead-control sweep needs), and a bright ring and bar in the middle (so Pour
// has a shape to pour). Rows are v = 0 first, the way GL stores a texture.
//---------------------------------------------------------------------------
Floats buildCard( int width, int height )
{
	Floats card( static_cast< size_t >( width ) * height * 4 );
	const double aspect = static_cast< double >( width ) / height;
	for( int y = 0; y < height; ++y )
		for( int x = 0; x < width; ++x )
		{
			const double u = ( x + 0.5 ) / height;//square units
			const double v = ( y + 0.5 ) / height;

			//Diagonal stripes of six colours, dim, so the bright shape stands out.
			const double band = std::fmod( 6.0 * ( u + 0.6 * v ), 6.0 );
			const int stripe  = static_cast< int >( band );
			static const double colours[ 6 ][ 3 ] = { { 0.55, 0.12, 0.10 }, { 0.50, 0.35, 0.05 }, { 0.12, 0.38, 0.12 },
				                                      { 0.08, 0.30, 0.42 }, { 0.15, 0.12, 0.45 }, { 0.40, 0.10, 0.35 } };
			double r = colours[ stripe ][ 0 ], g = colours[ stripe ][ 1 ], b = colours[ stripe ][ 2 ];

			//A grid of small dots.
			const double gx = std::fmod( u * 14.0, 1.0 ) - 0.5, gy = std::fmod( v * 14.0, 1.0 ) - 0.5;
			if( gx * gx + gy * gy < 0.03 )
				r = g = b = 0.8;

			//The bright shape: a ring and a bar through it.
			const double cx = u - 0.5 * aspect, cy = v - 0.5;
			const double rr = std::sqrt( cx * cx + cy * cy );
			if( std::fabs( rr - 0.22 ) < 0.045 || ( std::fabs( cy ) < 0.04 && std::fabs( cx ) < 0.30 ) )
				r = g = b = 0.97;

			const size_t o = ( static_cast< size_t >( y ) * width + x ) * 4;
			card[ o + 0 ]  = static_cast< float >( r );
			card[ o + 1 ]  = static_cast< float >( g );
			card[ o + 2 ]  = static_cast< float >( b );
			card[ o + 3 ]  = 1.0f;
		}
	return card;
}

/// A coordinate card: R is the frame's u and G its v, exactly, at pixel
/// centres; B is 0. Sampled bilinearly anywhere, it returns the coordinates of
/// the point sampled -- so the output IS the displacement map.
Floats coordinateCard( int width, int height )
{
	Floats card( static_cast< size_t >( width ) * height * 4 );
	for( int y = 0; y < height; ++y )
		for( int x = 0; x < width; ++x )
		{
			const size_t o = ( static_cast< size_t >( y ) * width + x ) * 4;
			card[ o + 0 ]  = static_cast< float >( ( x + 0.5 ) / width );
			card[ o + 1 ]  = static_cast< float >( ( y + 0.5 ) / height );
			card[ o + 2 ]  = 0.0f;
			card[ o + 3 ]  = 1.0f;
		}
	return card;
}

//---------------------------------------------------------------------------
// The audio the harness feeds, written into the Audio buffer's elements the
// way the host writes them.
//---------------------------------------------------------------------------
enum class AudioFeed
{
	Silence,
	Pulses,///< a bass-heavy spectrum with a hit every half second
	Steady ///< the same spectrum, steady: music already playing
};

void feedAudio( BassaltPlugin& plugin, double seconds, AudioFeed feed, int frame )
{
	float strike = 0.0f;
	if( feed == AudioFeed::Pulses )
		strike = static_cast< float >( 0.15 + 1.5 * std::exp( -std::fmod( seconds, 0.5 ) / 0.06 ) );
	else if( feed == AudioFeed::Steady )
		strike = 0.4f;
	( void )frame;
	for( int bin = 0; bin < audio::kBins; ++bin )
	{
		const float across = static_cast< float >( bin ) / static_cast< float >( audio::kBins - 1 );
		const float shape  = 0.7f * ( 1.0f - across ) * ( 1.0f - across ) + 0.2f * ( 0.5f + 0.5f * std::sin( 25.0f * across ) );
		plugin.SetParamElementValue( PT_AUDIO, static_cast< unsigned int >( bin ), shape * strike );
	}
}

//---------------------------------------------------------------------------
// A rig: the real plugin, a float input texture and a float output
// framebuffer, at one size, on a synthetic 60 fps clock.
//---------------------------------------------------------------------------
struct Rig
{
	BassaltPlugin plugin;
	int width = 0, height = 0;
	GLuint sourceTexture = 0, outputTexture = 0, outputFBO = 0;
	int frame            = 0;
	double fps           = 60.0;
	AudioFeed feed       = AudioFeed::Silence;
	std::function< void( int ) > beforeFrame;

	ProcessOpenGLStruct process    = {};
	FFGLTextureStruct inputStruct  = {};
	FFGLTextureStruct* inputs[ 1 ] = { nullptr };

	~Rig()
	{
		plugin.DeInitGL();
		if( outputFBO )
			glDeleteFramebuffers( 1, &outputFBO );
		if( outputTexture )
			glDeleteTextures( 1, &outputTexture );
		if( sourceTexture )
			glDeleteTextures( 1, &sourceTexture );
	}

	bool Init( int w, int h, const Floats* picture = nullptr )
	{
		width  = w;
		height = h;

		FFGLViewportStruct viewport = {};
		viewport.width              = static_cast< FFUInt32 >( width );
		viewport.height             = static_cast< FFUInt32 >( height );
		if( plugin.InitGL( &viewport ) != FF_SUCCESS )
		{
			std::fprintf( stderr, "InitGL failed -- see the diagnostics log for which shader\n" );
			return false;
		}
		plugin.SetClockScaleForTest( 1.0 );

		const Floats card = picture ? *picture : buildCard( width, height );
		sourceTexture     = makeTexture( width, height, card.data() );
		outputTexture     = makeTexture( width, height, nullptr );
		glGenFramebuffers( 1, &outputFBO );
		glBindFramebuffer( GL_FRAMEBUFFER, outputFBO );
		glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputTexture, 0 );
		if( glCheckFramebufferStatus( GL_FRAMEBUFFER ) != GL_FRAMEBUFFER_COMPLETE )
		{
			std::fprintf( stderr, "the harness's own output framebuffer is not complete\n" );
			return false;
		}

		inputStruct.Width = inputStruct.HardwareWidth = static_cast< FFUInt32 >( width );
		inputStruct.Height = inputStruct.HardwareHeight = static_cast< FFUInt32 >( height );
		inputStruct.Handle                              = sourceTexture;
		inputs[ 0 ]                                     = &inputStruct;

		process.numInputTextures = 1;
		process.inputTextures    = inputs;
		process.HostFBO          = outputFBO;
		return true;
	}

	void Upload( const Floats& picture )
	{
		glBindTexture( GL_TEXTURE_2D, sourceTexture );
		glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, picture.data() );
		glBindTexture( GL_TEXTURE_2D, 0 );
	}

	void Set( unsigned int id, float value )
	{
		plugin.SetFloatParameter( id, value );
	}

	/// An event parameter, pressed and released, as a host sends one.
	void Press( unsigned int id )
	{
		plugin.SetFloatParameter( id, 1.0f );
		plugin.SetFloatParameter( id, 0.0f );
	}

	bool Render( int frames = 1 )
	{
		for( int i = 0; i < frames; ++i )
		{
			if( beforeFrame )
				beforeFrame( frame );
			const double seconds = static_cast< double >( frame ) / fps;
			plugin.SetTime( seconds );
			feedAudio( plugin, seconds, feed, frame );
			++frame;

			glBindFramebuffer( GL_FRAMEBUFFER, outputFBO );
			glViewport( 0, 0, width, height );
			glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
			glClear( GL_COLOR_BUFFER_BIT );

			if( plugin.ProcessOpenGL( &process ) != FF_SUCCESS )
			{
				std::fprintf( stderr, "ProcessOpenGL failed\n" );
				return false;
			}
		}
		return true;
	}

	/// The output, RGBA floats, row 0 at the bottom.
	Floats Output() const
	{
		Floats pixels( static_cast< size_t >( width ) * height * 4 );
		glBindFramebuffer( GL_FRAMEBUFFER, outputFBO );
		glPixelStorei( GL_PACK_ALIGNMENT, 1 );
		glReadPixels( 0, 0, width, height, GL_RGBA, GL_FLOAT, pixels.data() );
		return pixels;
	}

	static Floats ReadTexture( GLuint texture, int w, int h, int channels )
	{
		Floats data( static_cast< size_t >( w ) * h * channels );
		if( texture == 0 )
			return data;
		glBindTexture( GL_TEXTURE_2D, texture );
		glPixelStorei( GL_PACK_ALIGNMENT, 1 );
		glGetTexImage( GL_TEXTURE_2D, 0, channels == 4 ? GL_RGBA : GL_RED, GL_FLOAT, data.data() );
		glBindTexture( GL_TEXTURE_2D, 0 );
		return data;
	}

	/// (phi, T, U, V) per cell, row 0 first.
	Floats State() const
	{
		const ph::Grid& g = plugin.CurrentGrid();
		return ReadTexture( plugin.StateTextureID(), g.nx, g.ny, 4 );
	}
	Floats Props() const
	{
		const ph::Grid& g = plugin.CurrentGrid();
		return ReadTexture( plugin.PropsTextureID(), g.nx, g.ny, 4 );
	}
	/// psi per finest node.
	Floats Psi() const
	{
		const ph::Grid& g = plugin.CurrentGrid();
		return ReadTexture( plugin.PsiTextureID(), g.nx + 1, g.ny + 1, 1 );
	}
	Floats Coef() const
	{
		const ph::Grid& g = plugin.CurrentGrid();
		return ReadTexture( plugin.CoefTextureID(), g.nx + 1, g.ny + 1, 4 );
	}
	Floats Rhs() const
	{
		const ph::Grid& g = plugin.CurrentGrid();
		return ReadTexture( plugin.RhsTextureID(), g.nx + 1, g.ny + 1, 1 );
	}
	Floats Thickness() const
	{
		const ph::Grid& g = plugin.CurrentGrid();
		return ReadTexture( plugin.ThicknessTextureID(), g.nx + 1, g.ny + 1, 1 );
	}
};

//---------------------------------------------------------------------------
// Parameters in physical units: the inverses of Controls.cpp, so a check can
// say "40 C" rather than "0.3333".
//---------------------------------------------------------------------------
float geometricParam( double value, double low, double high )
{
	return static_cast< float >( std::log( value / low ) / std::log( high / low ) );
}
float linearParam( double value, double low, double high )
{
	return static_cast< float >( ( value - low ) / ( high - low ) );
}
float heightParam( double metres ) { return geometricParam( metres, 0.1, 1.0 ); }
float gapParam( double metres ) { return geometricParam( metres, 0.001, 0.020 ); }
float speedParam( double speed ) { return ParamFromSpeed( static_cast< float >( speed ) ); }
float ambientParam( double c ) { return linearParam( c, 10.0, 40.0 ); }
float bulbParam( double watts ) { return linearParam( watts, 0.0, 60.0 ); }
float lagParam( double seconds ) { return geometricParam( seconds, 0.5, 120.0 ); }
float saltParam( double percent ) { return linearParam( percent, 0.0, 4.0 ); }
float tensionParam( double sigma ) { return static_cast< float >( std::sqrt( sigma / 0.01 ) ); }
float viscosityParam( double mu ) { return geometricParam( mu, 0.001, 1.0 ); }
float meltParam( double c ) { return linearParam( c, 35.0, 70.0 ); }
float detailParam( int cells )
{
	for( int i = 0; i < kDetailCount; ++i )
		if( kDetailCells[ i ] == cells )
			return static_cast< float >( i );
	return 2.0f;
}

/// The ambient the Ambient control will actually deliver for a request, as the
/// plugin computes it (float), so a check can set a temperature exactly.
double deliveredAmbient( float param )
{
	return static_cast< double >( AmbientFromParam( param ) );
}

//---------------------------------------------------------------------------
// Fields.
//---------------------------------------------------------------------------
struct Field
{
	int nx = 0, ny = 0;
	Floats data;///< RGBA per cell

	float& at( int i, int j, int c )
	{
		return data[ ( static_cast< size_t >( j ) * nx + i ) * 4 + c ];
	}
	float at( int i, int j, int c ) const
	{
		return data[ ( static_cast< size_t >( j ) * nx + i ) * 4 + c ];
	}
};

/// A lamp state built on the CPU: phi from a signed distance (positive in the
/// wax) through the equilibrium profile, T given, dye at rest.
Field makeState( const ph::Grid& g, const std::function< double( double, double ) >& distance,
                 const std::function< double( double, double ) >& temperature )
{
	Field f;
	f.nx = g.nx;
	f.ny = g.ny;
	f.data.assign( static_cast< size_t >( g.nx ) * g.ny * 4, 0.0f );
	const double xi = ph::InterfaceWidth( g );
	for( int j = 0; j < g.ny; ++j )
		for( int i = 0; i < g.nx; ++i )
		{
			const double x = ( i + 0.5 ) * g.dx, y = ( j + 0.5 ) * g.dy;
			const double d = distance( x, y );
			f.at( i, j, 0 ) = static_cast< float >( 1.0 / ( 1.0 + std::exp( -d / xi ) ) );
			f.at( i, j, 1 ) = static_cast< float >( temperature( x, y ) );
			f.at( i, j, 2 ) = static_cast< float >( ( i + 0.5 ) / g.nx );
			f.at( i, j, 3 ) = static_cast< float >( ( j + 0.5 ) / g.ny );
		}
	return f;
}

/// Face speeds from psi at the nodes: u on vertical faces, v on horizontal.
struct Velocity
{
	int nx = 0, ny = 0;
	Doubles u;///< (nx + 1) x ny: the face left of cell i is u[ j * (nx+1) + i ]
	Doubles v;///< nx x (ny + 1)
	/// Infinity if any face is not finite: see maxAbs.
	double MaxSpeed() const
	{
		double m = 0.0;
		for( const Doubles* faces : { &u, &v } )
			for( double a : *faces )
			{
				if( !std::isfinite( a ) )
					return std::numeric_limits< double >::infinity();
				m = std::max( m, std::fabs( a ) );
			}
		return m;
	}
};

Velocity velocityFrom( const Floats& psi, const ph::Grid& g )
{
	Velocity vel;
	vel.nx = g.nx;
	vel.ny = g.ny;
	vel.u.assign( static_cast< size_t >( g.nx + 1 ) * g.ny, 0.0 );
	vel.v.assign( static_cast< size_t >( g.nx ) * ( g.ny + 1 ), 0.0 );
	auto P = [ & ]( int I, int J ) { return static_cast< double >( psi[ static_cast< size_t >( J ) * ( g.nx + 1 ) + I ] ); };
	for( int j = 0; j < g.ny; ++j )
		for( int I = 0; I <= g.nx; ++I )
			vel.u[ static_cast< size_t >( j ) * ( g.nx + 1 ) + I ] = ( P( I, j + 1 ) - P( I, j ) ) / g.dy;
	for( int J = 0; J <= g.ny; ++J )
		for( int i = 0; i < g.nx; ++i )
			vel.v[ static_cast< size_t >( J ) * g.nx + i ] = -( P( i + 1, J ) - P( i, J ) ) / g.dx;
	return vel;
}

/// Wax-weighted mean velocity of the blob: sum phi u / sum phi, from the
/// face speeds averaged onto the cell.
void blobVelocity( const Field& s, const Velocity& vel, double& U, double& V )
{
	double wu = 0.0, wv = 0.0, w = 0.0;
	for( int j = 0; j < s.ny; ++j )
		for( int i = 0; i < s.nx; ++i )
		{
			const double phi = std::clamp( static_cast< double >( s.at( i, j, 0 ) ), 0.0, 1.0 );
			const double uc  = 0.5 * ( vel.u[ static_cast< size_t >( j ) * ( s.nx + 1 ) + i ] + vel.u[ static_cast< size_t >( j ) * ( s.nx + 1 ) + i + 1 ] );
			const double vc  = 0.5 * ( vel.v[ static_cast< size_t >( j ) * s.nx + i ] + vel.v[ static_cast< size_t >( j + 1 ) * s.nx + i ] );
			wu += phi * uc;
			wv += phi * vc;
			w += phi;
		}
	U = w > 0.0 ? wu / w : 0.0;
	V = w > 0.0 ? wv / w : 0.0;
}

double sumChannel( const Field& s, int c )
{
	double total = 0.0;
	for( int j = 0; j < s.ny; ++j )
		for( int i = 0; i < s.nx; ++i )
			total += s.at( i, j, c );
	return total;
}

Field readField( const Rig& rig )
{
	Field f;
	f.nx   = rig.plugin.CurrentGrid().nx;
	f.ny   = rig.plugin.CurrentGrid().ny;
	f.data = rig.State();
	return f;
}

/// A physics rig: small raster (the grid is in metres, the raster only
/// decides how it would be looked at), bulb off, flow on, one frame rendered
/// so the buffers exist, the lamp warm and level. Checks load their own state.
bool physicsRig( Rig& rig, int cells, double height, int width = 320, int heightPx = 180 )
{
	if( !rig.Init( width, heightPx ) )
		return false;
	rig.Set( PT_DETAIL, detailParam( cells ) );
	rig.Set( PT_LAMP_HEIGHT, heightParam( height ) );
	rig.Set( PT_BULB, 0.0f );
	rig.Set( PT_SPEED, speedParam( 1.0 ) );
	rig.Set( PT_GLOW, 0.0f );
	return rig.Render( 1 );
}

//===========================================================================
// Deliberate errors, for --negative. Each field, when set, makes one check
// score the plugin against a model that is wrong by an amount the check is
// supposed to be able to see -- or, where the field says so, runs the plugin
// with one line of its GLSL replaced by the wrong model's.
//===========================================================================
struct Perturb
{
	bool stillTilt         = false;///< --still: the slab one cell out of level across the lamp
	bool frozenInterface   = false;///< --still: Cahn-Hilliard's mobility zeroed in the shader
	bool nonConservative   = false;///< --volume: each cell uses its own outflow speed on both faces
	bool unlimited         = false;///< --volume: the heat limiter replaced by a plain average
	double expansionSign   = 1.0;  ///< --crossover: -1 expects the wax to contract on heating
	bool noDepolarisation  = false;///< --darcy: expects U = K_out drho g, no depolarisation
	bool noCoarseCorrection = false;///< --multigrid: the coarse-grid correction dropped in the shader
	double diffusionFactor = 2.0;  ///< --diffusion: sigma^2 grows as this times kappa t
	bool noTension         = false;///< --rt: expects the growth law without surface tension
	double faces           = 2.0;  ///< --heat: glass faces losing heat
	double lagFraction     = 0.0;  ///< --bulb: nonzero expects this fraction at tau
	bool unprimed          = false;///< --bulb: the onset detector not primed
	double index           = 0.0;  ///< --glass: nonzero expects this water index
};

/// Replace one exact substring of a shipped shader, which must occur exactly
/// once, for this process. Returns false if it does not: a mutation that
/// silently matched nothing would make a negative control pass for nothing.
bool overrideShader( ShaderId id, const std::string& from, const std::string& to )
{
	std::string text        = ShippedSource( id );
	const size_t at         = text.find( from );
	if( at == std::string::npos || text.find( from, at + 1 ) != std::string::npos )
	{
		std::printf( "  FAIL  the %s shader does not contain '%s' exactly once\n", ShaderName( id ), from.c_str() );
		return false;
	}
	text.replace( at, from.size(), to );
	SetShaderOverride( id, text );
	return true;
}

/// Put every shader back as shipped when a check that overrode one returns.
struct ShippedShaders
{
	~ShippedShaders()
	{
		ClearShaderOverrides();
	}
};

using CheckFn = int ( * )( const Perturb& );
std::map< std::string, CheckFn >& CheckTable()
{
	static std::map< std::string, CheckFn > table;
	return table;
}
bool CheckNamed( const std::string& name )
{
	return CheckTable().count( name ) > 0;
}
int RunNamed( const std::string& name )
{
	return CheckTable()[ name ]( Perturb {} );
}
struct Registrar
{
	Registrar( const char* name, CheckFn fn )
	{
		CheckTable()[ name ] = fn;
	}
};

//===========================================================================
// The CPU's own Darcy system, in double, written from the equations in
// AGENTS.md and NOT from the shaders -- so that the GPU and the CPU agreeing
// means something.
//===========================================================================
struct System
{
	int nx = 0, ny = 0;///< cells; nodes are (nx+1) x (ny+1)
	Doubles aE, aN, s, rhs;

	size_t N( int I, int J ) const
	{
		return static_cast< size_t >( J ) * ( nx + 1 ) + I;
	}
	bool Wall( int I, int J ) const
	{
		return I <= 0 || J <= 0 || I >= nx || J >= ny;
	}

	/// (A x) at every interior node; walls hold 0.
	void Apply( const Doubles& x, Doubles& y ) const
	{
		y.assign( x.size(), 0.0 );
		for( int J = 1; J < ny; ++J )
			for( int I = 1; I < nx; ++I )
			{
				const size_t n = N( I, J );
				auto at        = [ & ]( int i, int j ) { return Wall( i, j ) ? 0.0 : x[ N( i, j ) ]; };
				const double p = x[ n ];
				y[ n ] = aE[ n ] * ( p - at( I + 1, J ) ) + aE[ N( I - 1, J ) ] * ( p - at( I - 1, J ) )
				         + aN[ n ] * ( p - at( I, J + 1 ) ) + aN[ N( I, J - 1 ) ] * ( p - at( I, J - 1 ) ) + s[ n ] * p;
			}
	}

	double Diagonal( int I, int J ) const
	{
		const size_t n = N( I, J );
		return aE[ n ] + aE[ N( I - 1, J ) ] + aN[ n ] + aN[ N( I, J - 1 ) ] + s[ n ];
	}

	/// rhs - A x, interior only.
	Doubles Residual( const Doubles& x ) const
	{
		Doubles ax;
		Apply( x, ax );
		Doubles r( x.size(), 0.0 );
		for( int J = 1; J < ny; ++J )
			for( int I = 1; I < nx; ++I )
				r[ N( I, J ) ] = rhs[ N( I, J ) ] - ax[ N( I, J ) ];
		return r;
	}

	double Dot( const Doubles& a, const Doubles& b ) const
	{
		double total = 0.0;
		for( int J = 1; J < ny; ++J )
			for( int I = 1; I < nx; ++I )
				total += a[ N( I, J ) ] * b[ N( I, J ) ];
		return total;
	}

	/// Preconditioned conjugate gradients (Jacobi), to a relative residual of
	/// `tolerance` in the preconditioned norm. Returns the iterations taken.
	int Solve( Doubles& x, const Doubles& b, double tolerance, int maxIterations ) const
	{
		const size_t count = static_cast< size_t >( nx + 1 ) * ( ny + 1 );
		x.assign( count, 0.0 );
		Doubles r = b, z( count, 0.0 ), p( count, 0.0 ), q;
		for( int J = 1; J < ny; ++J )
			for( int I = 1; I < nx; ++I )
				z[ N( I, J ) ] = r[ N( I, J ) ] / Diagonal( I, J );
		p               = z;
		double rz       = Dot( r, z );
		const double r0 = std::sqrt( std::max( Dot( b, b ), 1e-300 ) );
		int it          = 0;
		for( ; it < maxIterations; ++it )
		{
			Apply( p, q );
			const double alpha = rz / Dot( p, q );
			for( int J = 1; J < ny; ++J )
				for( int I = 1; I < nx; ++I )
				{
					const size_t n = N( I, J );
					x[ n ] += alpha * p[ n ];
					r[ n ] -= alpha * q[ n ];
				}
			if( std::sqrt( Dot( r, r ) ) < tolerance * r0 )
				break;
			for( int J = 1; J < ny; ++J )
				for( int I = 1; I < nx; ++I )
					z[ N( I, J ) ] = r[ N( I, J ) ] / Diagonal( I, J );
			const double rzNew = Dot( r, z );
			const double beta  = rzNew / rz;
			rz                 = rzNew;
			for( int J = 1; J < ny; ++J )
				for( int I = 1; I < nx; ++I )
					p[ N( I, J ) ] = z[ N( I, J ) ] + beta * p[ N( I, J ) ];
		}
		return it + 1;
	}

	/// sqrt( x^T A x ): the energy norm, which for psi is the viscous
	/// dissipation of the flow it describes.
	double Energy( const Doubles& x ) const
	{
		Doubles ax;
		Apply( x, ax );
		return std::sqrt( std::max( Dot( x, ax ), 0.0 ) );
	}
};

/// The system as the GPU assembled it: its own coefficient and right-hand
/// side textures, read back.
System gpuSystem( const Rig& rig )
{
	const ph::Grid& g = rig.plugin.CurrentGrid();
	System sys;
	sys.nx = g.nx;
	sys.ny = g.ny;
	const Floats coef = rig.Coef();
	const Floats rhs  = rig.Rhs();
	const size_t n    = static_cast< size_t >( g.nx + 1 ) * ( g.ny + 1 );
	sys.aE.resize( n );
	sys.aN.resize( n );
	sys.s.resize( n );
	sys.rhs.resize( n );
	for( size_t k = 0; k < n; ++k )
	{
		sys.aE[ k ]  = coef[ k * 4 + 0 ];
		sys.aN[ k ]  = coef[ k * 4 + 1 ];
		sys.s[ k ]   = coef[ k * 4 + 2 ];
		sys.rhs[ k ] = rhs[ k ];
	}
	return sys;
}

/// The same system built on the CPU from the state, in double, from the
/// equations: Cahn-Hilliard potential, the density and viscosity laws
/// (Physics.cpp), face resistivities, the circulation of the body force.
System cpuSystem( const Field& s, const ph::Grid& g, const ph::Lamp& lamp )
{
	System sys;
	sys.nx = g.nx;
	sys.ny = g.ny;
	const size_t n = static_cast< size_t >( g.nx + 1 ) * ( g.ny + 1 );
	sys.aE.assign( n, 0.0 );
	sys.aN.assign( n, 0.0 );
	sys.s.assign( n, 0.0 );
	sys.rhs.assign( n, 0.0 );

	auto clampCell = [ & ]( int& i, int& j ) {
		i = std::clamp( i, 0, g.nx - 1 );
		j = std::clamp( j, 0, g.ny - 1 );
	};
	auto phi = [ & ]( int i, int j ) {
		clampCell( i, j );
		return static_cast< double >( s.at( i, j, 0 ) );
	};
	auto temp = [ & ]( int i, int j ) {
		clampCell( i, j );
		return static_cast< double >( s.at( i, j, 1 ) );
	};
	const double xi = ph::InterfaceWidth( g );
	auto potential  = [ & ]( int i, int j ) {
		const double p = phi( i, j );
		const double lap = ( phi( i + 1, j ) - 2.0 * p + phi( i - 1, j ) ) / ( g.dx * g.dx )
		                   + ( phi( i, j + 1 ) - 2.0 * p + phi( i, j - 1 ) ) / ( g.dy * g.dy );
		return 2.0 * p * ( 1.0 - p ) * ( 1.0 - 2.0 * p ) - 2.0 * xi * xi * lap;
	};
	auto rho    = [ & ]( int i, int j ) { return ph::DensityAnomaly( std::clamp( phi( i, j ), 0.0, 1.0 ), temp( i, j ), lamp.salt ); };
	auto resist = [ & ]( int i, int j ) { return ph::Resistivity( phi( i, j ), temp( i, j ), lamp ); };
	const double beta = ph::ChemicalScale( g, lamp.tension );

	auto forceX = [ & ]( int I, int j ) {
		return -0.5 * ( phi( I - 1, j ) + phi( I, j ) ) * beta * ( potential( I, j ) - potential( I - 1, j ) ) / g.dx;
	};
	auto forceY = [ & ]( int i, int J ) {
		return -ph::kGravity * 0.5 * ( rho( i, J - 1 ) + rho( i, J ) )
		       - 0.5 * ( phi( i, J - 1 ) + phi( i, J ) ) * beta * ( potential( i, J ) - potential( i, J - 1 ) ) / g.dy;
	};

	for( int J = 0; J <= g.ny; ++J )
		for( int I = 0; I <= g.nx; ++I )
		{
			const size_t k = sys.N( I, J );
			sys.aE[ k ]    = ( g.dy / g.dx ) * 0.5 * ( resist( I, J - 1 ) + resist( I, J ) );
			sys.aN[ k ]    = ( g.dx / g.dy ) * 0.5 * ( resist( I - 1, J ) + resist( I, J ) );
			if( !sys.Wall( I, J ) )
				sys.rhs[ k ] = g.dx * forceX( I, J - 1 ) + g.dy * forceY( I, J ) - g.dx * forceX( I, J ) - g.dy * forceY( I - 1, J );
		}
	return sys;
}

Doubles toDoubles( const Floats& f )
{
	return Doubles( f.begin(), f.end() );
}

/// The Cauchy-Schwarz bound on how far a linear functional of the face speeds
/// can be from the exact solve's, given psi's residual: the functional is
/// sum_f w_f u_f, the error's energy is sqrt( r^T A^-1 r ), and
/// |sum w e| <= sqrt( sum w^2 / a ) * ||e||_A over the faces. Evaluated for the
/// wax-weighted vertical velocity.
double functionalBound( const System& sys, const Field& s, const ph::Grid& g, const Doubles& residual )
{
	//||e||_A exactly: solve A z = r and take sqrt( r . z ).
	Doubles z;
	sys.Solve( z, residual, 1e-10, 20000 );
	const double energy = std::sqrt( std::max( sys.Dot( residual, z ), 0.0 ) );

	//The functional's weights on the horizontal faces (v): each cell gives
	//half its phi to its top and bottom face, over the total. A horizontal face
	//is the node edge E, whose coefficient is aE and whose "u" in the energy is
	//(psi difference) / dx... the energy term for that edge is aE (dpsi)^2 =
	//aE dx^2 v^2, so sum w^2 / ( aE dx^2 ).
	double total = 0.0;
	for( int j = 0; j < s.ny; ++j )
		for( int i = 0; i < s.nx; ++i )
			total += std::clamp( static_cast< double >( s.at( i, j, 0 ) ), 0.0, 1.0 );
	double weights = 0.0;
	for( int J = 1; J < g.ny; ++J )
		for( int i = 0; i < g.nx; ++i )
		{
			const double w = 0.5 * ( std::clamp( static_cast< double >( s.at( i, J - 1, 0 ) ), 0.0, 1.0 )
			                         + std::clamp( static_cast< double >( s.at( i, J, 0 ) ), 0.0, 1.0 ) ) / total;
			weights += w * w / ( sys.aE[ sys.N( i, J ) ] * g.dx * g.dx );
		}
	return std::sqrt( weights ) * energy;
}


//===========================================================================
// Shared set-ups.
//===========================================================================

/// --dump DIR: checks that call dumpViews() write what the lamp looks like
/// there, as each view. A development aid; nothing is checked by it.
std::string g_dumpDir;

void dumpViews( Rig& rig, const std::string& name )
{
	if( g_dumpDir.empty() )
		return;
	const float view  = rig.plugin.GetFloatParameter( PT_VIEW );
	const float speed = rig.plugin.GetFloatParameter( PT_SPEED );
	const char* names[] = { "lamp", "temperature", "wax", "velocity", "density" };
	//Speed at its floor and one frame: the dump must not move the lamp far.
	const double before = rig.plugin.SimTime();
	for( int v = 0; v < 5; ++v )
	{
		rig.Set( PT_VIEW, static_cast< float >( v ) );
		rig.Set( PT_SPEED, 0.0f );
		rig.Render( 1 );
		writePng( g_dumpDir + "/" + name + "-" + names[ v ] + ".png", rig.width, rig.height, rig.Output() );
	}
	rig.Set( PT_VIEW, view );
	rig.Set( PT_SPEED, speed );
	std::printf( "  (dumped %s; the dump ran the lamp %.3f s)\n", name.c_str(), rig.plugin.SimTime() - before );
}

/// The salt that puts T* at `target`, as the Salt control will actually
/// deliver it (a float parameter, then SaltFromParam).
float saltForCrossover( double target, double& delivered )
{
	double lo = 0.0, hi = 4.0;
	for( int i = 0; i < 200; ++i )
	{
		const double mid = 0.5 * ( lo + hi );
		( ph::Crossover( mid ) > target ? lo : hi ) = mid;
	}
	const float param = saltParam( 0.5 * ( lo + hi ) );
	delivered         = static_cast< double >( SaltFromParam( param ) );
	return param;
}

/// Load a state and make sure the plugin sees it on its next frame.
void load( Rig& rig, const Field& f )
{
	rig.plugin.LoadStateForTest( f.data );
}

/// The largest magnitude -- and infinity for a NaN, which std::max would
/// otherwise swallow and report as a perfectly still lamp. (It did.)
double maxAbs( const Floats& v )
{
	double m = 0.0;
	for( float a : v )
	{
		if( !std::isfinite( a ) )
			return std::numeric_limits< double >::infinity();
		m = std::max( m, static_cast< double >( std::fabs( a ) ) );
	}
	return m;
}

//===========================================================================
// --still
//===========================================================================
int runStill( const Perturb& perturb )
{
	std::printf( "\n=== still: a level lamp at rest stays at rest\n" );
	ShippedShaders restoreShaders;
	if( perturb.frozenInterface
	    && !overrideShader( ShaderId::Update, "float mobility = Mobility * max(", "float mobility = 0.0 * max(" ) )
		return 1;

	//-------------------------------------------------------------------
	// A: a cold slab, bulb off, everything at the room's temperature.
	//-------------------------------------------------------------------
	{
		Rig rig;
		if( !physicsRig( rig, 128, 0.3 ) )
			return 1;
		const float ambientP = ambientParam( 22.0 );
		rig.Set( PT_AMBIENT, ambientP );
		rig.Set( PT_SPEED, speedParam( 300.0 ) );
		rig.Render( 1 );
		const ph::Grid g  = rig.plugin.CurrentGrid();
		const double Ta   = deliveredAmbient( ambientP );
		const double W    = g.nx * g.dx;
		const double slab = 0.15 * g.ny * g.dy;
		const Field start = makeState(
			g,
			[ & ]( double x, double y ) {
				//Tilted: one cell of rise across the whole lamp.
				const double tilt = perturb.stillTilt ? g.dy * ( x / W - 0.5 ) : 0.0;
				return slab + tilt - y;
			},
			[ & ]( double, double ) { return Ta; } );
		load( rig, start );

		double worstPsi = 0.0;
		int unevenRows = 0, warmCells = 0;
		if( perturb.stillTilt )
		{
			rig.plugin.KeepStepLog( true );
			for( int f = 0; f < 12; ++f )
			{
				rig.Render( 1 );
				const Field s = readField( rig );
				double tmin = 1e30, tmax = -1e30, pmin = 1e30, pmax = -1e30;
				for( float v : s.data ) { (void)v; }
				for( int j = 0; j < s.ny; ++j ) for( int i = 0; i < s.nx; ++i ) { tmin = std::min( tmin, (double)s.at(i,j,1) ); tmax = std::max( tmax, (double)s.at(i,j,1) ); pmin = std::min( pmin, (double)s.at(i,j,0) ); pmax = std::max( pmax, (double)s.at(i,j,0) ); }
				const auto& lg = rig.plugin.StepLog();
				std::printf( "TILT f%d t=%.2f sub %d |u| %.3e est %.3e dt %.4f T [%.4f %.4f] phi [%.4f %.4f]\n", f, rig.plugin.SimTime(), rig.plugin.LastSubsteps(), velocityFrom( rig.Psi(), g ).MaxSpeed(), lg.back().speed, lg.back().dt, tmin, tmax, pmin, pmax );
			}
		}
		for( int block = 0; block < 6; ++block )
		{
			rig.Render( 20 );
			worstPsi       = std::max( worstPsi, maxAbs( rig.Psi() ) );
			const Field s  = readField( rig );
			for( int j = 0; j < s.ny; ++j )
				for( int i = 1; i < s.nx; ++i )
					if( s.at( i, j, 0 ) != s.at( 0, j, 0 ) || s.at( i, j, 1 ) != s.at( 0, j, 1 ) )
						++unevenRows;
			for( int j = 0; j < s.ny; ++j )
				for( int i = 0; i < s.nx; ++i )
					if( s.at( i, j, 1 ) != static_cast< float >( Ta ) )
						++warmCells;
		}
		//Exact: the circulation of a horizontally uniform field is the
		//difference of two bit-identical floats, so the solve's right-hand
		//side is exactly zero and so is psi; nothing then moves, so nothing
		//ever differs across a row.
		Check( worstPsi == 0.0,
		       fmt( "cold level slab, %.0f lamp-minutes: largest |psi| %.3g (must be exactly 0)", rig.plugin.SimTime() / 60.0, worstPsi ) );
		Check( unevenRows == 0 && warmCells == 0,
		       fmt( "every row identical to the bit, every cell at the room's %.4f C (%d rows differ, %d cells moved)", Ta,
		            unevenRows, warmCells ) );
	}

	//-------------------------------------------------------------------
	// B: a round blob of neutral density. Only surface tension can move it,
	// and at the Cahn-Hilliard equilibrium it balances exactly; the currents
	// of the transient on the way there must be bounded and die away.
	//-------------------------------------------------------------------
	{
		Rig rig;
		if( !physicsRig( rig, 128, 0.3 ) )
			return 1;
		double salt       = 0.0;
		const float saltP = saltForCrossover( 30.0, salt );
		const double Tstar = ph::Crossover( salt );
		const float ambientP = ambientParam( Tstar );
		rig.Set( PT_SALT, saltP );
		rig.Set( PT_AMBIENT, ambientP );
		rig.Set( PT_MELTING_POINT, meltParam( 35.0 ) );
		rig.Set( PT_WAX_VISCOSITY, viscosityParam( 0.002 ) );
		rig.Set( PT_SPEED, speedParam( 300.0 ) );
		rig.Render( 1 );
		const ph::Grid g = rig.plugin.CurrentGrid();
		const ph::Lamp lamp = rig.plugin.CurrentLamp();
		const double Ta  = deliveredAmbient( ambientP );
		const double R   = 0.04;
		const double cx = 0.5 * g.nx * g.dx, cy = 0.5 * g.ny * g.dy;
		load( rig, makeState( g, [ & ]( double x, double y ) { return R - std::hypot( x - cx, y - cy ); },
		                      [ & ]( double, double ) { return Ta; } ) );

		//The scale: the whole Laplace pressure sigma / R driving flow across
		//the blob's own size, at the most permeable phase's K.
		const double K     = lamp.gap * lamp.gap / ( 12.0 * std::min( ph::kLiquidViscosity, lamp.waxViscosity ) );
		const double scale = K * lamp.tension / ( R * R );

		std::vector< double > times, speeds;
		const int marks[] = { 1, 4, 12, 40, 120, 400, 800 };
		int done = 0;
		for( int mark : marks )
		{
			rig.Render( mark - done );
			done = mark;
			times.push_back( rig.plugin.SimTime() );
			speeds.push_back( velocityFrom( rig.Psi(), g ).MaxSpeed() );
		}
		bool falling = true;
		std::string trace;
		for( size_t k = 0; k < speeds.size(); ++k )
		{
			trace += fmt( " %.0fs:%.2e", times[ k ], speeds[ k ] );
			if( k > 0 && speeds[ k ] >= speeds[ k - 1 ] )
				falling = false;
		}
		std::printf( "  neutral blob R = %.0f mm, sigma = %.1f mN/m, T = T* = %.4f C; max |u| (m/s):%s\n", R * 1000.0,
		             lamp.tension * 1000.0, Tstar, trace.c_str() );
		Check( speeds.front() <= scale,
		       fmt( "spurious currents bounded: %.2e m/s against the capillary Darcy speed K sigma / R^2 = %.2e", speeds.front(), scale ) );
		Check( falling && speeds.back() <= 0.1 * speeds.front(),
		       fmt( "and falling at every mark, by %.0fx over the run (at least 10x: the run is %.0f Cahn-Hilliard times xi^2/D)",
		            speeds.front() / std::max( speeds.back(), 1e-30 ),
		            times.back() * ph::InterfaceMobility( g ) / std::pow( ph::InterfaceWidth( g ), 2 ) ) );
	}

	return Verdict();
}
const Registrar kStill( "still", runStill );

//===========================================================================
// --crossover
//===========================================================================
int runCrossover( const Perturb& perturb )
{
	std::printf( "\n=== crossover: wax rises iff it is hotter than T*, and T* is the linear laws'\n" );

	//The model the check expects. Normally Physics.cpp's; the negative control
	//has the wax contracting as it warms.
	auto expectedCrossover = [ & ]( double salt ) {
		const double slope = perturb.expansionSign * ph::kWaxDensity * ph::kWaxExpansion
		                     - ph::LiquidDensity0( salt ) * ph::kLiquidExpansion;
		return ph::kReferenceT + ( ph::kWaxDensity - ph::LiquidDensity0( salt ) ) / slope;
	};
	auto expectedRises = [ & ]( double salt, double T ) {
		//Wax lighter than water rises: rho_w( T ) < rho_l( T ).
		const double rw = ph::kWaxDensity * ( 1.0 - perturb.expansionSign * ph::kWaxExpansion * ( T - ph::kReferenceT ) );
		const double rl = ph::LiquidDensity0( salt ) * ( 1.0 - ph::kLiquidExpansion * ( T - ph::kReferenceT ) );
		return rw < rl;
	};

	Rig rig;
	if( !physicsRig( rig, 128, 0.3 ) )
		return 1;
	rig.Set( PT_TENSION, 0.0f );
	rig.Set( PT_MELTING_POINT, meltParam( 35.0 ) );
	rig.Render( 1 );
	const ph::Grid g = rig.plugin.CurrentGrid();
	const double cx = 0.5 * g.nx * g.dx, cy = 0.5 * g.ny * g.dy;

	//The blob's vertical velocity with the whole lamp at T: one solve,
	//nothing moves.
	auto velocityAt = [ & ]( double T ) {
		const float t = static_cast< float >( T );
		load( rig, makeState( g, [ & ]( double x, double y ) { return 0.03 - std::hypot( x - cx, y - cy ); },
		                      [ & ]( double, double ) { return static_cast< double >( t ); } ) );
		rig.plugin.SolveForTest( 8, true );
		double U = 0.0, V = 0.0;
		blobVelocity( readField( rig ), velocityFrom( rig.Psi(), g ), U, V );
		return V;
	};

	//Tolerance: the bisection stops at 1e-5 K, and the sign of
	//D1 ( T* - T ) changes exactly where float( T ) passes float( T* ):
	//two float ulps of a temperature below 64 C (3.8e-6 each) on top.
	const double tolerance = 1e-5 + 2.0 * 3.8e-6;

	const double targets[] = { 22.0, 30.0, 38.0 };
	std::vector< double > salts, measured;
	for( double target : targets )
	{
		double salt       = 0.0;
		const float saltP = saltForCrossover( target, salt );
		rig.Set( PT_SALT, saltP );
		const double law      = ph::Crossover( salt );
		const double expected = expectedCrossover( salt );

		const double above = velocityAt( law + 0.5 ), below = velocityAt( law - 0.5 );
		Check( ( above > 0.0 ) == expectedRises( salt, law + 0.5 ) && ( below > 0.0 ) == expectedRises( salt, law - 0.5 ),
		       fmt( "salt %.3f%%: at T* + 0.5 K the blob moves %+.3f mm/s, at T* - 0.5 K %+.3f mm/s", salt, above * 1000.0,
		            below * 1000.0 ) );

		double lo = expected - 2.0, hi = expected + 2.0;
		const double vlo = velocityAt( lo ), vhi = velocityAt( hi );
		double found = std::numeric_limits< double >::quiet_NaN();
		if( ( vlo > 0.0 ) != ( vhi > 0.0 ) )
		{
			const bool loRises = vlo > 0.0;
			while( hi - lo > 1e-5 )
			{
				const double mid = 0.5 * ( lo + hi );
				( ( velocityAt( mid ) > 0.0 ) == loRises ? lo : hi ) = mid;
			}
			found = 0.5 * ( lo + hi );
		}
		Check( std::fabs( found - expected ) <= tolerance,
		       fmt( "salt %.3f%%: the blob stops at %.6f C; the linear laws put T* at %.6f C (|diff| %.1e K, bound %.1e)", salt,
		            found, expected, std::fabs( found - expected ), tolerance ) );
		salts.push_back( salt );
		measured.push_back( found );
	}
	std::printf( "  T* moves %.3f K per 1%% of salt measured, %.3f by the laws\n",
	             ( measured.back() - measured.front() ) / ( salts.back() - salts.front() ),
	             ( ph::Crossover( salts.back() ) - ph::Crossover( salts.front() ) ) / ( salts.back() - salts.front() ) );
	return Verdict();
}
const Registrar kCrossover( "crossover", runCrossover );


//===========================================================================
// --multigrid
//===========================================================================

/// Relative residual after each of `cycles` V-cycles from zero, measured in
/// double from the GPU's own psi, coefficients and right-hand side.
std::vector< double > residualHistory( Rig& rig, int cycles, double& floor )
{
	std::vector< double > history;
	rig.plugin.SolveForTest( 0, true );
	const System sys = gpuSystem( rig );
	const double b   = std::sqrt( sys.Dot( sys.rhs, sys.rhs ) );
	for( int c = 0; c < cycles; ++c )
	{
		rig.plugin.SolveForTest( 1, false );
		const Doubles psi = toDoubles( rig.Psi() );
		const Doubles r   = sys.Residual( psi );
		history.push_back( std::sqrt( sys.Dot( r, r ) ) / b );
		if( c == cycles - 1 )
		{
			//The floor: psi is stored in float, so the best any solve can do
			//is psi to half an ulp at every node, and the residual of THAT is
			//what the stencil makes of half-ulp errors -- sum_n a_n ( ulp_0 +
			//ulp_n ) / 2 at each node.
			double f = 0.0;
			for( int J = 1; J < sys.ny; ++J )
				for( int I = 1; I < sys.nx; ++I )
				{
					auto ulp = [ & ]( int i, int j ) {
						if( sys.Wall( i, j ) )
							return 0.0;
						const float v = static_cast< float >( psi[ sys.N( i, j ) ] );
						return static_cast< double >( std::nextafter( std::fabs( v ), INFINITY ) - std::fabs( v ) );
					};
					const size_t k = sys.N( I, J );
					const double e = 0.5
					                 * ( sys.aE[ k ] * ( ulp( I, J ) + ulp( I + 1, J ) ) + sys.aE[ sys.N( I - 1, J ) ] * ( ulp( I, J ) + ulp( I - 1, J ) )
					                     + sys.aN[ k ] * ( ulp( I, J ) + ulp( I, J + 1 ) ) + sys.aN[ sys.N( I, J - 1 ) ] * ( ulp( I, J ) + ulp( I, J - 1 ) ) );
					f += e * e;
				}
			floor = std::sqrt( f ) / b;
		}
	}
	return history;
}

int runMultigrid( const Perturb& perturb )
{
	std::printf( "\n=== multigrid: the residual falls by the stated factor, and the step gets the flow it needs\n" );
	ShippedShaders restoreShaders;
	if( perturb.noCoarseCorrection
	    && !overrideShader( ShaderId::Prolong, "texelFetch( Psi, n, 0 ).x + Weight * e", "texelFetch( Psi, n, 0 ).x + 0.0 * e" ) )
		return 1;

	auto report = [ & ]( const char* what, const std::vector< double >& h, double floor, double stated ) {
		std::string trace;
		double worst = 0.0;
		for( size_t k = 0; k < h.size(); ++k )
		{
			trace += fmt( " %.1e", h[ k ] );
			//A cycle counts once its starting residual is well clear of the
			//float floor; below that the residual is rounding, not error.
			const double before = k == 0 ? 1.0 : h[ k - 1 ];
			if( before > 30.0 * floor )
				worst = std::max( worst, h[ k ] / before );
		}
		std::printf( "  %s, relative residual per cycle:%s (float floor %.1e)\n", what, trace.c_str(), floor );
		Check( worst <= stated, fmt( "%s: worst factor per cycle %.3f (stated %.2f)", what, worst, stated ) );
	};

	//-------------------------------------------------------------------
	// Constant coefficients: all water, a buoyant temperature pattern. The
	// stated factor is local Fourier analysis's for this cycle -- red-black
	// Gauss-Seidel, one sweep before and one after, full weighting, bilinear
	// prolongation: 0.074 for two grids (Trottenberg, Oosterlee & Schuller,
	// Multigrid, section 4.5) -- plus the same again, because a V-cycle
	// solves each coarse problem only to about that factor itself: 0.15.
	//-------------------------------------------------------------------
	{
		Rig rig;
		if( !physicsRig( rig, 128, 0.3 ) )
			return 1;
		const ph::Grid g = rig.plugin.CurrentGrid();
		const double W = g.nx * g.dx, H = g.ny * g.dy;
		load( rig, makeState( g, []( double, double ) { return -1.0; },
		                      [ & ]( double x, double y ) { return 30.0 + 5.0 * std::sin( 3.0 * kPi * x / W ) * std::sin( 2.0 * kPi * y / H ) + 2.0 * std::cos( 17.0 * x / W + 9.0 * y / H ); } ) );
		double floor = 0.0;
		report( "water only", residualHistory( rig, 8, floor ), floor, 0.15 );
	}

	//-------------------------------------------------------------------
	// The lamp's own coefficients: wax at up to 10^4 times its melt's
	// viscosity against water, after a minute of convection. There is no
	// Fourier analysis for jumps of 10^5; the requirement is the one that
	// matters, below, and this line states what it takes.
	//-------------------------------------------------------------------
	Rig lamp;
	if( !lamp.Init( 320, 180 ) )
		return 1;
	lamp.Set( PT_SPEED, speedParam( 30.0 ) );
	lamp.Render( 120 );
	{
		double floor = 0.0;
		report( "a convecting lamp", residualHistory( lamp, 10, floor ), floor, 0.5 );
	}

	//-------------------------------------------------------------------
	// What the step needs. One warm-started V-cycle per substep must leave
	// psi within 1% of the exact solve, in the energy norm (the flow's own
	// dissipation): a substep moves nothing further than a quarter cell, so
	// a 1% error in the flow is a four-hundredth of a cell per step.
	//-------------------------------------------------------------------
	{
		lamp.plugin.SolveForTest( 0, false );//nothing: just the fields in place
		double worst = 0.0;
		for( int k = 0; k < 5; ++k )
		{
			lamp.Render( 12 );
			const System sys = gpuSystem( lamp );
			Doubles exact;
			sys.Solve( exact, sys.rhs, 1e-11, 40000 );
			const Doubles psi = toDoubles( lamp.Psi() );
			Doubles error( psi.size() );
			for( size_t n = 0; n < psi.size(); ++n )
				error[ n ] = psi[ n ] - exact[ n ];
			worst = std::max( worst, sys.Energy( error ) / sys.Energy( exact ) );
		}
		Check( worst <= 0.01, fmt( "running at 30x, one warm-started cycle per substep: energy error %.2e of the flow (needs 1e-2)", worst ) );
	}
	return Verdict();
}
const Registrar kMultigrid( "multigrid", runMultigrid );


//===========================================================================
// --darcy
//===========================================================================

/// The walls' effect on a blob moving at U in the middle of a W x H box with
/// psi = 0 all round, to leading order in ( R / L )^2. Outside, the blob is a
/// dipole, psi = -U R^2 x / r^2. The walls are its images: the same sign at
/// ( m W, n H ) for the side walls, alternating in n for the top and bottom
/// (psi odd about each wall). Their velocity at the blob, per unit U R^2:
///   sum_{(m,n) != 0} (-1)^n ( n^2 H^2 - m^2 W^2 ) / ( m^2 W^2 + n^2 H^2 )^2
/// (-pi^2/3 / W^2 from the side row alone, -pi^2/6 / H^2 from the column.)
/// Summed row by row, which is the order the images are built in.
double imageSum( double W, double H )
{
	double total = 0.0;
	const int reach = 4000;
	for( int n = -reach; n <= reach; ++n )
	{
		double row = 0.0;
		for( int m = -reach; m <= reach; ++m )
		{
			if( m == 0 && n == 0 )
				continue;
			const double X = m * W, Y = n * H, r2 = X * X + Y * Y;
			row += ( Y * Y - X * X ) / ( r2 * r2 );
		}
		total += ( n % 2 == 0 ? 1.0 : -1.0 ) * row;
	}
	return total;
}

int runDarcy( const Perturb& perturb )
{
	std::printf( "\n=== darcy: a blob moves at the Hele-Shaw speed, and the GPU solves the system the CPU does\n" );

	Rig rig;
	if( !physicsRig( rig, 256, 1.0 ) )
		return 1;
	double salt       = 0.0;
	const float saltP = saltForCrossover( 33.0, salt );
	const float ambP  = ambientParam( 40.0 );
	rig.Set( PT_SALT, saltP );
	rig.Set( PT_AMBIENT, ambP );
	rig.Set( PT_TENSION, 0.0f );
	rig.Set( PT_MELTING_POINT, meltParam( 35.0 ) );
	rig.Render( 1 );
	const ph::Grid g = rig.plugin.CurrentGrid();
	const double W = g.nx * g.dx, H = g.ny * g.dy;
	const double R = 0.08, cx = 0.5 * W, cy = 0.5 * H;
	const double T = deliveredAmbient( ambP );
	const double xi = ph::InterfaceWidth( g );
	const double images = imageSum( W, H ) * R * R;
	std::printf( "  lamp %.3f x %.3f m, %dx%d cells, blob R = %.0f mm (%.0f cells, xi/R = %.3f), T = %.2f C, T* = %.2f C\n", W, H,
	             g.nx, g.ny, R * 1000.0, R / g.dy, xi / R, T, ph::Crossover( salt ) );
	std::printf( "  the walls' images: %.4f of U at the blob (side walls alone %.4f)\n", images, -kPi * kPi / 3.0 * R * R / ( W * W ) );

	const double ratios[] = { 0.5, 1.0, 10.0, 100.0 };
	for( double ratio : ratios )
	{
		const float viscP = viscosityParam( ratio * ph::kLiquidViscosity );
		rig.Set( PT_WAX_VISCOSITY, viscP );
		const Field state = makeState( g, [ & ]( double x, double y ) { return R - std::hypot( x - cx, y - cy ); },
		                               [ & ]( double, double ) { return T; } );
		load( rig, state );
		rig.plugin.SolveForTest( 12, true );
		const ph::Lamp lamp = rig.plugin.CurrentLamp();

		//The system as assembled on the GPU and as written from the equations.
		const System gpu = gpuSystem( rig );
		const System cpu = cpuSystem( state, g, lamp );
		double coefWorst = 0.0, rhsWorst = 0.0, rhsScale = 0.0;
		for( size_t k = 0; k < gpu.rhs.size(); ++k )
			rhsScale = std::max( rhsScale, std::fabs( cpu.rhs[ k ] ) );
		for( int J = 1; J < g.ny; ++J )
			for( int I = 0; I < g.nx; ++I )
			{
				const size_t k = gpu.N( I, J );
				coefWorst = std::max( coefWorst, std::fabs( gpu.aE[ k ] - cpu.aE[ k ] ) / cpu.aE[ k ] );
				if( I > 0 )
					rhsWorst = std::max( rhsWorst, std::fabs( gpu.rhs[ k ] - cpu.rhs[ k ] ) / rhsScale );
			}
		for( int J = 0; J < g.ny; ++J )
			for( int I = 1; I < g.nx; ++I )
				coefWorst = std::max( coefWorst, std::fabs( gpu.aN[ gpu.N( I, J ) ] - cpu.aN[ gpu.N( I, J ) ] ) / cpu.aN[ gpu.N( I, J ) ] );

		//The GPU's solve, and the exact solve of the GPU's own system.
		const Doubles psi = toDoubles( rig.Psi() );
		Doubles exact;
		const int iterations = gpu.Solve( exact, gpu.rhs, 1e-12, 60000 );
		Floats exactF( exact.begin(), exact.end() );
		double Ug = 0.0, Vg = 0.0, Ue = 0.0, Ve = 0.0;
		blobVelocity( state, velocityFrom( rig.Psi(), g ), Ug, Vg );
		blobVelocity( state, velocityFrom( exactF, g ), Ue, Ve );
		const double bound = functionalBound( gpu, state, g, gpu.Residual( psi ) );

		//The analytic speed: the depolarisation result and the walls' images
		//(the blob answers a uniform stream V with 2 K_in / ( K_in + K_out ) V).
		const double kIn  = lamp.gap * lamp.gap / ( 12.0 * lamp.waxViscosity );
		const double kOut = lamp.gap * lamp.gap / ( 12.0 * ph::kLiquidViscosity );
		const double drho = -ph::CrossoverSlope( salt ) * ( ph::Crossover( salt ) - T );//rho_out - rho_in
		double U = perturb.noDepolarisation ? kOut * drho * ph::kGravity : ph::BlobSpeed( kIn, kOut, drho );
		U *= 1.0 + 2.0 * kIn / ( kIn + kOut ) * images;

		//The formula is for a sharp interface; the lamp's is xi = one cell
		//wide, and the error that makes is first order in xi / R. So the same
		//blob is also solved exactly on the CPU at half and a quarter of the
		//resolution (xi two and four times as wide) and extrapolated to xi = 0.
		//With U_h = U0 + a h + b h^2, E1 = 2 U_h - U_2h = U0 - 2 b h^2 and
		//E2 = 2 U_2h - U_4h = U0 - 8 b h^2: E1 is the answer, and its own
		//error, 2 b h^2, is ( E1 - E2 ) / 3 -- measured, not assumed.
		auto cpuSpeed = [ & ]( int cellsUp ) {
			const ph::Grid cg     = ph::ChooseGrid( W, H, cellsUp );
			const Field cs        = makeState( cg, [ & ]( double x, double y ) { return R - std::hypot( x - cx, y - cy ); },
			                                   [ & ]( double, double ) { return T; } );
			const System csys     = cpuSystem( cs, cg, lamp );
			Doubles cpsi;
			csys.Solve( cpsi, csys.rhs, 1e-12, 60000 );
			Floats cpsiF( cpsi.begin(), cpsi.end() );
			double cu = 0.0, cv = 0.0;
			blobVelocity( cs, velocityFrom( cpsiF, cg ), cu, cv );
			return cv;
		};
		const double V2 = cpuSpeed( g.ny / 2 ), V4 = cpuSpeed( g.ny / 4 );
		const double E1 = 2.0 * Ve - V2, E2 = 2.0 * V2 - V4;
		const double V0 = E1;

		//Tolerance: the extrapolation's error is ( E1 - E2 ) / 3 to leading
		//order; the whole of | E1 - E2 | is allowed, three times that, for the
		//orders it does not see (a 100:1 viscosity jump has large ones). And the
		//images' second order: the dipole's own strength answering its images,
		//( g I )^2 with g I the first-order correction above, and the images'
		//next multipole, ( R / L )^4. At R / L = 0.13 the image terms were
		//visible (0.33% at 1:2); the lamp is 1 m tall to keep them small.
		const double gI        = 2.0 * kIn / ( kIn + kOut ) * images;
		const double first     = std::fabs( Ve - V2 ) / std::fabs( U );
		const double tolerance = std::fabs( E1 - E2 ) / std::fabs( U ) + gI * gI + std::pow( R / std::min( W, H ), 4 );

		std::printf( "  mu_wax / mu_water = %g: K_in / K_out = %.3f, drho = %.3f kg/m^3\n", lamp.waxViscosity / ph::kLiquidViscosity,
		             kIn / kOut, drho );
		Check( coefWorst <= 1e-6 && rhsWorst <= 1e-5,
		       fmt( "    the GPU assembled the equations' system: coefficients to %.1e (bound 1e-6: six float operations), "
		            "right-hand side to %.1e of its largest (bound 1e-5: a few ulps of rho', which is ~3x the anomaly it differences)",
		            coefWorst, rhsWorst ) );
		Check( std::fabs( Vg - Ve ) <= bound + 1e-12,
		       fmt( "    GPU %.6f mm/s against the exact solve of its own system %.6f mm/s (%d CG iterations): %.1e, "
		            "within the residual's Cauchy-Schwarz bound %.1e",
		            Vg * 1000.0, Ve * 1000.0, iterations, std::fabs( Vg - Ve ), bound ) );
		Check( std::fabs( V0 - U ) <= tolerance * std::fabs( U ),
		       fmt( "    Hele-Shaw %.5f mm/s; the solve %.5f at %d cells up, %.5f at %d, %.5f at %d, so %.5f at xi = 0: "
		            "off by %.3f%% (allowance %.3f%%; the first-order term was %.2f%%)",
		            U * 1000.0, Ve * 1000.0, g.ny, V2 * 1000.0, g.ny / 2, V4 * 1000.0, g.ny / 4, V0 * 1000.0,
		            100.0 * std::fabs( V0 - U ) / std::fabs( U ), 100.0 * tolerance, 100.0 * first ) );
	}
	std::printf( "  the limits, from the formula: K_in = K_out gives K drho g / 2; K_in -> infinity gives K_out drho g; "
	             "K_in -> 0 gives K_in drho g\n" );
	return Verdict();
}
const Registrar kDarcy( "darcy", runDarcy );


//===========================================================================
// --volume
//===========================================================================
double ulpOf( double value )
{
	const float v = static_cast< float >( std::fabs( value ) );
	return static_cast< double >( std::nextafter( v, INFINITY ) - v );
}

int substepsLogged( const BassaltPlugin& plugin, double interfaceLimit, int& subcycles )
{
	subcycles = 0;
	for( const StepRecord& step : plugin.StepLog() )
		subcycles += std::max( 1, static_cast< int >( std::ceil( step.dt / interfaceLimit ) ) );
	return static_cast< int >( plugin.StepLog().size() );
}

int runVolume( const Perturb& perturb )
{
	std::printf( "\n=== volume: wax is conserved through convection; heat makes no new extrema\n" );
	ShippedShaders restoreShaders;
	if( perturb.nonConservative && !overrideShader( ShaderId::Update, "vec2 left   = fluxX( i - 1, j, uL );", "vec2 left   = fluxX( i - 1, j, uR );" ) )
		return 1;
	if( perturb.unlimited
	    && !overrideShader( ShaderId::Update, "return p > 0.0 ? 2.0 * p / ( back + ahead ) : 0.0;", "return 0.5 * ( back + ahead );" ) )
		return 1;

	Rig rig;
	if( !rig.Init( 320, 180 ) )
		return 1;
	rig.Set( PT_DETAIL, detailParam( 64 ) );
	rig.Set( PT_SPEED, speedParam( 300.0 ) );
	rig.Render( 2 );
	rig.plugin.KeepStepLog( true );
	const ph::Grid g = rig.plugin.CurrentGrid();
	const Field start = readField( rig );
	const double before = sumChannel( start, 0 );
	rig.Render( 150 );
	const Field end = readField( rig );
	const double after = sumChannel( end, 0 );
	int subcycles = 0;
	const int steps = substepsLogged( rig.plugin, ph::InterfaceLimit( g ), subcycles );

	//The bound. Every flux is shared bit for bit by its two cells, so the sum
	//telescopes and only each cell's own arithmetic rounds: phi + dt * change,
	//at most half an ulp of the result plus the product's rounding, together
	//under 2 ulps of 1 (|phi| stays below 2, and a step's change below 1).
	//Per cell per pass, over every cell, over every pass that writes phi (the
	//update and each Cahn-Hilliard subcycle). A worst case: the errors do not
	//all go one way, and the measured drift is far inside it.
	const double perCell = 2.0 * ulpOf( 1.5 );
	const double bound   = perCell * static_cast< double >( g.nx ) * g.ny * ( steps + subcycles );
	Check( std::fabs( after - before ) <= bound,
	       fmt( "%.1f lamp-minutes at up to %.1f mm/s, %d steps and %d Cahn-Hilliard subcycles: sum phi %.6f -> %.6f, "
	            "drift %.2e of it (bound %.2e: 2 ulps a cell a pass)",
	            rig.plugin.SimTime() / 60.0, 1000.0 * velocityFrom( rig.Psi(), g ).MaxSpeed(), steps, subcycles, before, after,
	            std::fabs( after - before ) / before, bound / before ) );

	//No new extrema in the heat. Bulb off and the cap off, so the only source
	//is the glass pulling every cell towards the room, which is itself inside
	//the range: every new T is then a convex combination of old Ts and the
	//room's. What rounding can add per step: the update's own ulp, and T
	//times the rounding of the discrete divergence, four differences of psi
	//each good to an ulp at a Courant number of 1/4 -- T * eps in all. Five
	//ulps of the hottest T a step, as a worst case.
	double tmin = 1e30, tmax = -1e30;
	for( int j = 0; j < end.ny; ++j )
		for( int i = 0; i < end.nx; ++i )
		{
			tmin = std::min( tmin, static_cast< double >( end.at( i, j, 1 ) ) );
			tmax = std::max( tmax, static_cast< double >( end.at( i, j, 1 ) ) );
		}
	const double room = deliveredAmbient( rig.plugin.GetFloatParameter( PT_AMBIENT ) );
	rig.Set( PT_BULB, 0.0f );
	rig.plugin.SetCapForTest( false );
	rig.plugin.KeepStepLog( true );
	double worstHigh = -1e30, worstLow = 1e30;
	for( int block = 0; block < 10; ++block )
	{
		rig.Render( 6 );
		const Field s = readField( rig );
		for( int j = 0; j < s.ny; ++j )
			for( int i = 0; i < s.nx; ++i )
			{
				const double t = s.at( i, j, 1 );
				if( !std::isfinite( t ) )
					worstHigh = 1e30;
				worstHigh = std::max( worstHigh, t - tmax );
				worstLow  = std::min( worstLow, t - std::min( tmin, room ) );
			}
	}
	rig.plugin.SetCapForTest( true );
	const int heatSteps = static_cast< int >( rig.plugin.StepLog().size() );
	const double slack  = 5.0 * ulpOf( tmax ) * heatSteps;
	Check( worstHigh <= slack && worstLow >= -slack,
	       fmt( "bulb and cap off, %d steps of convection: T stayed in [%.4f, %.4f] C to %.2e above and %.2e below "
	            "(allowed %.2e: 5 ulps a step)",
	            heatSteps, std::min( tmin, room ), tmax, std::max( worstHigh, 0.0 ), std::max( -worstLow, 0.0 ), slack ) );
	return Verdict();
}
const Registrar kVolume( "volume", runVolume );

//===========================================================================
// --diffusion
//===========================================================================
int runDiffusion( const Perturb& perturb )
{
	std::printf( "\n=== diffusion: with the flow off, a hot spot spreads as sigma^2 = sigma0^2 + 2 kappa t\n" );
	Rig rig;
	if( !physicsRig( rig, 128, 0.3 ) )
		return 1;
	const float ambientP = ambientParam( 10.0 );
	rig.Set( PT_AMBIENT, ambientP );
	rig.Set( PT_COIL, 0.0f );
	rig.Set( PT_TENSION, 0.0f );
	rig.Set( PT_SPEED, speedParam( 300.0 ) );
	rig.plugin.SetFlowFrozenForTest( true );
	rig.Render( 1 );
	const ph::Grid g  = rig.plugin.CurrentGrid();
	const double Ta   = deliveredAmbient( ambientP );
	const double x0 = 0.5 * g.nx * g.dx, y0 = 0.5 * g.ny * g.dy, s0 = 0.015;
	load( rig, makeState( g, []( double, double ) { return -1.0; },
	                      [ & ]( double x, double y ) {
		                      return Ta + 50.0 * std::exp( -( ( x - x0 ) * ( x - x0 ) + ( y - y0 ) * ( y - y0 ) ) / ( 2.0 * s0 * s0 ) );
	                      } ) );
	const double kappa = ph::kLiquidConductivity / ph::kHeatCapacity;

	struct Moments
	{
		double sx = 0, sy = 0, mass = 0;
		int active = 0;
		double weight = 0;///< sum over cells with any excess of r^2, for the rounding bound
	};
	auto moments = [ & ]( const Field& s ) {
		Moments m;
		double cx = 0, cy = 0;
		for( int j = 0; j < s.ny; ++j )
			for( int i = 0; i < s.nx; ++i )
			{
				const double e = s.at( i, j, 1 ) - static_cast< float >( Ta );
				const double x = ( i + 0.5 ) * g.dx, y = ( j + 0.5 ) * g.dy;
				m.mass += e;
				cx += e * x;
				cy += e * y;
			}
		cx /= m.mass;
		cy /= m.mass;
		for( int j = 0; j < s.ny; ++j )
			for( int i = 0; i < s.nx; ++i )
			{
				const double e = s.at( i, j, 1 ) - static_cast< float >( Ta );
				const double x = ( i + 0.5 ) * g.dx - cx, y = ( j + 0.5 ) * g.dy - cy;
				m.sx += e * x * x;
				m.sy += e * y * y;
				if( e != 0.0 )
				{
					++m.active;
					m.weight += x * x + y * y;
				}
			}
		m.sx /= m.mass;
		m.sy /= m.mass;
		return m;
	};

	rig.plugin.KeepStepLog( true );
	const Moments first = moments( readField( rig ) );
	const double t0     = rig.plugin.SimTime();
	rig.Render( 160 );
	const Moments last  = moments( readField( rig ) );
	const double t      = rig.plugin.SimTime() - t0;
	const int steps     = static_cast< int >( rig.plugin.StepLog().size() );
	rig.plugin.SetFlowFrozenForTest( false );

	//The discrete identity: the five-point Laplacian of x^2 is exactly 2, so
	//on a grid with no walls in reach one explicit step adds exactly
	//2 kappa dt sum E to sum x^2 E. The glass takes r dt of every cell's excess
	//in the SAME step, so the normalised moment grows by 2 kappa dt / (1 - r dt)
	//a step: 0.075% more than 2 kappa dt at these steps, which the continuous
	//law does not see. What remains is float: each cell with any excess rounds
	//by up to 2 ulps a step, which moves sigma^2 by at most
	//2 ulp( T ) sum r^2 / sum E.
	const double rate = ph::FaceLossRate( rig.plugin.CurrentLamp().gap );
	double expected   = 0.0;
	for( const StepRecord& step : rig.plugin.StepLog() )
		expected += perturb.diffusionFactor * kappa * step.dt / ( 1.0 - rate * step.dt );
	const double bound    = steps * 2.0 * ulpOf( Ta + 50.0 ) * last.weight / last.mass;
	for( int axis = 0; axis < 2; ++axis )
	{
		const double grown = ( axis == 0 ? last.sx - first.sx : last.sy - first.sy );
		Check( std::fabs( grown - expected ) <= bound,
		       fmt( "%s: sigma^2 grew %.8e m^2 in %.0f s; the discrete law says %.8e (2 kappa t is %.8e) -- |diff| %.1e, "
		            "float bound %.1e over %d steps",
		            axis == 0 ? "across" : "up", grown, t, expected, 2.0 * kappa * t, std::fabs( grown - expected ), bound, steps ) );
	}
	std::printf( "  sigma0 = %.2f mm, now %.2f mm across and %.2f mm up; kappa = %.3e m^2/s, the water's\n",
	             1000.0 * std::sqrt( first.sx ), 1000.0 * std::sqrt( last.sx ), 1000.0 * std::sqrt( last.sy ), kappa );
	return Verdict();
}
const Registrar kDiffusion( "diffusion", runDiffusion );

//@CHECKS@

//===========================================================================
// --negative
//===========================================================================
int runNegative( const Perturb& )
{
	struct Case
	{
		const char* name;
		CheckFn check;
		Perturb perturb;
		const char* what;
	};
	std::vector< Case > cases;
	auto add = [ & ]( const char* name, const char* what, const std::function< void( Perturb& ) >& set ) {
		Perturb p;
		set( p );
		cases.push_back( { name, CheckTable()[ name ], p, what } );
	};
	add( "still", "the slab one cell out of level across the lamp: not hydrostatic, so it must move", []( Perturb& p ) { p.stillTilt = true; } );
	add( "still", "Cahn-Hilliard's mobility zeroed in the shader: the interface cannot relax, the currents cannot die",
	     []( Perturb& p ) { p.frozenInterface = true; } );
	add( "volume", "each cell advects through its left face at its right face's speed: not conservative",
	     []( Perturb& p ) { p.nonConservative = true; } );
	add( "volume", "the heat's limiter replaced by a plain average: new extrema", []( Perturb& p ) { p.unlimited = true; } );
	add( "crossover", "expect the wax to contract as it warms", []( Perturb& p ) { p.expansionSign = -1.0; } );
	add( "darcy", "expect U = K_out drho g: no depolarisation factor", []( Perturb& p ) { p.noDepolarisation = true; } );
	add( "multigrid", "the coarse-grid correction dropped from the shader", []( Perturb& p ) { p.noCoarseCorrection = true; } );
	add( "diffusion", "expect sigma^2 to grow as 4 kappa t", []( Perturb& p ) { p.diffusionFactor = 4.0; } );
//@NEGATIVES@

	int unfalsifiable = 0;
	for( const Case& c : cases )
	{
		std::printf( "\n=== negative control: %s -- %s\n", c.name, c.what );
		const int before = g_failures;
		g_failures       = 0;
		c.check( c.perturb );
		const int observed = g_failures;
		g_failures         = before;
		if( observed > 0 )
			std::printf( "  ok    %s failed %d check%s, as it must\n", c.name, observed, observed == 1 ? "" : "s" );
		else
		{
			std::printf( "  FAIL  %s PASSED against a wrong model -- it cannot fail, so it is not a check\n", c.name );
			++unfalsifiable;
		}
	}
	std::printf( "\nnegative controls: %zu wrong models, %d of them undetected\n", cases.size(), unfalsifiable );
	std::printf( "\n  %s\n", unfalsifiable == 0 ? "PASS" : "FAIL" );
	return unfalsifiable == 0 ? 0 : 1;
}
const Registrar kNegative( "negative", runNegative );

//===========================================================================
// --bench
//===========================================================================
double timeFrames( Rig& rig, int frames )
{
	glFinish();
	const auto start = std::chrono::steady_clock::now();
	rig.Render( frames );
	glFinish();
	return std::chrono::duration< double, std::milli >( std::chrono::steady_clock::now() - start ).count() / frames;
}

int runBench( const Perturb& )
{
	std::printf( "\n=== bench: ms per frame, warm lamp, after 60 frames of warm-up\n" );
	struct Size
	{
		int w, h;
		const char* name;
		int cells;
		double speed;
		int view;
	};
	const Size sizes[] = {
		{ 1280, 720, "720p", 128, 2.0, 0 },  { 1920, 1080, "1080p", 128, 2.0, 0 }, { 3840, 2160, "4K", 128, 2.0, 0 },
		{ 1920, 1080, "1080p", 64, 2.0, 0 }, { 1920, 1080, "1080p", 256, 2.0, 0 }, { 1920, 1080, "1080p", 128, 30.0, 0 },
		{ 1920, 1080, "1080p", 128, 300.0, 0 }, { 1920, 1080, "1080p", 128, 2.0, 2 },
	};
	for( const Size& size : sizes )
	{
		Rig rig;
		if( !rig.Init( size.w, size.h ) )
			return 1;
		rig.Set( PT_DETAIL, detailParam( size.cells ) );
		rig.Set( PT_SPEED, speedParam( size.speed ) );
		rig.Set( PT_VIEW, static_cast< float >( size.view ) );
		rig.Render( 60 );
		int substeps = 0;
		const double t0 = rig.plugin.SimTime();
		const double ms = timeFrames( rig, 60 );
		const double lamp = ( rig.plugin.SimTime() - t0 ) / 1.0;
		substeps = rig.plugin.LastSubsteps();
		const ph::Grid& g = rig.plugin.CurrentGrid();
		std::printf( "  %-6s Detail %-3d Speed %5.0fx %-5s %6.2f ms/frame (%4.1f%% of 60 fps; grid %dx%d, %d levels, %d substeps, "
		             "lamp ran %.0fx)\n",
		             size.name, size.cells, size.speed, size.view == 0 ? "Lamp" : "Wax", ms, 100.0 * ms / ( 1000.0 / 60.0 ), g.nx, g.ny,
		             g.levels, substeps, lamp );
	}
	return 0;
}

const Registrar kBench( "bench", runBench );

/// --pipe and --film. Raw RGBA, top row first, one frame at a time, on the
/// synthetic 60 fps clock.
int runPipe( int width, int height, const std::string& scriptPath, int filmFrames, bool beat,
             const std::vector< std::string >& settings )
{
	Rig rig;
	if( !rig.Init( width, height ) )
		return 1;
	if( beat )
		rig.feed = AudioFeed::Pulses;

	for( const std::string& setting : settings )
	{
		std::string error;
		if( !applySetting( rig.plugin, setting, error ) )
		{
			std::fprintf( stderr, "--set %s: %s\n", setting.c_str(), error.c_str() );
			return 2;
		}
	}

	std::map< unsigned int, Track > automation;
	if( !scriptPath.empty() )
	{
		std::string error;
		const std::map< std::string, Track > tracks = loadScript( scriptPath, error );
		if( !error.empty() )
		{
			std::fprintf( stderr, "%s\n", error.c_str() );
			return 2;
		}
		const std::vector< NamedParameter > known = listParameters( rig.plugin );
		for( const auto& entry : tracks )
		{
			bool found = false;
			for( const NamedParameter& parameter : known )
				if( parameter.name == entry.first )
				{
					automation[ parameter.index ] = entry.second;
					found                         = true;
				}
			if( !found )
			{
				std::fprintf( stderr, "script names '%s', which is not a parameter (try --list)\n", entry.first.c_str() );
				return 2;
			}
		}
	}

	const Floats card = buildCard( width, height );
	std::vector< unsigned char > in( static_cast< size_t >( width ) * height * 4 );
	Floats picture( in.size() );

	for( int index = 0; filmFrames < 0 || index < filmFrames; ++index )
	{
		if( filmFrames < 0 )
		{
			size_t filled = 0;
			while( filled < in.size() )
			{
				const ssize_t got = read( STDIN_FILENO, in.data() + filled, in.size() - filled );
				if( got <= 0 )
					break;
				filled += static_cast< size_t >( got );
			}
			if( filled < in.size() )
				break;
			for( int y = 0; y < height; ++y )
				for( int x = 0; x < width * 4; ++x )
					picture[ static_cast< size_t >( height - 1 - y ) * width * 4 + x ] =
						in[ static_cast< size_t >( y ) * width * 4 + x ] / 255.0f;
			rig.Upload( picture );
		}
		else if( index == 0 )
			rig.Upload( card );

		for( const auto& track : automation )
			rig.plugin.SetFloatParameter( track.first, valueAt( track.second, index ) );

		if( !rig.Render( 1 ) )
			return 1;

		const Floats out = rig.Output();
		std::vector< unsigned char > bytes( in.size() );
		for( int y = 0; y < height; ++y )
			for( int x = 0; x < width * 4; ++x )
				bytes[ static_cast< size_t >( y ) * width * 4 + x ] = static_cast< unsigned char >( std::lround(
					std::clamp( out[ static_cast< size_t >( height - 1 - y ) * width * 4 + x ], 0.0f, 1.0f ) * 255.0f ) );

		size_t written = 0;
		while( written < bytes.size() )
		{
			const ssize_t put = write( STDOUT_FILENO, bytes.data() + written, bytes.size() - written );
			if( put <= 0 )
				return 1;
			written += static_cast< size_t >( put );
		}
	}
	return 0;
}

//---------------------------------------------------------------------------
// --probe: what the lamp is doing, every N frames. A development aid.
//---------------------------------------------------------------------------
int runProbe( int width, int height, int frames, const std::vector< std::string >& settings, int every )
{
	Rig rig;
	if( !rig.Init( width, height ) )
		return 1;
	for( const std::string& setting : settings )
	{
		std::string error;
		if( !applySetting( rig.plugin, setting, error ) )
		{
			std::fprintf( stderr, "--set %s: %s\n", setting.c_str(), error.c_str() );
			return 2;
		}
	}
	rig.plugin.KeepStepLog( true );
	for( int f = 0; f < frames; ++f )
	{
		const auto start = std::chrono::steady_clock::now();
		if( !rig.Render( 1 ) )
			return 1;
		glFinish();
		const double ms = std::chrono::duration< double, std::milli >( std::chrono::steady_clock::now() - start ).count();
		if( f % every != 0 && f != frames - 1 )
			continue;
		const Field s = readField( rig );
		double tmin = 1e9, tmax = -1e9, pmin = 1e9, pmax = -1e9;
		for( int j = 0; j < s.ny; ++j )
			for( int i = 0; i < s.nx; ++i )
			{
				tmin = std::min( tmin, static_cast< double >( s.at( i, j, 1 ) ) );
				tmax = std::max( tmax, static_cast< double >( s.at( i, j, 1 ) ) );
				pmin = std::min( pmin, static_cast< double >( s.at( i, j, 0 ) ) );
				pmax = std::max( pmax, static_cast< double >( s.at( i, j, 0 ) ) );
			}
		const Velocity vel = velocityFrom( rig.Psi(), rig.plugin.CurrentGrid() );
		const auto& log = rig.plugin.StepLog();
		if( !log.empty() )
			std::printf( "  last step: dt %.4f s, speed used %.3f mm/s\n", log.back().dt, log.back().speed * 1000.0 );
		std::printf( "frame %4d  t=%8.1f s  substeps %2d  short %.3f  wax %.2f  phi [%.4f %.4f]  T [%.2f %.2f] mean %.3f  "
		             "|u| %.2f mm/s  bulb %.1f W  %.2f ms\n",
		             f, rig.plugin.SimTime(), rig.plugin.LastSubsteps(), rig.plugin.LastShortfall(),
		             sumChannel( s, 0 ) / ( s.nx * s.ny ), pmin, pmax, tmin, tmax, sumChannel( s, 1 ) / ( s.nx * s.ny ),
		             vel.MaxSpeed() * 1000.0, rig.plugin.BulbPower(), ms );
	}
	return 0;
}
} // namespace

//---------------------------------------------------------------------------
int main( int argc, char** argv )
{
	std::string outPath = "/tmp/bassalt.png";
	std::string cardPath;
	std::vector< std::string > settings;
	int width  = 1280;
	int height = 720;
	int frames = 120;
	int every  = 30;
	std::vector< int > warmFrames, resetFrames, pourFrames;
	bool beat = false;
	std::string mode;
	std::string scriptPath;
	int filmFrames = -1;

	for( int i = 1; i < argc; ++i )
	{
		const std::string argument = argv[ i ];
		const bool hasNext         = i + 1 < argc;

		if( argument == "--help" || argument == "-h" )
		{
			std::printf(
				"bstest -- render Bassalt offline and measure its lamp\n\n"
				"  --out PATH        render the card through the lamp and write it here\n"
				"  --card PATH       write the card itself\n"
				"  --size WxH        render size (default 1280x720)\n"
				"  --frames N        frames at 60 fps before reading back (default 120)\n"
				"  --warm N          press Warm on frame N. Repeatable. Likewise --reset, --pour.\n"
				"  --beat            feed a beat every half second into the Audio buffer\n"
				"  --set \"Name=V\"    set a parameter by its display name. Repeatable.\n"
				"  --list            print every parameter and its default, then exit\n"
				"  --pipe            raw RGBA frames on stdin, raw RGBA frames on stdout\n"
				"  --film N          N frames of the card, raw RGBA frames on stdout\n"
				"  --script PATH     parameter cues for --pipe/--film: 'frame Name value'\n"
				"  --probe [N]       print the lamp's state every N frames\n\n"
				"  --still --volume --crossover --darcy --multigrid --diffusion --rt --heat\n"
				"  --bulb --glass --state --negative --mutate --bench\n" );
			return 0;
		}
		else if( argument == "--out" && hasNext )
			outPath = argv[ ++i ];
		else if( argument == "--card" && hasNext )
			cardPath = argv[ ++i ];
		else if( argument == "--set" && hasNext )
			settings.push_back( argv[ ++i ] );
		else if( argument == "--frames" && hasNext )
			frames = std::atoi( argv[ ++i ] );
		else if( argument == "--warm" && hasNext )
			warmFrames.push_back( std::atoi( argv[ ++i ] ) );
		else if( argument == "--reset" && hasNext )
			resetFrames.push_back( std::atoi( argv[ ++i ] ) );
		else if( argument == "--pour" && hasNext )
			pourFrames.push_back( std::atoi( argv[ ++i ] ) );
		else if( argument == "--beat" )
			beat = true;
		else if( argument == "--pipe" )
			mode = "pipe";
		else if( argument == "--film" && hasNext )
		{
			mode       = "pipe";
			filmFrames = std::max( 1, std::atoi( argv[ ++i ] ) );
		}
		else if( argument == "--script" && hasNext )
			scriptPath = argv[ ++i ];
		else if( argument == "--list" )
			mode = "list";
		else if( argument == "--dump" && hasNext )
			g_dumpDir = argv[ ++i ];
		else if( argument == "--probe" )
		{
			mode = "probe";
			if( hasNext && std::isdigit( static_cast< unsigned char >( argv[ i + 1 ][ 0 ] ) ) )
				every = std::max( 1, std::atoi( argv[ ++i ] ) );
		}
		else if( argument.rfind( "--", 0 ) == 0 && CheckNamed( argument.substr( 2 ) ) )
			mode = argument.substr( 2 );
		else if( argument == "--size" && hasNext )
		{
			const std::string value = argv[ ++i ];
			const size_t cross      = value.find( 'x' );
			if( cross != std::string::npos )
			{
				width  = std::atoi( value.substr( 0, cross ).c_str() );
				height = std::atoi( value.substr( cross + 1 ).c_str() );
			}
		}
		else
		{
			std::fprintf( stderr, "unknown argument '%s' (try --help)\n", argument.c_str() );
			return 2;
		}
	}

	if( mode == "list" )
	{
		BassaltPlugin plugin;
		std::printf( "%-3s %-18s %-9s %s\n", "id", "name", "kind", "default" );
		for( const NamedParameter& parameter : listParameters( plugin ) )
			std::printf( "%-3u %-18s %-9s %.4f\n", parameter.index, parameter.name.c_str(), parameter.kind.c_str(),
			             parameter.value );
		return 0;
	}

	if( !cardPath.empty() )
	{
		if( !writePng( cardPath, width, height, buildCard( width, height ) ) )
		{
			std::fprintf( stderr, "could not write %s\n", cardPath.c_str() );
			return 1;
		}
		std::printf( "wrote %s -- the card, %dx%d\n", cardPath.c_str(), width, height );
		return 0;
	}

	CGLContextObj context = createContext();
	if( context == nullptr )
	{
		std::fprintf( stderr, "could not create an OpenGL 4.1 core context\n" );
		return 1;
	}

	int result = 0;
	if( mode == "pipe" )
		result = runPipe( width, height, scriptPath, filmFrames, beat, settings );
	else if( mode == "probe" )
		result = runProbe( width, height, frames, settings, every );
	else if( !mode.empty() )
		result = RunNamed( mode );
	else
	{
		Rig rig;
		if( !rig.Init( width, height ) )
			result = 1;
		else
		{
			for( const std::string& setting : settings )
			{
				std::string error;
				if( !applySetting( rig.plugin, setting, error ) )
				{
					std::fprintf( stderr, "--set %s: %s\n", setting.c_str(), error.c_str() );
					return 2;
				}
			}
			if( beat )
				rig.feed = AudioFeed::Pulses;

			auto has = []( const std::vector< int >& list, int f ) { return std::find( list.begin(), list.end(), f ) != list.end(); };
			for( int f = 0; f < std::max( frames, 1 ) && result == 0; ++f )
			{
				if( has( warmFrames, f ) )
					rig.Press( PT_WARM );
				if( has( resetFrames, f ) )
					rig.Press( PT_RESET );
				if( has( pourFrames, f ) )
					rig.Press( PT_POUR );
				if( !rig.Render( 1 ) )
					result = 1;
			}

			if( result == 0 )
			{
				if( writePng( outPath, width, height, rig.Output() ) )
					std::printf( "wrote %s -- %dx%d, %d frames (%.1f s of lamp)\n", outPath.c_str(), width, height,
					             frames, rig.plugin.SimTime() );
				else
				{
					std::fprintf( stderr, "could not write %s\n", outPath.c_str() );
					result = 1;
				}
			}
		}
	}

	CGLSetCurrentContext( nullptr );
	CGLDestroyContext( context );
	return result;
}
