#include "Controls.h"

#include <algorithm>
#include <cmath>

namespace bassalt
{
namespace
{
/// value^0 = low, value^1 = high, with every octave the same width of travel.
float geometric( float value, float low, float high )
{
	return low * std::pow( high / low, std::clamp( value, 0.0f, 1.0f ) );
}

float linear( float value, float low, float high )
{
	return low + ( high - low ) * std::clamp( value, 0.0f, 1.0f );
}

constexpr float kSpeedLow  = 1.0f;
constexpr float kSpeedHigh = 300.0f;
} // namespace

float LampHeightFromParam( float value )
{
	return geometric( value, 0.1f, 1.0f );
}

float GapFromParam( float value )
{
	return geometric( value, 0.001f, 0.020f );
}

float SpeedFromParam( float value )
{
	return geometric( value, kSpeedLow, kSpeedHigh );
}

float ParamFromSpeed( float speed )
{
	const float clamped = std::clamp( speed, kSpeedLow, kSpeedHigh );
	return std::log( clamped / kSpeedLow ) / std::log( kSpeedHigh / kSpeedLow );
}

float AmbientFromParam( float value )
{
	return linear( value, 10.0f, 40.0f );
}

float BulbFromParam( float value )
{
	return linear( value, 0.0f, 60.0f );
}

float BulbLagFromParam( float value )
{
	return geometric( value, 0.5f, 120.0f );
}

float CoilFromParam( float value )
{
	return linear( value, 0.0f, 6.0f );
}

float SaltFromParam( float value )
{
	return linear( value, 0.0f, 4.0f );
}

float WaxAmountFromParam( float value )
{
	return linear( value, 0.05f, 0.40f );
}

float TensionFromParam( float value )
{
	const float v = std::clamp( value, 0.0f, 1.0f );
	return 0.01f * v * v;
}

float WaxViscosityFromParam( float value )
{
	return geometric( value, 0.001f, 1.0f );
}

float MeltingPointFromParam( float value )
{
	return linear( value, 35.0f, 70.0f );
}

float BassFromParam( float value )
{
	return linear( value, 0.0f, 60.0f );
}

int BassBandFromParam( float value )
{
	return std::clamp( static_cast< int >( std::lround( linear( value, 1.0f, 8.0f ) ) ), 1, 8 );
}

float KickFromParam( float value )
{
	return linear( value, 0.0f, 300.0f );
}

float ClipHeatFromParam( float value )
{
	return linear( value, 0.0f, 60.0f );
}

float GlowFromParam( float value )
{
	return linear( value, 0.0f, 2.0f );
}

float RefractionFromParam( float value )
{
	return linear( value, 0.0f, 1.0f );
}

} // namespace bassalt
