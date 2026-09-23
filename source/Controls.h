#pragma once

/**
    The host's parameters, and what they mean in physical units.

    Every numeric parameter the host sees is a plain 0..1 float, because
    `SetParamInfo` clamps an `FF_TYPE_STANDARD` default into 0..1 *before*
    `SetParamRange` could widen it -- so a control that stands for a height in
    metres or a power in watts cannot declare one as its default. The
    conversions all live in Controls.cpp, one function per control, and the
    shaders are handed the physical value.

    Option and event parameters are the exception: they hold the element value
    itself.

    ------------------------------------------------------------ lengths

    **Every length is in metres, and the frame height is Lamp Height metres.**
    The lamp is simulated on a grid sized in metres, not pixels, so the same
    settings run the same lamp at 720p and at 4K -- the raster only decides how
    finely the finished lamp is looked at. That is why every *physics* check in
    `bstest` is raster-independent by construction, and only the *optics*
    checks have to run at two rasters.
*/

namespace bassalt
{
/**
    Parameter ids.

    **Append only.** `SetParamGroup` collapses runs of consecutive same-group
    ids, so inserting an id mid-enum silently splits a group in two; and every
    saved composition stores parameters by index, so a renumber rewrites what
    an operator's old project means.
*/
enum ParamId : unsigned int
{
	// -- Lamp ---------------------------------------------------------------
	PT_LAMP_HEIGHT = 0,
	PT_GAP,
	PT_DETAIL,
	PT_SPEED,
	PT_AMBIENT,
	PT_WARM,
	PT_RESET,

	// -- Bulb ---------------------------------------------------------------
	PT_BULB,
	PT_BULB_LAG,
	PT_COIL,

	// -- Fluids -------------------------------------------------------------
	PT_SALT,
	PT_WAX_AMOUNT,
	PT_TENSION,
	PT_WAX_VISCOSITY,
	PT_MELTING_POINT,

	// -- Audio --------------------------------------------------------------
	PT_AUDIO,
	PT_BASS,
	PT_BASS_BAND,
	PT_KICK,

	// -- Clip ---------------------------------------------------------------
	PT_CLIP,
	PT_POUR,
	PT_POUR_THRESHOLD,
	PT_CLIP_HEAT,

	// -- Look ---------------------------------------------------------------
	PT_WAX_R,
	PT_WAX_G,
	PT_WAX_B,
	PT_TINT_R,
	PT_TINT_G,
	PT_TINT_B,
	PT_GLOW,
	PT_GLASS,
	PT_REFRACTION,

	// -- Output -------------------------------------------------------------
	PT_VIEW,
	PT_MIX,

	// -- The Stoatworks About block -----------------------------------------
	//
	// One display-only text line, then one button per link the block carries.
	// Bassalt.cpp static_asserts this run against `about::kParamCount`, so a
	// user guide (which adds a link) cannot shift PT_COUNT silently.
	PT_ABOUT_TEXT,
	PT_ABOUT_BUTTON_1,
	PT_ABOUT_BUTTON_2,
	PT_ABOUT_BUTTON_3,
	PT_COUNT
};

/// How the clip goes through the lamp.
enum class ClipMode
{
	Behind = 0,///< the clip is the world behind the lamp, seen through it
	Dyed,      ///< the clip is the wax, carried by the flow

	Count
};

/// The container.
enum class Glass
{
	Flat = 0,///< the frame is a flat panel lamp
	Cylinder,///< the frame is a vertical glass cylinder, seen side on

	Count
};

/// What the output shows. Lamp is the effect; the others show one field of
/// the model on its own.
enum class View
{
	Lamp = 0,
	Temperature,
	Wax,
	Velocity,
	Density,

	Count
};

/// Grid cells up the lamp's height. The width follows the frame's aspect,
/// rounded to a multiple of 16 so the multigrid can halve it four times.
constexpr int kDetailCells[] = { 64, 96, 128, 192, 256 };
constexpr int kDetailCount   = 5;

//---------------------------------------------------------------------------
// The mappings. Each says its range and its shape.
//---------------------------------------------------------------------------

/// The lamp's height (the frame's height) in metres: 0.1 to 1, geometrically.
float LampHeightFromParam( float value );

/// The Hele-Shaw gap -- the lamp's depth front to back -- in metres: 1 mm to
/// 20 mm, geometrically. The flow goes as its square (K = b^2 / 12 mu) and the
/// warm-up time as the gap itself (more water to heat per square metre of
/// glass losing it).
float GapFromParam( float value );

/// Lamp seconds per host second: 1 to 300, geometrically. A real lamp takes an
/// hour to warm; at 300x that is twelve seconds.
float SpeedFromParam( float value );
float ParamFromSpeed( float speed );

/// The room, degrees C: 10 to 40, linearly.
float AmbientFromParam( float value );

/// The bulb's electrical power, watts: 0 to 60, linearly.
float BulbFromParam( float value );

/// The bulb's thermal time constant, lamp seconds: 0.5 to 120, geometrically.
float BulbLagFromParam( float value );

/// The coil's added conductivity, W/(m K): 0 to 6, linearly. The coil is a
/// spiral of wire on the base; the number is the conductivity of the layer it
/// sits in, averaged over wire and the wax between the turns.
float CoilFromParam( float value );

/// Salt in the water, per cent by mass: 0 to 4, linearly.
float SaltFromParam( float value );

/// Wax, as a fraction of the lamp's volume: 0.05 to 0.40, linearly. Applied on
/// Reset.
float WaxAmountFromParam( float value );

/// Wax-water interfacial tension, N/m: 0.01 * value^2, so 0 is none and the
/// default 0.447 is 2 mN/m, a lamp's surfactant-laden wax. Clean paraffin on
/// clean water is 50 mN/m and out of range: the flow step's capillary limit
/// goes as 1 / sigma (Physics.h), and at 50 it would take a hundred substeps a
/// frame.
float TensionFromParam( float value );

/// Hot (melted) wax's dynamic viscosity, Pa s: 1 mPa s to 1 Pa s,
/// geometrically.
float WaxViscosityFromParam( float value );

/// Wax melting point, degrees C: 35 to 70, linearly.
float MeltingPointFromParam( float value );

/// Bass: watts added to the bulb at full bass level, 0 to 60, linearly.
float BassFromParam( float value );

/// Bass Band: how many of the lowest FFT bins are the bass, 1 to 8.
int BassBandFromParam( float value );

/// Kick: joules of heat per onset, put into the base in one step: 0 to 300.
float KickFromParam( float value );

/// Clip Heat: watts the clip adds when it is all white, 0 to 60, linearly.
float ClipHeatFromParam( float value );

/// Glow: the bulb's light in the lamp, 0 to 2, linearly. 1 is the default look.
float GlowFromParam( float value );

/// Refraction: the distance from the lamp's back to the world behind it, in
/// metres: 0 to 1, linearly. At 0 the world is painted on the back glass and
/// nothing is displaced; the further away it is, the more the wax's lenses and
/// the cylinder move it.
float RefractionFromParam( float value );

} // namespace bassalt
