#pragma once

#include <array>

/**
	The audio side: a spectrum in, a level and an onset out.

	Copied from millpond, which took it from rosette, which adapted it from
	macroblock's analyser -- the one that has been driven with real music
	through Resolume's FFT buffer. Bassalt adds a bass level (the mean of the
	lowest Bass Band bins, normalised against its own recent peak) that drives
	the bulb, and keeps the onset detector (each onset is a Kick of heat),
	including the primed first frame that keeps it from going deaf on a clip
	trigger.

	**The bin law is an assumption.** Resolume does not document what
	frequencies its 64 bins span, and nobody in the fleet has measured it. If
	they are linear to a 22 kHz Nyquist, bin 0 alone is 0-345 Hz and the default
	Bass Band of 2 reaches 690 Hz; if they are log-spaced, 2 bins is the kick
	and the sub. Bass Band is how an operator compensates.

	**Where the audio comes from.** FFGL has no audio path. What Resolume
	provides is a buffer parameter declared `FF_USAGE_FFT`, which the host
	fills with a 64-bin spectrum once per frame -- a *modulation* source at
	video rate, not a signal source, so the smallest interval this can
	resolve is a frame and a kick lands on the frame after the transient.

	**Normalised against its own recent peak**, so the same Audio Rain means
	the same downpour on a quiet stem and a mastered track. The cost is that a
	long loud passage reads as "1" throughout, because relative to the last
	few seconds it is.
*/
namespace bassalt::audio
{

/// The spectrum Resolume delivers. Fixed by the host, not chosen here.
constexpr int kBins = 64;

struct Settings
{
	float attackSeconds  = 0.010f;
	float releaseSeconds = 0.250f;

	/// How easily an onset fires: the margin over the adaptive flux floor.
	float sensitivity = 0.6f;

	/// The time constant the latched kick decays with.
	float holdSeconds = 0.30f;

	/// How many of the lowest bins are the bass.
	int bassBins = 2;

	/// Prime on the first frame (see Update). Off only for the harness's
	/// negative control, which shows what the fleet's bug looked like.
	bool prime = true;
};

class Analyser
{
public:
	/// `bins` is what the host handed over; `count` may be less than kBins if
	/// it handed over fewer. `dt` is the frame in seconds.
	void Update( const float* bins, int count, float dt, const Settings& settings );

	/// The full-range level, normalised against its recent peak, 0..1.
	float Level() const;

	/// The latched kick: the level an onset arrived at, decaying since. 0..1.
	float Kick() const;

	/// The bass: the lowest bins' mean (of the square roots, enveloped), NOT
	/// normalised. A peak-normalised level re-scales a breakdown back up to
	/// "1" within seconds, and the point of bass driving a bulb is that a
	/// breakdown lets the lamp cool. Bass is the gain. 0..1.
	float Bass() const;

	/// True on the frame an onset was detected.
	bool Fired() const
	{
		return fired;
	}

	/// How many onsets have been detected since the plugin loaded.
	unsigned long long Onsets() const
	{
		return onsets;
	}

	/// Forget everything. Used when the host's clock jumps.
	void Reset();

private:
	std::array< float, kBins > binLevel {};
	std::array< float, kBins > binPrevious {};

	float level      = 0.0f;///< enveloped, sqrt-compressed
	float bass       = 0.0f;///< the lowest bins, enveloped
	float peak       = 0.0f;///< slow decay, the normalising reference
	float flux       = 0.0f;///< positive spectral difference this frame
	float fluxMean   = 0.0f;///< running mean of the above, the adaptive floor
	float held       = 0.0f;///< the latched kick, decaying
	float refractory = 0.0f;///< seconds still to wait before another onset
	bool fired       = false;

	/// False until a frame has been seen. The first frame has no previous
	/// frame, so its flux is not small -- it is undefined, and computing it
	/// against a buffer of zeroes reports the entire spectrum as a rise. See
	/// Update().
	bool primed = false;

	unsigned long long onsets = 0;
};

} // namespace bassalt::audio
