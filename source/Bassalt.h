#pragma once

#include "Audio.h"
#include "Controls.h"
#include "PassBuffer.h"
#include "Physics.h"
#include "Shaders.h"

#include <FFGLSDK.h>

// After FFGLSDK.h, which is where FFUInt32 comes from.
#include "StoatworksAboutParams.h"

#include <unordered_map>
#include <vector>

namespace bassalt
{
/// A compiled pass with its uniform locations cached. FFGLShader::Set looks
/// every name up with glGetUniformLocation, and a frame here sets a few
/// thousand uniforms across a hundred-odd passes; the lookups alone cost more
/// than the arithmetic. Keyed by the literal's address: each call site passes
/// the same pointer every time.
struct Program
{
	ffglex::FFGLShader shader;
	std::unordered_map< const char*, GLint > cache;

	GLuint Id() const
	{
		return shader.GetGLID();
	}
	GLint Loc( const char* name );
	void Set( const char* name, float v );
	void Set( const char* name, float a, float b );
	void Set( const char* name, float a, float b, float c );
	void SetInt( const char* name, int v );
	void SetInts( const char* name, int a, int b );
	void Release();
};

/// One elliptic problem's multigrid storage: per level, the coefficients
/// (aE, aN, s), the right-hand side and a ping-pong pair for the solution.
struct Multigrid
{
	struct Level
	{
		PassBuffer coef;
		PassBuffer rhs;
		PassBuffer psi[ 2 ];
		int current = 0;
		int nodesX  = 0;
		int nodesY  = 0;
	};
	std::vector< Level > levels;

	bool Ensure( const physics::Grid& grid );
	void Destroy();
	void ClearSolution();
	GLuint Solution() const
	{
		return levels.empty() ? 0 : levels[ 0 ].psi[ levels[ 0 ].current ].TextureID();
	}
};

/// What one substep did, for the harness's bookkeeping.
struct StepRecord
{
	double dt        = 0.0;///< lamp seconds
	double bulbPower = 0.0;///< W through the base, the kick included
	double clipPower = 0.0;///< W at full white
	double speed     = 0.0;///< the fastest face, m/s
};

/**
    The plugin. See AGENTS.md for why each piece is built the way it is.
*/
class BassaltPlugin : public CFFGLPlugin
{
public:
	BassaltPlugin();

	FFResult InitGL( const FFGLViewportStruct* viewport ) override;
	FFResult ProcessOpenGL( ProcessOpenGLStruct* input ) override;
	FFResult DeInitGL() override;

	FFResult SetFloatParameter( unsigned int index, float value ) override;
	float GetFloatParameter( unsigned int index ) override;

	char* GetTextParameter( unsigned int index ) override;

	/// Declared only so the About line can accept its own default.
	/// instantiateGL pushes every declared default back through the setters on
	/// a fresh instance and deletes the instance if one fails, and
	/// CFFGLPlugin's SetTextParameter is a stub that returns exactly that
	/// failure -- so without this override no real host can load the plugin,
	/// while every offline harness here carries on passing.
	FFResult SetTextParameter( unsigned int index, const char* value ) override;

	FFResult SetTime( double time ) override;

	//-------------------------------------------------------------------
	// For the harness. Nothing in the plugin's own operation calls these;
	// they exist so `bstest` can measure the fields the shipping passes
	// produced, rather than a copy of them computed for the test.
	//-------------------------------------------------------------------

	/// The harness declares its clock (seconds) rather than letting the plugin
	/// vote on it against the wall clock.
	void SetClockScaleForTest( double scale );

	const physics::Grid& CurrentGrid() const
	{
		return grid;
	}
	physics::Lamp CurrentLamp() const;

	GLuint StateTextureID() const;
	GLuint PropsTextureID() const
	{
		return props.TextureID();
	}
	GLuint PsiTextureID() const
	{
		return flow.Solution();
	}
	GLuint CoefTextureID() const
	{
		return flow.levels.empty() ? 0 : flow.levels[ 0 ].coef.TextureID();
	}
	GLuint RhsTextureID() const
	{
		return flow.levels.empty() ? 0 : flow.levels[ 0 ].rhs.TextureID();
	}
	GLuint ThicknessTextureID() const
	{
		return inflate.Solution();
	}

	/// Replace (phi, T, U, V), one RGBA float per cell, row 0 first.
	void LoadStateForTest( const std::vector< float >& rgba );

	/// Flow frozen: psi held at zero and no solve. For --diffusion and --heat,
	/// which the spec asks for with the flow off.
	void SetFlowFrozenForTest( bool frozen )
	{
		flowFrozen = frozen;
	}

	/// The cap's extra loss off, so every cell loses heat at the same rate and
	/// the lamp's mean obeys the lumped law exactly (--heat).
	void SetCapForTest( bool on )
	{
		capOn = on;
	}

	/// Run the flow solve alone on the current state -- the same passes a
	/// substep runs before its update -- with `cycles` V-cycles, from zero if
	/// asked. Nothing moves.
	void SolveForTest( int cycles, bool fromZero );

	/// The substeps taken so far, if asked to keep them.
	void KeepStepLog( bool keep )
	{
		keepLog = keep;
		stepLog.clear();
	}
	const std::vector< StepRecord >& StepLog() const
	{
		return stepLog;
	}

	double SimTime() const
	{
		return simTime;
	}
	double BulbPower() const
	{
		return bulbPower;
	}
	double BulbTarget() const;
	int LastSubsteps() const
	{
		return lastSubsteps;
	}
	double LastShortfall() const
	{
		return lastShortfall;
	}

	/// The analyser, for --bulb: its onsets, and its priming switched off for
	/// the negative control.
	audio::Analyser& AnalyserForTest()
	{
		return analyser;
	}
	void SetAudioPrimingForTest( bool prime )
	{
		audioPriming = prime;
	}

private:
	void UpdateClock();
	void UpdateAudio( double dt );

	bool ensureBuffers( const physics::Grid& wanted, GLsizei width, GLsizei height );

	/// The passes.
	void Begin( PassBuffer& target, Program& program );
	void BindTexture( int unit, GLuint texture );
	void Draw();
	void PropsPass();
	void FlowSolve( int cycles, bool fromZero );
	void Coarsen( Multigrid& system );
	void Cycle( Multigrid& system, int level );
	void Smooth( Multigrid& system, int level, bool zeroGuess );
	void ReduceSpeed();
	double FastestFaceNow();
	void RequestSpeed();
	void CollectSpeed();
	void UpdatePass( double dt, double power, double clipPower, GLuint clipTexture, const FFGLTexCoords& maxUV );
	void InterfacePass( double dt );
	void EventPass( int mode, float temperature, GLuint clipTexture, const FFGLTexCoords& maxUV );
	void InflatePass();
	void Simulate( double dt, GLuint clipTexture, const FFGLTexCoords& maxUV );

	float params[ PT_COUNT ] = {};

	Program programs[ static_cast< int >( ShaderId::Count ) ];
	Program& P( ShaderId id )
	{
		return programs[ static_cast< int >( id ) ];
	}
	ffglex::FFGLScreenQuad quad;

	PassBuffer state[ 2 ];///< RGBA32F cells: (phi, T, U, V)
	PassBuffer props;     ///< RGBA32F cells: (mu^, rho', A, k)
	PassBuffer reduce[ 2 ];
	Multigrid flow;       ///< psi
	Multigrid inflate;    ///< the lens thickness
	int stateIndex = 0;
	bool haveThickness = false;

	/// The fastest face, read back a frame late (see Simulate). Negative:
	/// unknown, read it now.
	GLuint speedBuffer   = 0;
	GLsync speedFence    = nullptr;
	double speedEstimate = -1.0;

	physics::Grid grid;
	physics::Lamp lamp;
	int bufferWidth  = 0;
	int bufferHeight = 0;

	//-------------------------------------------------------------------
	// Time. The host's clock unit is voted on against the wall clock
	// (rosette's code), because Resolume has sent both seconds and
	// milliseconds.
	//-------------------------------------------------------------------
	double hostTime     = -1.0;
	double lastRawTime  = -1.0;
	double lastWallTime = -1.0;
	double wallStart    = -1.0;
	double clockScale   = 0.0;
	int secondsVotes    = 0;
	int millisVotes     = 0;
	int clockFrames     = 0;
	double now          = 0.0;
	double lastNow      = -1.0;
	double simTime      = 0.0;
	bool settledJump    = false;

	//-------------------------------------------------------------------
	// The bulb and the events.
	//-------------------------------------------------------------------
	double bulbPower   = 0.0;///< W, after the bulb's lag
	double pendingKick = 0.0;///< J, into the base on the next substep
	bool warmWanted    = false;
	bool resetWanted   = false;
	bool pourWanted    = false;
	bool warmHeld      = false;
	bool resetHeld     = false;
	bool pourHeld      = false;
	bool fresh         = true;///< no state yet: start warm

	audio::Analyser analyser;
	bool audioPriming = true;

	//-------------------------------------------------------------------
	// Test switches, all off in the plugin's own operation.
	//-------------------------------------------------------------------
	bool flowFrozen = false;
	bool capOn      = true;
	bool keepLog    = false;
	std::vector< StepRecord > stepLog;
	int lastSubsteps     = 0;
	double lastShortfall = 0.0;
};

} // namespace bassalt
