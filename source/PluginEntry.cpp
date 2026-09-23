#include "Bassalt.h"

/**
    The one registration.

    This file is listed directly in the Bassalt MODULE target, not in
    bassalt_core: `CFFGLPluginInfo` registers itself from a file-scope
    constructor and nothing ever references it by name, so in a STATIC archive
    the linker is entitled to drop the whole translation unit -- giving a
    bundle that loads, exports `plugMain`, and reports that it contains no
    plugins. The core stays an OBJECT library for the same reason.

        nm -gU Bassalt.bundle/Contents/MacOS/Bassalt | grep plugMain

    The name is `SW Bassalt`, ten characters. The FFGL name field is
    `char[ 16 ]` and is **not** null-terminated, so the host truncates without
    saying so. `oxbow probe` is what reads it back the way a host does.
*/
namespace
{
class BassaltEffect : public bassalt::BassaltPlugin
{
};
} // namespace

static CFFGLPluginInfo PluginInfo(
	PluginFactory< BassaltEffect >,// Create method
	"BS01",                        // Plugin unique ID of maximum length 4
	"SW Bassalt",                  // Plugin name
	2,                             // API major version number
	1,                             // API minor version number
	0,                             // Plugin major version number
	1,                             // Plugin minor version number
	FF_EFFECT,                     // Plugin type
	"A lava lamp as a heat engine: a bulb, wax and salted water, Cahn-Hilliard wax, advected heat and "
	"Hele-Shaw flow. The rising pillars, the pinch-off, the warm-up and the stalls fall out of the model, "
	"and the clip is the world behind the lamp or the wax itself.",
	"Bassalt FFGL effect"          // About
);

extern "C" const char* BassaltBuildStamp()
{
	return "bassalt " BASSALT_VERSION ", built " __DATE__ " " __TIME__;
}
