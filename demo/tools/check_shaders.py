"""The demo's GLSL must be the plugin's GLSL, character for character.

    python3 demo/tools/check_shaders.py

Called from `tools/verify.sh`. Exit code 1 means the two have drifted.

------------------------------------------------------------------- the point

`demo/plugin.js` carries a second copy of every shader in `source/Shaders.cpp`,
because a browser cannot include a C++ file. Two copies of a shader is exactly
the arrangement that drifts, and the drift is invisible from both sides: the
plugin keeps working, the page keeps working, and they quietly stop being the
same effect. The page's whole claim is that what it runs is the plugin's own
code, so the moment that stops being checkable the page is a lie.

This compares the text, not the behaviour. Reformatting counts as drift, and
that is deliberate -- "it is only whitespace" is how a real change gets waved
through.

The update pass is two adjacent raw strings in the C++ (MSVC caps one literal at
about 16 KB), which the compiler joins; they are joined here the same way, and
the demo carries the joined text as one constant.

--------------------------------------------------------------- what it cannot

Nothing here checks the *ported* orchestration. `controls`, `physics`,
`chooseGrid` and `createRenderer` in plugin.js are a hand translation of
`source/Controls.cpp`, `source/Physics.cpp` and `BassaltPlugin` in
`source/Bassalt.cpp` -- the pass sequence, the multigrid, the substep limits and
the uniforms -- and only a reader can tell whether they still agree.

Nor does it check what the kit does to the text on its way to the compiler.
`port()` in demo/vendor/gl.js rewrites the version line, prepends the precision
block and DROPS `precise`, which bassalt relies on (AGENTS.md). That happens to
the string at run time, so the constants here stay byte-identical to the C++
and this check keeps its grip; the page says what dropping it costs.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# JS constant, C++ file, C++ symbol. Every ShaderId in source/Shaders.h.
SHADERS = [
    ("VERTEX", "source/Shaders.cpp", "kVertexShader"),
    ("PROPS", "source/Shaders.cpp", "kPropsShader"),
    ("COEF", "source/Shaders.cpp", "kCoefShader"),
    ("CIRCULATION", "source/Shaders.cpp", "kCirculationShader"),
    ("COARSEN", "source/Shaders.cpp", "kCoarsenShader"),
    ("SMOOTH", "source/Shaders.cpp", "kSmoothShader"),
    ("RESTRICT", "source/Shaders.cpp", "kRestrictShader"),
    ("COARSEST", "source/Shaders.cpp", "kCoarsestShader"),
    ("PROLONG", "source/Shaders.cpp", "kProlongShader"),
    ("UPDATE", "source/Shaders.cpp", "kUpdateShader"),
    ("INTERFACE", "source/Shaders.cpp", "kInterfaceShader"),
    ("REDUCE", "source/Shaders.cpp", "kReduceShader"),
    ("EVENT", "source/Shaders.cpp", "kEventShader"),
    ("INFLATE", "source/Shaders.cpp", "kInflateShader"),
    ("COMPOSITE", "source/Shaders.cpp", "kCompositeShader"),
]


def from_cpp(path, symbol):
    """`const char* const symbol = R"(...)" [R"(...)" ...];`, joined."""
    with open(os.path.join(REPO, path)) as handle:
        source = handle.read()
    match = re.search(
        r'const char\* const ' + re.escape(symbol) + r' = ((?:\s*R"\(.*?\)")+)\s*;',
        source,
        re.S,
    )
    if match is None:
        return None
    return "".join(re.findall(r'R"\((.*?)\)"', match.group(1), re.S))


def from_js(source, name):
    match = re.search(r'^const ' + re.escape(name) + r' = `(.*?)`;$', source, re.S | re.M)
    if match is None:
        return None, None

    body = match.group(1)

    # Undo the one escape the literal needs -- a comment in the update pass
    # quotes `precise` in backticks -- and refuse the rest. The C++ carries no
    # backslash at all, so a stray one here is either a typo or a difference
    # being smuggled through the JS string decoder.
    stray = re.search(r"\\(?!`)", body)
    if stray is not None:
        upto = body[: stray.start()]
        return None, f"backslash that is not an escaped backtick, at line {upto.count(chr(10)) + 1}"
    if "${" in body:
        return None, "template substitution"

    return body.replace("\\`", "`"), None


def main():
    with open(os.path.join(REPO, "demo", "plugin.js")) as handle:
        js = handle.read()

    problems = 0
    for name, path, symbol in SHADERS:
        cpp_text = from_cpp(path, symbol)
        js_text, complaint = from_js(js, name)

        if cpp_text is None:
            print(f"FAIL  {symbol} not found in {path}")
            problems += 1
            continue
        if "\\" in cpp_text or "${" in cpp_text:
            print(f"FAIL  {symbol} contains a backslash or ${{ -- the JS copy could not be compared")
            problems += 1
            continue
        if complaint is not None:
            print(f"FAIL  {name} in demo/plugin.js has a {complaint}")
            problems += 1
            continue
        if js_text is None:
            print(f"FAIL  {name} not found in demo/plugin.js")
            problems += 1
            continue

        if cpp_text == js_text:
            print(f"ok    {name:<12} matches {symbol} ({len(cpp_text)} chars)")
            continue

        problems += 1
        print(f"FAIL  {name} has drifted from {symbol} in {path}")

        cpp_lines = cpp_text.splitlines()
        js_lines = js_text.splitlines()
        for i in range(max(len(cpp_lines), len(js_lines))):
            a = cpp_lines[i] if i < len(cpp_lines) else "<missing>"
            b = js_lines[i] if i < len(js_lines) else "<missing>"
            if a != b:
                print(f"        first difference at line {i + 1}")
                print(f"          C++: {a}")
                print(f"          js : {b}")
                break

    print()
    if problems:
        print(f"{problems} shader(s) differ -- copy the C++ across, do not edit plugin.js by hand")
        return 1

    print(f"all {len(SHADERS)} shaders are identical to the plugin's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
