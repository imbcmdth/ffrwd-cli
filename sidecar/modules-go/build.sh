#!/bin/sh
# Builds every Go module here to a wasm32-wasip2 component in build/.
#
# The whole recipe, from a bare machine to a built component:
#
#   1. Go 1.24.4 or newer, from https://go.dev/dl/. `go version`.
#
#   2. TinyGo 0.41.1, from
#      https://github.com/tinygo-org/tinygo/releases/download/v0.41.1/tinygo0.41.1.windows-amd64.zip
#      (or the linux-amd64 / darwin-arm64 asset of the same tag), unzipped
#      anywhere. Add its bin/ to PATH, or set TINYGO to the binary.
#      Mainline Go speaks wasip1 only; wasip2 and the component model are
#      TinyGo's.
#
#   3. wasm-tools 1.258.0, from
#      https://github.com/bytecodealliance/wasm-tools/releases/download/v1.258.0/wasm-tools-1.258.0-x86_64-windows.zip
#      on PATH. TinyGo shells out to it to turn the linked core module into a
#      component; it does not bundle one.
#
#   4. binaryen 132, from
#      https://github.com/WebAssembly/binaryen/releases/download/version_132/binaryen-version_132-x86_64-windows.tar.gz
#      with its bin/ on PATH, for wasm-opt. The Windows TinyGo archive ships
#      no wasm-opt and refuses to build without one; the Linux archive does
#      ship it, so this step is Windows-only.
#
#   5. wit-bindgen-go, for the generated bindings under internal/. They are
#      checked in, so this is needed only to regenerate them after a change
#      to ../wit/av.wit:
#
#        go install go.bytecodealliance.org/cmd/wit-bindgen-go@v0.7.0
#        wit-bindgen-go generate --world window-module --out internal \
#          --package-root github.com/imbcmdth/ffrwd/sidecar/modules-go/internal \
#          ../wit
#
#      Pin go.bytecodealliance.org/cm to v0.3.0, which is what that generator
#      emits imports for. `go mod tidy` resolves it to v0.7.0, where the root
#      package no longer exists, and the build stops on a missing package.
#
#   6. git clone the repo, then from sidecar/modules-go: sh build.sh
#
# Set TINYGO to build with a TinyGo not on PATH.
set -eu

here=$(dirname "$0")
cd "$here"

tinygo=${TINYGO:-tinygo}
out=build
wit=$out/wit

# The world TinyGo encodes around is assembled rather than checked in: the
# module interfaces come from ../wit/av.wit, the single copy the Rust modules
# also build against, and the wasi imports come from whatever TinyGo ships.
tinygo_root=$("$tinygo" env TINYGOROOT)
wasi_wit=$tinygo_root/lib/wasi-cli/wit

rm -rf "$wit"
mkdir -p "$wit/deps/ffrwd-av" "$wit/deps/cli"
cp wit/world.wit "$wit/world.wit"
cp ../wit/av.wit "$wit/deps/ffrwd-av/av.wit"
cp "$wasi_wit"/*.wit "$wit/deps/cli/"
cp -R "$wasi_wit"/deps/* "$wit/deps/"

# -gc=leaking and -scheduler=none are not tuning. Without both, the host reads
# corrupt output: with any collecting GC the frames and timestamps a call
# returns are overwritten before the host lifts them, whatever the
# optimisation level, and holding the Go values in package-level variables
# does not save them. Small frames hide it - 8x8 survives hundreds of calls -
# so it shows up first at a real frame size. Leaking never reuses memory, so
# nothing the host has yet to read can be handed to something else.
#
# What that costs: an input frame is never freed either, which caps these
# modules at roughly a thousand 640x480 frames before the instance runs out
# of memory. They are a spike, not something to ship a video through.
#
# -opt is left at its default: -scheduler=none does not build at -opt=0.
for module in invert-go window3-go; do
    # The host finds a module by file name, and its fleet uses underscores.
    artifact=$out/$(echo "$module" | tr - _).wasm
    echo "building $artifact"
    "$tinygo" build -target=wasip2 -scheduler=none -gc=leaking -o "$artifact" \
        --wit-package "$wit" --wit-world window-module-go \
        "./$module"
done
