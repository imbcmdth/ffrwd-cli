#!/bin/sh
# Builds every Go module under big/ to a wasm32-wasip2 component in
# build-big/, using mainline Go and componentize-go instead of TinyGo. Same
# world, same wit, same module behaviour as build.sh; a different toolchain
# from the ground up, because mainline Go has no wasip2 target of its own.
#
# The whole recipe, from a bare machine to a built component:
#
#   1. Go 1.25.5 or newer, from https://go.dev/dl/. `go version`. This is a
#      separate requirement from build.sh's TinyGo: mainline Go's `go:wasmexport`
#      directive (the reactor-mode export mechanism this whole road rests on)
#      only exists from 1.24 and only stabilised at 1.25.5.
#
#   2. componentize-go, from https://github.com/bytecodealliance/componentize-go.
#      `go install github.com/bytecodealliance/componentize-go@latest` builds a
#      Go wrapper that, on first run, downloads the real (Rust-built) binary
#      from the matching GitHub release - but on Windows it asks for
#      `componentize-go-windows-amd64.tar.gz`, an asset that release does not
#      publish (Windows ships a `.zip`; every other platform gets a `.tar.gz`,
#      and the wrapper does not check), so that first run 404s. The workaround:
#      download `componentize-go-windows-amd64.zip` from the release tagged
#      with the version `go install` resolved (v0.4.1 as of this writing, from
#      https://github.com/bytecodealliance/componentize-go/releases) directly,
#      and put the extracted `componentize-go.exe` on PATH ahead of anything
#      else named that - the wrapper is never invoked once the real binary
#      resolves first.
#
#   3. Set COMPONENTIZE_GO to the binary from step 2 if it is not on PATH
#      under the name `componentize-go`. Set GO_BIG to the Go from step 1 if
#      it is not the machine's default `go` - this is componentize-go's own
#      `--go` flag underneath, not PATH, because build.sh's TinyGo wants a Go
#      between 1.19 and 1.26 for itself, on the same PATH this script also
#      needs TinyGo's TINYGOROOT from (see below) - the two roads can disagree
#      about what plain `go` on PATH should be, so this one takes its own Go
#      by path instead of asking PATH to satisfy both at once.
#
#   4. git clone the repo, then from sidecar/modules-go: sh build-big.sh
#
# What componentize-go actually does with a Go module: `go build` for
# GOOS=wasip1 GOARCH=wasm in reactor mode (an empty `main`, so
# `//go:wasmexport` functions stay callable after `init()` runs once),
# embedding the WIT world named on the command line into the resulting core
# module, then wrapping that module into a wasip2 component with the
# wasi_snapshot_preview1 adapter. No TinyGo anywhere in the chain.
#
# build-big/wit/ is assembled the same way build.sh assembles build/wit/, from
# the same source: wit/world.wit, ../worlds/0.10.0/av.wit, and the wasi imports from
# wherever a wasi:cli/imports@0.2.0 wit tree is found (TinyGo's copy, since one
# is a build.sh prerequisite already, and the wit text itself is the standard
# unversioned-content wasi:cli@0.2.0 interfaces - nothing TinyGo-specific rides
# along). Both roads target the same world.
#
# Each module directory under big/ (invert-go/, window3-go/) holds its own
# go.mod: componentize-go builds one component per Go module, from bindings
# generated straight into that directory (`componentize-go bindings
# --generate-stubs`), so the two modules do not share one build the way the
# TinyGo road's single go.mod does. The generated `export_ffrwd_av_window_filter`
# package is what a change to ../worlds/0.10.0/av.wit regenerates; the hand-written
# bodies inside it (same behaviour as the sibling TinyGo module) are what
# survives that regeneration only if copied back in - componentize-go
# overwrites the whole file, unlike wit-bindgen-go's split between generated
# and hand-written packages.
set -eu

here=$(dirname "$0")
cd "$here"

componentize_go=${COMPONENTIZE_GO:-componentize-go}
# componentize-go's own `--go` flag, not PATH: TinyGo needs a Go between 1.19
# and 1.26 on PATH for itself, and componentize-go needs 1.25.5+, so the two
# roads' prerequisites can disagree about what plain `go` on PATH should be.
go_bin=${GO_BIG:-go}
out=build-big
wit=$out/wit

# Same wasi:cli/imports@0.2.0 wit deps build.sh copies from TinyGo's own
# vendored copy - the content is the standard wasi:cli@0.2.0 interfaces,
# nothing TinyGo-specific.
tinygo=${TINYGO:-tinygo}
tinygo_root=$("$tinygo" env TINYGOROOT)
wasi_wit=$tinygo_root/lib/wasi-cli/wit

rm -rf "$wit"
mkdir -p "$wit/deps/ffrwd-av" "$wit/deps/cli"
cp wit/world.wit "$wit/world.wit"
cp ../worlds/0.10.0/av.wit "$wit/deps/ffrwd-av/av.wit"
cp "$wasi_wit"/*.wit "$wit/deps/cli/"
cp -R "$wasi_wit"/deps/* "$wit/deps/"

mkdir -p "$out"
for module in invert-go window3-go; do
    artifact=$out/$(echo "$module" | tr - _).wasm
    echo "building $artifact"
    (cd "big/$module" && "$go_bin" mod tidy >/dev/null && \
        "$componentize_go" -d "../../$wit" -w window-module-go build \
            --go "$go_bin" -o "../../$artifact")
done
