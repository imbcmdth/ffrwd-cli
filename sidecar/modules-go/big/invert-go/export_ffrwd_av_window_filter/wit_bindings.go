// The componentize-go road's invert-go implementation. componentize-go
// generated this file's shape (package name, function signatures) from
// ../../wit against the window-module-go world; the bodies below are
// hand-written, same behaviour as ../../../invert-go/main.go (the TinyGo
// road): every colour byte complemented, alpha untouched, window 1 stride 1.
package export_ffrwd_av_window_filter

import (
	"runtime/debug"
	"strings"

	witTypes "go.bytecodealliance.org/pkg/wit/types"
	"wit_component/ffrwd_av_types"
	"wit_component/ffrwd_av_window_filter"
)

// EXPERIMENT: a real frame's `cabi_realloc` - the guest allocator the host
// calls to get space for an incoming argument, running before this module's
// own Process ever sees a frame - can itself trigger a GC assist, and Go's
// mark-termination phase calls `time.now`, which is a wasi import.
// Calling an import from inside `cabi_realloc` is a Component Model
// canonical-ABI reentrancy violation, and wasmtime traps it outright: "cannot
// leave component instance". That happens whether or not this module
// allocates anything of its own. Turning the collector off here is the same
// trade TinyGo's `-gc=leaking` made - nothing is ever freed - to find out
// whether it is the collector running at all, rather than this module's own
// allocation pattern, that the trap depends on.
func init() {
	debug.SetGCPercent(-1)
}

const paramsSchema = `{"type":"object","properties":{},"additionalProperties":false}`

func Describe() ffrwd_av_window_filter.WindowMeta {
	return ffrwd_av_window_filter.WindowMeta{
		Meta: ffrwd_av_types.Meta{
			Name:          "invert-go",
			Version:       "0.1.0",
			ParamsSchema:  paramsSchema,
			RowsSchema:    "",
			PixelFormats:  []string{"rgba"},
			SampleFormats: []string{},
			SampleRates:   []uint32{},
			ChannelCounts: []uint32{},
			RowsLanguage:  []string{},
		},
		Window:       1,
		Stride:       1,
		Pure:         true,
		OneToOne:     true,
		ReadsRows:    false,
		ForwardsRows: false,
		Inputs:       1,
	}
}

func validateParams(params string) witTypes.Result[witTypes.Unit, string] {
	switch strings.TrimSpace(params) {
	case "", "{}":
		return witTypes.Ok[witTypes.Unit, string](witTypes.Unit{})
	default:
		return witTypes.Err[witTypes.Unit, string]("invert-go takes no params, got: " + params)
	}
}

func Init(_ ffrwd_av_types.Format, _ ffrwd_av_types.StreamInfo, params string) witTypes.Result[witTypes.Unit, string] {
	return validateParams(params)
}

func SetParams(params string) witTypes.Result[witTypes.Unit, string] {
	return validateParams(params)
}

// Writes the complement of every colour byte into out, alpha copied through.
func invert(out, in []byte) {
	for i := 0; i+3 < len(in); i += 4 {
		out[i] = 255 - in[i]
		out[i+1] = 255 - in[i+1]
		out[i+2] = 255 - in[i+2]
		out[i+3] = in[i+3]
	}
}

var noRows = []string{}

// Reused across calls so a steady stream of same-sized frames does not keep
// asking the allocator for a fresh buffer every call - an experiment against
// mainline Go's collector, not a correctness requirement the way it was for
// the TinyGo road's leaking collector (pinning already keeps a freshly
// allocated buffer alive for as long as the host needs it here).
var outBufs [][]byte

func Process(frames []ffrwd_av_window_filter.InFrame, _ []string, _ bool) ffrwd_av_window_filter.Processed {
	if cap(outBufs) < len(frames) {
		outBufs = make([][]byte, len(frames))
	}
	outBufs = outBufs[:len(frames)]
	out := make([]ffrwd_av_window_filter.OutFrame, len(frames))
	for i := range frames {
		pixels := frames[i].Frame
		if cap(outBufs[i]) < len(pixels) {
			outBufs[i] = make([]byte, len(pixels))
		}
		buf := outBufs[i][:len(pixels)]
		invert(buf, pixels)
		out[i] = ffrwd_av_window_filter.OutFrame{
			Pts:   frames[i].Pts,
			Frame: ffrwd_av_window_filter.MakeFramePayloadNew(buf),
			Rows:  noRows,
		}
	}
	return ffrwd_av_window_filter.Processed{
		Frames:   out,
		Trailing: noRows,
	}
}
