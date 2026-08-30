// Every colour byte replaced by its complement, alpha untouched: the Go twin
// of the fleet's `invert`, byte for byte over the same frames.
//
// It exports `window-filter` at window 1, stride 1, which is what a per-frame
// video filter is in that interface's terms. The older per-frame `filter`
// would say the same thing; this one is what a windowed Go module will use,
// so it is what this proves.
package main

import (
	"strings"

	"go.bytecodealliance.org/cm"

	"github.com/imbcmdth/ffrwd/sidecar/modules-go/internal/ffrwd/av/types"
	windowfilter "github.com/imbcmdth/ffrwd/sidecar/modules-go/internal/ffrwd/av/window-filter"
)

const paramsSchema = `{"type":"object","properties":{},"additionalProperties":false}`

// Validates that params is empty or `{}`; invert-go takes no parameters.
func validateParams(params string) cm.Result[string, struct{}, string] {
	switch strings.TrimSpace(params) {
	case "", "{}":
		return cm.OK[cm.Result[string, struct{}, string]](struct{}{})
	default:
		return cm.Err[cm.Result[string, struct{}, string]]("invert-go takes no params, got: " + params)
	}
}

func describe() windowfilter.WindowMeta {
	return windowfilter.WindowMeta{
		Meta: types.Meta{
			Name:         "invert-go",
			Version:      "0.1.0",
			ParamsSchema: paramsSchema,
			RowsSchema:   "",
			PixelFormats: cm.ToList([]string{"rgba"}),
			// Not an audio module, so it names no sample formats.
			SampleFormats: cm.ToList([]string{}),
			SampleRates:   cm.ToList([]uint32{}),
			ChannelCounts: cm.ToList([]uint32{}),
			RowsLanguage:  cm.ToList([]string{}),
		},
		Window: 1,
		Stride: 1,
		// Nothing carries over between calls.
		Pure:     true,
		OneToOne: true,
		// No rows are read, and none arriving are passed on.
		ReadsRows:    false,
		ForwardsRows: false,
		Inputs:       1,
	}
}

// What the call in flight hands back, allocated on the first call and reused
// by every call after it.
//
// Two things force this. The host reads the output after `process` returns,
// through component-model pointers the collector does not keep memory alive
// for - and any allocation inside the call is what lets the collector run at
// all. Allocating once and reusing avoids both: package-level buffers are
// permanent roots, and a call that allocates nothing cannot collect. The
// frame size is fixed for the life of an instance, so one call's buffer fits
// every later call's frame.
var (
	outPixels []byte
	outFrames []windowfilter.OutFrame
	noRows    = []string{}
)

// Writes the complement of every colour byte into out, alpha copied through.
// Reading and writing separate buffers is what lets the output survive the
// call; complementing the host's buffer in place would not.
func invert(out, in []byte) {
	for i := 0; i+3 < len(in); i += 4 {
		out[i] = 255 - in[i]
		out[i+1] = 255 - in[i+1]
		out[i+2] = 255 - in[i+2]
		out[i+3] = in[i+3]
	}
}

// Rows arriving with a frame stop here, as they do in the Rust invert, so
// nothing an upstream module emitted leaves on this module's output.
func process(frames cm.List[windowfilter.InFrame], _ cm.List[string], _ bool) windowfilter.Processed {
	in := frames.Slice()

	total := 0
	for i := range in {
		total += int(in[i].Frame.Len())
	}
	if cap(outPixels) < total {
		outPixels = make([]byte, total)
	}
	if cap(outFrames) < len(in) {
		outFrames = make([]windowfilter.OutFrame, len(in))
	}
	outPixels, outFrames = outPixels[:total], outFrames[:len(in)]

	at := 0
	for i := range in {
		pixels := in[i].Frame.Slice()
		out := outPixels[at : at+len(pixels)]
		invert(out, pixels)
		at += len(pixels)
		outFrames[i] = windowfilter.OutFrame{
			Pts:   in[i].Pts,
			Frame: windowfilter.FramePayloadNew(cm.ToList(out)),
			Rows:  cm.ToList(noRows),
		}
	}

	return windowfilter.Processed{
		Frames:   cm.ToList(outFrames),
		Trailing: cm.ToList(noRows),
	}
}

func init() {
	windowfilter.Exports.Describe = describe
	windowfilter.Exports.Init = func(_ types.Format, _ types.StreamInfo, params string) cm.Result[string, struct{}, string] {
		return validateParams(params)
	}
	windowfilter.Exports.SetParams = validateParams
	windowfilter.Exports.Process = process
}

// Required by the toolchain; a component's work happens in its exports.
func main() {}
