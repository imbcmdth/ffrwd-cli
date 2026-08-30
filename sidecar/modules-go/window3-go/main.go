// A rolling window of three frames, advancing one frame at a time: window 3,
// stride 1. Frames pass through untouched, and each carries away a row naming
// how many frames its call saw and the span of timestamps in that call.
//
// It exists to drive the shape a buffering module needs - a window wider than
// its stride, so a call sees the frames around the one it consumes - which no
// module in the fleet declares. The host cuts these windows already; nothing
// had asked it to.
//
// One output per call, at the timestamp of the frame that call consumed: the
// window's oldest, since that is the one the stride drains. The final call
// carries whatever the strides left buffered - window minus one frames, or
// the whole stream when it was shorter than a window - and every one of them
// leaves at its own timestamp. So the frames leaving are the frames that
// arrived, each once, in order.
package main

import (
	"strconv"
	"strings"

	"go.bytecodealliance.org/cm"

	"github.com/imbcmdth/ffrwd/sidecar/modules-go/internal/ffrwd/av/types"
	windowfilter "github.com/imbcmdth/ffrwd/sidecar/modules-go/internal/ffrwd/av/window-filter"
)

const paramsSchema = `{"type":"object","properties":{},"additionalProperties":false}`
const rowsSchema = `{"type":"object","properties":{"saw":{"type":"integer"},"first":{"type":"integer"},"last":{"type":"integer"}},"required":["saw","first","last"],"additionalProperties":false}`

// Frames one call sees, and how many of them it consumes.
const window = 3
const stride = 1

// Validates that params is empty or `{}`; window3-go takes no parameters.
func validateParams(params string) cm.Result[string, struct{}, string] {
	switch strings.TrimSpace(params) {
	case "", "{}":
		return cm.OK[cm.Result[string, struct{}, string]](struct{}{})
	default:
		return cm.Err[cm.Result[string, struct{}, string]]("window3-go takes no params, got: " + params)
	}
}

func describe() windowfilter.WindowMeta {
	return windowfilter.WindowMeta{
		Meta: types.Meta{
			Name:          "window3-go",
			Version:       "0.1.0",
			ParamsSchema:  paramsSchema,
			RowsSchema:    rowsSchema,
			PixelFormats:  cm.ToList([]string{"rgba"}),
			SampleFormats: cm.ToList([]string{}),
			SampleRates:   cm.ToList([]uint32{}),
			ChannelCounts: cm.ToList([]uint32{}),
			RowsLanguage:  cm.ToList([]string{}),
		},
		Window: window,
		Stride: stride,
		// Every call depends only on the frames it was handed.
		Pure: true,
		// One output per frame consumed, at that frame's own timestamp.
		OneToOne: true,
		// Incoming rows are passed on beside this module's own, not read.
		ReadsRows:    false,
		ForwardsRows: true,
		Inputs:       1,
	}
}

// The row one call reports: the frames it saw and the timestamps at the two
// ends of that window.
func row(saw int, first, last int64) string {
	var b strings.Builder
	b.WriteString(`{"saw":`)
	b.WriteString(strconv.Itoa(saw))
	b.WriteString(`,"first":`)
	b.WriteString(strconv.FormatInt(first, 10))
	b.WriteString(`,"last":`)
	b.WriteString(strconv.FormatInt(last, 10))
	b.WriteString(`}`)
	return b.String()
}

// The frames this call consumed: the leading stride of an ordinary window,
// and everything left on the final call.
func consumed(in []windowfilter.InFrame, last bool) []windowfilter.InFrame {
	if last || len(in) < stride {
		return in
	}
	return in[:stride]
}

// What the call in flight is handing back. Everything the host reads after
// `process` returns is reachable only through component-model pointers, and
// TinyGo's collector reuses memory those pointers alone keep alive - so the
// Go values behind them are held here until the next call replaces them.
var (
	liveFrames []windowfilter.OutFrame
	liveRows   [][]string
	liveTrail  []string
)

func process(frames cm.List[windowfilter.InFrame], trailing cm.List[string], last bool) windowfilter.Processed {
	in := frames.Slice()
	liveFrames = liveFrames[:0]
	liveRows = liveRows[:0]

	if len(in) > 0 {
		// The call's own row rides its first output, as a call carrying
		// several frames still describes one window.
		note := []string{row(len(in), in[0].Pts, in[len(in)-1].Pts)}
		for _, frame := range consumed(in, last) {
			rows := append(note, frame.Rows.Slice()...)
			note = nil
			liveRows = append(liveRows, rows)
			liveFrames = append(liveFrames, windowfilter.OutFrame{
				Pts: frame.Pts,
				// The bytes this call was handed at this timestamp; nothing
				// is copied out.
				Frame: windowfilter.FramePayloadSame(),
				Rows:  cm.ToList(rows),
			})
		}
	}

	liveTrail = append(liveTrail[:0], trailing.Slice()...)
	return windowfilter.Processed{
		Frames:   cm.ToList(liveFrames),
		Trailing: cm.ToList(liveTrail),
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
