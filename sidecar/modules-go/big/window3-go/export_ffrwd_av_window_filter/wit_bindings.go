// The componentize-go road's window3-go implementation. componentize-go
// generated this file's shape from ../../wit against the window-module-go
// world; the bodies below are hand-written, same behaviour as
// ../../../window3-go/main.go (the TinyGo road): a rolling window of three
// frames advancing one at a time, frames passed through untouched, each
// carrying a row naming how many frames its call saw and the span of
// timestamps in that call.
package export_ffrwd_av_window_filter

import (
	"runtime/debug"
	"strconv"
	"strings"

	witTypes "go.bytecodealliance.org/pkg/wit/types"
	"wit_component/ffrwd_av_types"
	"wit_component/ffrwd_av_window_filter"
)

// See invert-go's copy of this comment: the guest `cabi_realloc` the host
// calls to place an incoming frame's bytes runs ahead of this module's own
// Process, and a GC cycle landing inside it is a Component Model canonical-ABI
// reentrancy violation (a WASI import call from inside cabi_realloc) that
// wasmtime traps outright. Off is the same trade TinyGo's `-gc=leaking` made.
func init() {
	debug.SetGCPercent(-1)
}

const paramsSchema = `{"type":"object","properties":{},"additionalProperties":false}`
const rowsSchema = `{"type":"object","properties":{"saw":{"type":"integer"},"first":{"type":"integer"},"last":{"type":"integer"}},"required":["saw","first","last"],"additionalProperties":false}`

const window = 3
const stride = 1

func Describe() ffrwd_av_window_filter.WindowMeta {
	return ffrwd_av_window_filter.WindowMeta{
		Meta: ffrwd_av_types.Meta{
			Name:          "window3-go",
			Version:       "0.1.0",
			ParamsSchema:  paramsSchema,
			RowsSchema:    rowsSchema,
			PixelFormats:  []string{"rgba"},
			SampleFormats: []string{},
			SampleRates:   []uint32{},
			ChannelCounts: []uint32{},
			RowsLanguage:  []string{},
		},
		Window:       window,
		Stride:       stride,
		Pure:         true,
		OneToOne:     true,
		ReadsRows:    false,
		ForwardsRows: true,
		Inputs:       1,
	}
}

func validateParams(params string) witTypes.Result[witTypes.Unit, string] {
	switch strings.TrimSpace(params) {
	case "", "{}":
		return witTypes.Ok[witTypes.Unit, string](witTypes.Unit{})
	default:
		return witTypes.Err[witTypes.Unit, string]("window3-go takes no params, got: " + params)
	}
}

func Init(_ ffrwd_av_types.Format, _ ffrwd_av_types.StreamInfo, params string) witTypes.Result[witTypes.Unit, string] {
	return validateParams(params)
}

func SetParams(params string) witTypes.Result[witTypes.Unit, string] {
	return validateParams(params)
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
func consumed(in []ffrwd_av_window_filter.InFrame, last bool) []ffrwd_av_window_filter.InFrame {
	if last || len(in) < stride {
		return in
	}
	return in[:stride]
}

func Process(frames []ffrwd_av_window_filter.InFrame, trailing []string, last bool) ffrwd_av_window_filter.Processed {
	var outFrames []ffrwd_av_window_filter.OutFrame

	if len(frames) > 0 {
		note := []string{row(len(frames), frames[0].Pts, frames[len(frames)-1].Pts)}
		for _, frame := range consumed(frames, last) {
			rows := append(append([]string{}, note...), frame.Rows...)
			note = nil
			outFrames = append(outFrames, ffrwd_av_window_filter.OutFrame{
				Pts: frame.Pts,
				// The bytes this call was handed at this timestamp; nothing
				// is copied out.
				Frame: ffrwd_av_window_filter.MakeFramePayloadSame(),
				Rows:  rows,
			})
		}
	}

	return ffrwd_av_window_filter.Processed{
		Frames:   outFrames,
		Trailing: append([]string{}, trailing...),
	}
}
