// A rolling window of three frames, advancing one frame at a time: window 3,
// stride 1. Frames pass through untouched, and each call writes a row on
// `rows` naming how many frames it saw and the span of timestamps in it.
//
// It exists to drive the shape a buffering module needs - a window wider than
// its stride, so a call sees the frames around the one it consumes.
//
// One output per call, at the timestamp of the frame that call consumed: the
// window's oldest, since that is the one the stride drains. The last call
// carries whatever the strides left buffered, the whole stream when it was
// shorter than a window, and every one of them leaves at its own timestamp.
package main

import (
	"strconv"
	"strings"

	"go.bytecodealliance.org/cm"

	"github.com/imbcmdth/ffrwd/sidecar/modules-go/internal/ffrwd/av/node"
	nodetick "github.com/imbcmdth/ffrwd/sidecar/modules-go/internal/ffrwd/av/node-tick"
	nodetypes "github.com/imbcmdth/ffrwd/sidecar/modules-go/internal/ffrwd/av/node-types"
	"github.com/imbcmdth/ffrwd/sidecar/modules-go/internal/ffrwd/av/types"
)

const paramsSchema = `{"type":"object","properties":{},"additionalProperties":false}`
const rowsSchema = `{"type":"object","properties":{"saw":{"type":"integer"},"first":{"type":"integer"},"last":{"type":"integer"}},"required":["saw","first","last"]}`

// Frames one call sees, and how many of them it consumes.
const window = 3
const stride = 1

type shapeResult = cm.Result[node.NodeShapeShape, node.NodeShape, string]
type emittedResult = cm.Result[node.EmittedShape, node.Emitted, string]

// Validates that params is empty or `{}`; window3-go takes no parameters.
func validateParams(params string) cm.Result[string, struct{}, string] {
	switch strings.TrimSpace(params) {
	case "", "{}":
		return cm.OK[cm.Result[string, struct{}, string]](struct{}{})
	default:
		return cm.Err[cm.Result[string, struct{}, string]]("window3-go takes no params, got: " + params)
	}
}

func describe() types.Meta {
	return types.Meta{
		Name:          "window3-go",
		Version:       "0.2.0",
		ParamsSchema:  paramsSchema,
		PixelFormats:  cm.ToList([]string{}),
		SampleFormats: cm.ToList([]string{}),
		SampleRates:   cm.ToList([]uint32{}),
		ChannelCounts: cm.ToList([]uint32{}),
		RowsLanguage:  cm.ToList([]string{}),
	}
}

func shape(params string, _ cm.List[node.Binding]) shapeResult {
	if bad := validateParams(params); bad.IsErr() {
		return cm.Err[shapeResult](*bad.Err())
	}
	v := nodetypes.InputPort{
		Name:     "v",
		Kind:     nodetypes.PortKindVideo,
		Required: true,
		Pairing:  nodetypes.PairingLockstep(),
		Rows:     nodetypes.RowsUseIgnore,
		Window:   window,
		Stride:   stride,
		Accepts: nodetypes.Accepts{
			PixelFormats:  cm.ToList([]string{"rgba"}),
			SampleFormats: cm.ToList([]string{}),
			SampleRates:   cm.ToList([]uint32{}),
			ChannelCounts: cm.ToList([]uint32{}),
			Codecs:        cm.ToList([]string{}),
			Wants:         types.WantsAll,
		},
	}
	pictures := nodetypes.OutputPort{
		Name:   "v",
		Kind:   nodetypes.PortKindVideo,
		Format: cm.Some(nodetypes.OutputFormatLike(nodetypes.LikeInput{Port: "v"})),
	}
	rows := nodetypes.OutputPort{
		Name:   "rows",
		Kind:   nodetypes.PortKindData,
		Format: cm.Some(nodetypes.OutputFormatData("json")),
		Schema: cm.Some(rowsSchema),
	}
	return cm.OK[shapeResult](node.NodeShape{
		Inputs:   cm.ToList([]nodetypes.InputPort{v}),
		Outputs:  cm.ToList([]nodetypes.OutputPort{pictures, rows}),
		Clock:    nodetypes.ClockInput("v"),
		Pure:     true,
		OneToOne: true,
		Bounded:  true,
		Relation: cm.ToList([]string{}),
	})
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

// The stream `v` is bound to, from init.
var stream uint32

// What the call in flight hands back, held until the next call replaces it:
// the host reads it after `process` returns, through component-model
// pointers no collector keeps alive.
var (
	liveItems []node.Emission
	liveRows  [][]byte
	noRows    = []string{}
)

func process(rep cm.Rep) emittedResult {
	tick := nodetick.Tick(cm.Resource(rep))
	// The tick arrives as a borrowed handle the bindings do not release,
	// and a call that returns holding one is refused.
	defer tick.ResourceDrop()
	in := tick.Frames(stream).Slice()
	liveItems = liveItems[:0]
	liveRows = liveRows[:0]
	if len(in) > 0 {
		consumed := in
		if !tick.Last() && len(in) >= stride {
			consumed = in[:stride]
		}
		note := []byte(row(len(in), in[0].Pts, in[len(in)-1].Pts))
		liveRows = append(liveRows, note)
		liveItems = append(liveItems, node.Emission{
			Port: "rows",
			Payload: node.PayloadMessage(nodetypes.Message{
				Pts:  consumed[0].Pts,
				Data: cm.ToList(note),
			}),
		})
		for _, frame := range consumed {
			liveItems = append(liveItems, node.Emission{
				Port: "v",
				Payload: node.PayloadSame(node.SameFrame{
					Pts:      frame.Pts,
					Duration: frame.Duration,
					ID:       stream,
					Index:    frame.Index,
				}),
			})
		}
	}
	return cm.OK[emittedResult](node.Emitted{
		Items: cm.ToList(liveItems),
		Rows:  cm.ToList(noRows),
	})
}

func init() {
	node.Exports.Describe = describe
	node.Exports.Shape = shape
	node.Exports.Init = func(bound cm.List[node.BoundStream], _ cm.List[string], params string) cm.Result[string, struct{}, string] {
		for _, b := range bound.Slice() {
			if b.Port == "v" {
				stream = b.ID
			}
		}
		return validateParams(params)
	}
	node.Exports.SetParams = validateParams
	node.Exports.Process = process
}

// Required by the toolchain; a component's work happens in its exports.
func main() {}
