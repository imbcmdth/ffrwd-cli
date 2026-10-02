// Every colour byte replaced by its complement, alpha untouched: the Go twin
// of the fleet's `invert`, as a node. One rgba picture in, the clock, and
// one out in its format, one frame for each frame at its own time.
package main

import (
	"strings"

	"go.bytecodealliance.org/cm"

	"github.com/imbcmdth/ffrwd/sidecar/modules-go/internal/ffrwd/av/node"
	nodetick "github.com/imbcmdth/ffrwd/sidecar/modules-go/internal/ffrwd/av/node-tick"
	nodetypes "github.com/imbcmdth/ffrwd/sidecar/modules-go/internal/ffrwd/av/node-types"
	"github.com/imbcmdth/ffrwd/sidecar/modules-go/internal/ffrwd/av/types"
)

const paramsSchema = `{"type":"object","properties":{},"additionalProperties":false}`

type shapeResult = cm.Result[node.NodeShapeShape, node.NodeShape, string]
type emittedResult = cm.Result[node.EmittedShape, node.Emitted, string]

// Validates that params is empty or `{}`; invert-go takes no parameters.
func validateParams(params string) cm.Result[string, struct{}, string] {
	switch strings.TrimSpace(params) {
	case "", "{}":
		return cm.OK[cm.Result[string, struct{}, string]](struct{}{})
	default:
		return cm.Err[cm.Result[string, struct{}, string]]("invert-go takes no params, got: " + params)
	}
}

func describe() types.Meta {
	return types.Meta{
		Name:          "invert-go",
		Version:       "0.2.0",
		ParamsSchema:  paramsSchema,
		PixelFormats:  cm.ToList([]string{}),
		SampleFormats: cm.ToList([]string{}),
		SampleRates:   cm.ToList([]uint32{}),
		ChannelCounts: cm.ToList([]uint32{}),
		RowsLanguage:  cm.ToList([]string{}),
	}
}

func shape(params string, _ cm.List[string]) shapeResult {
	if bad := validateParams(params); bad.IsErr() {
		return cm.Err[shapeResult](*bad.Err())
	}
	v := nodetypes.InputPort{
		Name:     "v",
		Kind:     nodetypes.PortKindVideo,
		Required: true,
		Pairing:  nodetypes.PairingLockstep(),
		Rows:     nodetypes.RowsUseIgnore,
		Window:   1,
		Stride:   1,
		Accepts: nodetypes.Accepts{
			PixelFormats:  cm.ToList([]string{"rgba"}),
			SampleFormats: cm.ToList([]string{}),
			SampleRates:   cm.ToList([]uint32{}),
			ChannelCounts: cm.ToList([]uint32{}),
			Codecs:        cm.ToList([]string{}),
			Wants:         types.WantsAll,
		},
	}
	out := nodetypes.OutputPort{
		Name:   "v",
		Kind:   nodetypes.PortKindVideo,
		Format: cm.Some(nodetypes.OutputFormatLike(nodetypes.LikeInput{Port: "v"})),
	}
	return cm.OK[shapeResult](node.NodeShape{
		Inputs:   cm.ToList([]nodetypes.InputPort{v}),
		Outputs:  cm.ToList([]nodetypes.OutputPort{out}),
		Clock:    nodetypes.ClockInput("v"),
		Pure:     true,
		OneToOne: true,
		Bounded:  true,
		Relation: cm.ToList([]string{}),
	})
}

// The stream `v` is bound to, from init.
var stream uint32

// What the call in flight hands back. The host reads it after `process`
// returns, through component-model pointers no collector keeps alive, so the
// Go values behind them stay here until the next call replaces them.
var (
	outPixels [][]byte
	outItems  []node.Emission
	noRows    = []string{}
)

// Writes the complement of every colour byte into out, alpha copied through.
func invert(out, in []byte) {
	for i := 0; i+3 < len(in); i += 4 {
		out[i] = 255 - in[i]
		out[i+1] = 255 - in[i+1]
		out[i+2] = 255 - in[i+2]
		out[i+3] = in[i+3]
	}
}

func process(rep cm.Rep) emittedResult {
	tick := nodetick.Tick(cm.Resource(rep))
	// The tick arrives as a borrowed handle the bindings do not release,
	// and a call that returns holding one is refused.
	defer tick.ResourceDrop()
	frames := tick.Frames(stream).Slice()
	outPixels = outPixels[:0]
	outItems = outItems[:0]
	for _, frame := range frames {
		in := tick.Fetch(stream, frame.Index).Slice()
		out := make([]byte, len(in))
		invert(out, in)
		outPixels = append(outPixels, out)
		outItems = append(outItems, node.Emission{
			Port: "v",
			Payload: node.PayloadFrame(types.RawFrame{
				Pts:      frame.Pts,
				Duration: frame.Duration,
				Data:     cm.ToList(out),
			}),
		})
	}
	return cm.OK[emittedResult](node.Emitted{
		Items: cm.ToList(outItems),
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
