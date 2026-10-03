// The componentize-go road's invert-go. componentize-go generated this
// file's shape (package name, function signatures) from ../../wit against the
// node-module-go world; the bodies are hand-written, the same node as
// ../../../invert-go/main.go (the TinyGo road): every colour byte
// complemented, alpha untouched, one frame for each frame.
package export_ffrwd_av_node

import (
	"runtime/debug"
	"strings"

	witTypes "go.bytecodealliance.org/pkg/wit/types"
	"wit_component/ffrwd_av_node"
	"wit_component/ffrwd_av_node_tick"
	"wit_component/ffrwd_av_node_types"
	"wit_component/ffrwd_av_types"
)

// A real frame's `cabi_realloc` can trigger a GC assist, and Go's
// mark-termination phase calls `time.now`, a wasi import: calling an import
// from inside `cabi_realloc` is a reentrancy the host traps ("cannot leave
// component instance"). The collector stays off, the same trade TinyGo's
// `-gc=leaking` makes.
func init() {
	debug.SetGCPercent(-1)
}

const paramsSchema = `{"type":"object","properties":{},"additionalProperties":false}`

func Describe() ffrwd_av_types.Meta {
	return ffrwd_av_types.Meta{
		Name:          "invert-go",
		Version:       "0.2.0",
		ParamsSchema:  paramsSchema,
		PixelFormats:  []string{},
		SampleFormats: []string{},
		SampleRates:   []uint32{},
		ChannelCounts: []uint32{},
		RowsLanguage:  []string{},
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

func Shape(params string, _ []ffrwd_av_node_types.Binding) witTypes.Result[ffrwd_av_node_types.NodeShape, string] {
	if bad := validateParams(params); bad.IsErr() {
		return witTypes.Err[ffrwd_av_node_types.NodeShape, string](bad.Err())
	}
	v := ffrwd_av_node_types.InputPort{
		Name:     "v",
		Kind:     ffrwd_av_node_types.PortKindVideo,
		Required: true,
		Pairing:  ffrwd_av_node_types.MakePairingLockstep(),
		Rows:     ffrwd_av_node_types.RowsUseIgnore,
		Window:   1,
		Stride:   1,
		Accepts: ffrwd_av_node_types.Accepts{
			PixelFormats:  []string{"rgba"},
			SampleFormats: []string{},
			SampleRates:   []uint32{},
			ChannelCounts: []uint32{},
			Codecs:        []string{},
			Wants:         ffrwd_av_types.WantsAll,
			Like:          witTypes.None[string](),
		},
		Schema: witTypes.None[string](),
	}
	out := ffrwd_av_node_types.OutputPort{
		Name: "v",
		Kind: ffrwd_av_node_types.PortKindVideo,
		Format: witTypes.Some(ffrwd_av_node_types.MakeOutputFormatLike(ffrwd_av_node_types.LikeInput{
			Port:         "v",
			PixelFormat:  witTypes.None[string](),
			SampleFormat: witTypes.None[string](),
		})),
		TimeBase: witTypes.None[ffrwd_av_types.Rational](),
		Schema:   witTypes.None[string](),
		Row:      witTypes.None[uint32](),
	}
	return witTypes.Ok[ffrwd_av_node_types.NodeShape, string](ffrwd_av_node_types.NodeShape{
		Inputs:   []ffrwd_av_node_types.InputPort{v},
		Outputs:  []ffrwd_av_node_types.OutputPort{out},
		Clock:    ffrwd_av_node_types.MakeClockInput("v"),
		Pure:     true,
		OneToOne: true,
		Bounded:  true,
		Relation: []string{},
	})
}

// The stream `v` is bound to, from Init.
var stream uint32

func Init(bound []ffrwd_av_node_types.BoundStream, _ []string, params string) witTypes.Result[witTypes.Unit, string] {
	for _, b := range bound {
		if b.Port == "v" {
			stream = b.Id
		}
	}
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

func Process(tick *ffrwd_av_node_tick.Tick) witTypes.Result[ffrwd_av_node.Emitted, string] {
	// The bindings leave the borrowed tick to the module, and a call that
	// returns holding one is refused.
	defer tick.Drop()
	items := []ffrwd_av_node.Emission{}
	for _, frame := range tick.Frames(stream) {
		in := tick.Fetch(stream, frame.Index)
		out := make([]byte, len(in))
		invert(out, in)
		items = append(items, ffrwd_av_node.Emission{
			Port: "v",
			Payload: ffrwd_av_node.MakePayloadFrame(ffrwd_av_types.RawFrame{
				Pts:      frame.Pts,
				Duration: frame.Duration,
				Data:     out,
			}),
		})
	}
	return witTypes.Ok[ffrwd_av_node.Emitted, string](ffrwd_av_node.Emitted{
		Items: items,
		Rows:  []string{},
	})
}
