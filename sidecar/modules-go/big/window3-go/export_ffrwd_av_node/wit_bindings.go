// The componentize-go road's window3-go. componentize-go generated this
// file's shape (package name, function signatures) from ../../wit against the
// node-module-go world; the bodies are hand-written, the same node as
// ../../../window3-go/main.go (the TinyGo road): window 3, stride 1, frames
// passed through, and a row a call on `rows`.
package export_ffrwd_av_node

import (
	"runtime/debug"
	"strconv"
	"strings"

	witTypes "go.bytecodealliance.org/pkg/wit/types"
	"wit_component/ffrwd_av_node"
	"wit_component/ffrwd_av_node_tick"
	"wit_component/ffrwd_av_node_types"
	"wit_component/ffrwd_av_types"
)

// See invert-go's: the collector stays off, as TinyGo's `-gc=leaking` does.
func init() {
	debug.SetGCPercent(-1)
}

const paramsSchema = `{"type":"object","properties":{},"additionalProperties":false}`
const rowsSchema = `{"type":"object","properties":{"saw":{"type":"integer"},"first":{"type":"integer"},"last":{"type":"integer"}},"required":["saw","first","last"]}`

// Frames one call sees, and how many of them it consumes.
const window = 3
const stride = 1

func Describe() ffrwd_av_types.Meta {
	return ffrwd_av_types.Meta{
		Name:          "window3-go",
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
		return witTypes.Err[witTypes.Unit, string]("window3-go takes no params, got: " + params)
	}
}

func Shape(params string, _ []string) witTypes.Result[ffrwd_av_node_types.NodeShape, string] {
	if bad := validateParams(params); bad.IsErr() {
		return witTypes.Err[ffrwd_av_node_types.NodeShape, string](bad.Err())
	}
	v := ffrwd_av_node_types.InputPort{
		Name:     "v",
		Kind:     ffrwd_av_node_types.PortKindVideo,
		Required: true,
		Pairing:  ffrwd_av_node_types.MakePairingLockstep(),
		Rows:     ffrwd_av_node_types.RowsUseIgnore,
		Window:   window,
		Stride:   stride,
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
	pictures := ffrwd_av_node_types.OutputPort{
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
	rows := ffrwd_av_node_types.OutputPort{
		Name:     "rows",
		Kind:     ffrwd_av_node_types.PortKindData,
		Format:   witTypes.Some(ffrwd_av_node_types.MakeOutputFormatData("json")),
		TimeBase: witTypes.None[ffrwd_av_types.Rational](),
		Schema:   witTypes.Some(rowsSchema),
		Row:      witTypes.None[uint32](),
	}
	return witTypes.Ok[ffrwd_av_node_types.NodeShape, string](ffrwd_av_node_types.NodeShape{
		Inputs:   []ffrwd_av_node_types.InputPort{v},
		Outputs:  []ffrwd_av_node_types.OutputPort{pictures, rows},
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

func Process(tick *ffrwd_av_node_tick.Tick) witTypes.Result[ffrwd_av_node.Emitted, string] {
	// The bindings leave the borrowed tick to the module, and a call that
	// returns holding one is refused.
	defer tick.Drop()
	in := tick.Frames(stream)
	items := []ffrwd_av_node.Emission{}
	if len(in) > 0 {
		consumed := in
		if !tick.Last() && len(in) >= stride {
			consumed = in[:stride]
		}
		items = append(items, ffrwd_av_node.Emission{
			Port: "rows",
			Payload: ffrwd_av_node.MakePayloadMessage(ffrwd_av_node_types.Message{
				Pts:  consumed[0].Pts,
				Data: []byte(row(len(in), in[0].Pts, in[len(in)-1].Pts)),
			}),
		})
		for _, frame := range consumed {
			items = append(items, ffrwd_av_node.Emission{
				Port: "v",
				Payload: ffrwd_av_node.MakePayloadSame(ffrwd_av_node.SameFrame{
					Pts:      frame.Pts,
					Duration: frame.Duration,
					Id:       stream,
					Index:    frame.Index,
				}),
			})
		}
	}
	return witTypes.Ok[ffrwd_av_node.Emitted, string](ffrwd_av_node.Emitted{
		Items: items,
		Rows:  []string{},
	})
}
