# One hold group for a run-time lateral: plan

A run-time lateral's picture and sound keep the time relation they had
in the feed, whichever calls they land in.

## The problem

A held input's clock offset is fixed per connection, from the first
frame that arrives on it (`sidecar/ffrwd-wasm/src/hold.rs`, "A feed's
one offset, clock time less source time", fixed by the lead member's
first frame). A connection serves one node (`feeds.rs`, `FeedSpec`
carries the node, the port and its members), and feeders share a
connection only when they name one group of one module and are fed from
one FROM item (`docs/dialect.md`, feeders and run-time laterals). A
group is a name the module's own shape declares (`NODE-SHAPE.md`,
`pairing: hold`, `group`), so two modules cannot name one group.

So a lateral whose picture feeds `compose` and whose sound feeds
`switch` gets two connections and two offsets. Each is anchored on its
own first arrival: the sound at once, the picture at its first decodable
frame, later still after a mid-GOP join. The two drift by whatever the
arrivals happened to be, and the drift changes on every rejoin.
Measured by foundmeat on 2026-10-17 (`E:\projects\foundmeat\docs\SYNTHETIC-CLOCK.md`):
sound minus picture from -383 ms to +348 ms across connections, where
one frame at 60 fps is 17 ms.

blitz's `compose` taking the sound beside the picture, on the same group
(branch `compose-audio`, 2026-10-17), closes the case for that one
module, since the two streams then ride one connection. Any other pair
of modules fed by one lateral still drifts. This plan is the general
fix.

## The rule

A run-time lateral instance feeding held inputs of several calls shares
one connection among them, and that connection has one offset, anchored
once, from the first picture where the lateral carries one and from the
first sound otherwise. Every stream the instance writes is restamped
with that one offset. A feeder with no lateral behind it is as today.

What a query writes does not change: `compose(k.video[1],
ARRAY[m.video], ...)` and `switch(a => s.audio[1], feed_audio =>
m.audio, ...)` with `m` the lateral stay as they are. The compiler sees
that both feeders are fed from one lateral FROM item and wires them to
one connection; the lateral's instance writes one NUT, picture then
sound then rows, as it does for one module's group today.

## The work

1. **The compiler.** Where feeders are given their ports
   (`cli/ffrwd/lower.py`, the run-time lateral's columns, "Feeders of
   one group fed by its picture and its sound share one connection"),
   widen the sharing from "one module's group" to "one lateral
   instance": every feeder fed from the same lateral FROM item takes
   one port, written into each call's port parameter, and `compile`
   lists the connection once with every call reading it. A feeder of
   the same lateral in a call that holds its frames on the GPU and one
   that holds them in host memory still share the connection; the
   conformance per member stays per member. Refuse, as today, a lateral
   whose streams reach anything but a feeder.
2. **The sidecar's feeds.** `FeedSpec` serves one node today. Let a
   feed's members belong to several nodes: the listener is opened once,
   the NUT's streams are dealt to the members by position, and each
   member's frames go to its node's hold input. The picture's conversion
   and the sound's conformance are per member, as now.
3. **One offset.** The feed's offset is fixed once, on the lead member
   (the picture where there is one), and every member of the connection
   restamps with it. `hold.rs` already keeps one offset per feed; what
   changes is that the members it serves may belong to different nodes,
   so each node's hold reads the feed's offset rather than fixing its
   own. A member whose node ticks on a different clock than another's
   is refused at compile time: one connection, one clock.
4. **Linger and timeout.** `linger` and `timeout` stay per hold input,
   since the picture and the sound may be shown and heard for different
   spans after a drop; the offset is the connection's, the presence is
   the input's.
5. **Rows.** A data input whose `interval` names the group takes the
   lateral's rows on the same connection, as today; with the connection
   shared across calls, a data input of either call may read them.

## Acceptance

The flash-and-beep feed (1080p60 H.264 and AAC, a white flash and a
50 ms beep at the same instant each second) as a run-time lateral under
a synthetic clock (`color=c=black:s=16x16:r=60`, realtime), picture into
`compose` and sound into `switch`, with a real template, dropped and
rejoined three times including a mid-GOP join: sound minus picture
within one frame, plus or minus 17 ms, on every connection. The same
with picture and sound into two test-fleet modules of the sidecar, so
the test runs in CI with no package. And `compile` listing one
connection for the two calls.

## Not this plan

- The further 200 to 350 ms of sound lag foundmeat measured in its full
  query and not in the bare one. That is not the anchoring; it grows
  with the template, the cam laterals and presence. It is measured first
  under `compose-audio`, where the sound rides compose's own offset, and
  fixed where it is found: the clock held for compose's render while the
  sound is not, or a hold's `lead`.
- Mixing several inputs' sound and a per-input volume: `compose`'s, on
  its own audio output, later.
