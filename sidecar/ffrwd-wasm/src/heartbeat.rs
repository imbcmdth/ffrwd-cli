//! Heartbeats on a data edge.
//!
//! A data stream is sparse: minutes may pass between two messages. ffmpeg
//! cannot open a NUT data input until its first packet arrives, and once
//! open it holds the other streams of a file back until the data stream's
//! next packet shows time has moved on. So a data edge also carries EMPTY
//! packets, zero bytes of payload, each saying "no message; time has reached
//! this pts". Every writer of a data edge in this host puts one out at start
//! and then one whenever time moves on with nothing written for
//! [`EVERY`] of programme time; every reader drops them before a module sees
//! them. ffmpeg copies them through like any other packet.

use ffrwd_wasm_runtime::runtime::TimeBase;

/// The most programme time a data edge goes without a packet while its
/// writer knows time is moving: a tenth of a second.
pub const EVERY: TimeBase = TimeBase { num: 1, den: 10 };

/// Whether a data packet is a heartbeat rather than a message.
pub fn is_heartbeat(data: &[u8]) -> bool {
    data.is_empty()
}

/// `pts` in `from` restated in `to`, rounded down, so a time converted is
/// never later than the one it came from.
pub fn rescale(pts: i64, from: TimeBase, to: TimeBase) -> i64 {
    let num = i128::from(pts) * i128::from(from.num) * i128::from(to.den);
    let den = i128::from(from.den) * i128::from(to.num);
    num.div_euclid(den.max(1)) as i64
}

/// One data output's timeline as written: the pts of its last packet,
/// message or heartbeat, in the output's own time base.
pub struct Beats {
    base: TimeBase,
    last: Option<i64>,
}

impl Beats {
    pub fn new(base: TimeBase) -> Beats {
        Beats { base, last: None }
    }

    /// The pts a message at `pts` is written at: its own, or the output's
    /// last where a heartbeat already went past it, so the output never
    /// steps back and no heartbeat it wrote is made untrue.
    pub fn place(&mut self, pts: i64) -> i64 {
        let at = self.last.map_or(pts, |last| pts.max(last));
        self.last = Some(at);
        at
    }

    /// A heartbeat that arrived at `pts` handed on: its pts where it moves
    /// the output on, None where the output has already written that far.
    pub fn pass(&mut self, pts: i64) -> Option<i64> {
        let ahead = self.last.is_none_or(|last| pts > last);
        if ahead {
            self.last = Some(pts);
        }
        ahead.then_some(pts)
    }

    /// The pts of the heartbeat due now that time has reached `now` in
    /// `clock`, or None: one is due when the output has written nothing yet,
    /// or nothing for [`EVERY`] of programme time, and never behind what it
    /// has written.
    pub fn due(&mut self, now: i64, clock: TimeBase) -> Option<i64> {
        let at = rescale(now, clock, self.base);
        let due = match self.last {
            // NUT carries no negative pts, and a stream that has not reached
            // zero has nothing to say yet.
            _ if at < 0 => false,
            None => true,
            Some(last) => at > last && rescale(at - last, self.base, EVERY) >= 1,
        };
        if due {
            self.last = Some(at);
        }
        due.then_some(at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MICROS: TimeBase = TimeBase {
        num: 1,
        den: 1_000_000,
    };
    const FRAMES: TimeBase = TimeBase { num: 1, den: 30 };

    #[test]
    fn a_time_is_restated_rounding_down() {
        assert_eq!(rescale(1, FRAMES, MICROS), 33_333);
        assert_eq!(rescale(3, FRAMES, MICROS), 100_000);
        assert_eq!(rescale(33_333, MICROS, FRAMES), 0);
    }

    #[test]
    fn a_heartbeat_is_due_first_and_then_every_tenth_of_a_second() {
        let mut beats = Beats::new(MICROS);
        let due: Vec<Option<i64>> = (0..8).map(|k| beats.due(k, FRAMES)).collect();
        assert_eq!(
            due,
            vec![
                Some(0),
                None,
                None,
                Some(100_000),
                None,
                None,
                Some(200_000),
                None
            ]
        );
    }

    #[test]
    fn a_message_counts_and_nothing_steps_back() {
        let mut beats = Beats::new(MICROS);
        assert_eq!(beats.place(150_000), 150_000);
        // Behind what was written, and then not yet a tenth past it.
        assert_eq!(beats.due(3, FRAMES), None);
        assert_eq!(beats.due(7, FRAMES), None);
        assert_eq!(beats.due(9, FRAMES), Some(300_000));
        // A message behind the heartbeat leaves at it.
        assert_eq!(beats.place(250_000), 300_000);
    }
}
