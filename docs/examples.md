# Cookbook

Real tasks. Every shown output on this page is real - a test reruns all of them and diffs the resulting ffmpeg commands, so if a recipe is here, it works.

Most recipes are parameterized (`:'source'`-style variables, filled by the `-v` flags in the shown command): swap the `-v` values and they run against your files. [Recipe 33](corpus.md#33-one-query-many-files) explains the mechanism; [packages/ffrwd/](../packages/ffrwd/) collects ready-made ones as installable packages.

These thirty are the ones worth reading in order. [corpus.md](corpus.md) holds a hundred more, each pinned the same way, for when you want the variation rather than the idea.

## 1. Transcode a file to H.264/AAC

The most-asked ffmpeg question there is. Select the streams, name the codecs in the sink, done - `faststart` moves the index to the front so the file starts playing before it finishes downloading:

```sql
COPY (
  SELECT f.video[1], f.audio[1]
  FROM input(:'source') f
) TO :'dest' WITH (
  video_codec 'libx264', crf 20, preset 'slow',
  audio_codec 'aac', audio_bitrate '192k', faststart true
)
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=film.mp4
ffmpeg -i film.mkv -map 0:v:0 -map 0:a:0 -c:0 libx264 -crf:0 20 -preset:0 slow -c:1 aac \
  -b:1 192k -movflags +faststart film.mp4
```

## 4. Trim a clip: fast stream copy, or frame-accurate re-encode

`WHERE t BETWEEN` becomes an input seek, and a stream nothing filters stays a copy - instant, but the cut snaps back to the previous keyframe, so it can start a little early:

```sql
COPY (
  SELECT a.video[1], a.audio[1]
  FROM input(:'source') a
  WHERE a.t BETWEEN 5 AND 60
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=clip.mp4 -v dest=cut.mp4
ffmpeg -ss 5 -to 60 -i clip.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy cut.mp4
```

When the exact cut point matters, re-encode: a decoded stream trims frame-accurate. The trade and the measurements behind it are in [docs/trimming.md](trimming.md):

```sql
COPY (
  SELECT a.video[1], a.audio[1]
  FROM input(:'source') a
  WHERE a.t BETWEEN 5 AND 60
) TO :'dest' WITH (video_codec 'libx264', crf 18, audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql -v source=clip.mp4 -v dest=cut.mp4
ffmpeg -ss 5 -to 60 -i clip.mp4 -map 0:v:0 -map 0:a:0 -c:0 libx264 -crf:0 18 -c:1 aac \
  cut.mp4
```

## 5. Resize to 1280 wide, or to half size

`-2` for the height means "keep the aspect ratio, rounded to an even number" - encoders insist on even dimensions, and this saves you doing the arithmetic:

```pgsql
COPY (
  SELECT scale(f.video[1], 1280, -2), f.audio[1]
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mp4 -v dest=small.mp4
ffmpeg -i film.mp4 -filter_complex '[0:v:0]scale=width=1280:height=-2[out0]' -map \
  '[out0]' -map 0:a:0 -c:1 copy small.mp4
```

Or express the width relative to the input - any string-typed option takes an ffmpeg expression - and let `-2` keep the aspect:

```pgsql
COPY (
  SELECT scale(f.video[1], 'iw/2', -2), f.audio[1]
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mp4 -v dest=half.mp4
ffmpeg -i film.mp4 -filter_complex '[0:v:0]scale=width=iw/2:height=-2[out0]' -map \
  '[out0]' -map 0:a:0 -c:1 copy half.mp4
```

## 8. Concatenate two clips

`UNION ALL` is ffmpeg's concat. SQL requires the branches to agree on column count, type and order, and that is exactly concat's segment contract - the interleaving that's so easy to get wrong by hand is generated for you:

```sql
COPY (
  SELECT a.video[1], a.audio[1] FROM input(:'first') a
  UNION ALL
  SELECT b.video[1], b.audio[1] FROM input(:'second') b
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v first=part1.mp4 -v second=part2.mp4 -v dest=joined.mp4
ffmpeg -i part1.mp4 -i part2.mp4 -filter_complex \
  '[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[out0][out1]' -map '[out0]' -map \
  '[out1]' joined.mp4
```

And it scales to files you'd rather not count streams in: splat the whole audio array and the languages pair up positionally, English with English, French with French, tags surviving. (This one needs real files - a splat has to know how many tracks there are.)

```pgsql
COPY (
  SELECT a.video[1], a.audio FROM input('tests/fixtures/av2.mp4') a
  UNION ALL
  SELECT b.video[1], b.audio FROM input('tests/fixtures/av3.mp4') b
) TO 'season.mkv'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -i tests/fixtures/av3.mp4 -filter_complex \
  '[0:v:0][0:a:0][0:a:1][1:v:0][1:a:0][1:a:1]concat=n=2:v=1:a=2[out0][out1][out2]' -map \
  '[out0]' -map '[out1]' -metadata:s:1 language=eng -map '[out2]' -metadata:s:2 \
  language=fra season.mkv
```

## 10. Mux external subtitles in, or pull them back out

A subtitle file is just another input. Select its track next to your video and audio; mp4 demands `mov_text`, mkv would take `srt` or `webvtt`:

```sql
COPY (
  SELECT f.video[1], f.audio[1], s.subtitle[1]
  FROM input(:'main') f, input(:'subs') s
) TO :'dest' WITH (subtitle_codec 'mov_text')
```

```
$ ffrwd compile -f query.sql -v main=film.mp4 -v subs=subs.en.vtt -v dest=captioned.mp4
ffmpeg -i film.mp4 -i subs.en.vtt -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -map 1:s:0 \
  -c:2 mov_text captioned.mp4
```

Extraction is the same idea with a shorter SELECT list - the container implies the format:

```sql
COPY (
  SELECT f.subtitle[1]
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=subs.en.srt
ffmpeg -i film.mkv -map 0:s:0 -c:0 copy subs.en.srt
```

## 15. Replace a video's audio, or duck music under the dialogue

Swapping is just selecting video from one input and audio from another:

```sql
COPY (
  SELECT v.video[1], m.audio[1]
  FROM input(:'main') v, input(:'voice') m
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v main=film.mp4 -v voice=voiceover.wav -v dest=dubbed.mp4
ffmpeg -i film.mp4 -i voiceover.wav -map 0:v:0 -c:0 copy -map 1:a:0 -c:1 copy dubbed.mp4
```

Keeping both, with the music turned down, is a mix:

```pgsql
COPY (
  SELECT v.video[1], amix(v.audio[1], volume(m.audio[1], 0.2))
  FROM input(:'main') v, input(:'music') m
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v main=film.mp4 -v music=music.m4a -v dest=scored.mp4
ffmpeg -i film.mp4 -i music.m4a -filter_complex \
  '[1:a:0]volume=volume=0.2[n1];[0:a:0][n1]amix=inputs=2[out1]' -map 0:v:0 -c:0 copy \
  -map '[out1]' scored.mp4
```

Real ducking - music that dips when someone speaks - is a sidechain compressor keyed off the dialogue. Naming `v.audio[1]` twice is fine; the compiler inserts the split:

```pgsql
COPY (
  SELECT v.video[1], amix(v.audio[1], sidechaincompress(m.audio[1], v.audio[1], threshold => 0.03, ratio => 8))
  FROM input(:'main') v, input(:'music') m
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v main=film.mp4 -v music=music.m4a -v dest=ducked.mp4
ffmpeg -i film.mp4 -i music.m4a -filter_complex \
  '[0:a:0]asplit=2[src_v_a_0_split0][src_v_a_0_split1];'\
'[1:a:0][src_v_a_0_split0]sidechaincompress=threshold=0.03:ratio=8[n1];'\
'[src_v_a_0_split1][n1]amix=inputs=2[out1]' -map 0:v:0 -c:0 copy -map '[out1]' \
  ducked.mp4
```

## 19. Blur a region, or blur during a time window

`ffrwd.blur_regions` is crop, blur and overlay in one call - the license-plate special:

```pgsql
COPY (
  SELECT ffrwd.blur_regions(f.video[1], 900, 60, 320, 180, 20), f.audio[1]
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=interview.mp4 -v dest=anonymized.mp4
ffmpeg -i interview.mp4 -filter_complex \
  '[0:v:0]split=2[src_f_v_0_split0][src_f_v_0_split1];'\
'[src_f_v_0_split0]crop=out_w=320:out_h=180:x=900:y=60,gblur=sigma=20[n2];'\
'[src_f_v_0_split1][n2]overlay=x=900:y=60[out0]' -map '[out0]' -map 0:a:0 -c:1 copy \
  anonymized.mp4
```

To apply an effect only during a time window, `enable` is the switch - no trimming, no branches, no concat, just a filter that turns itself on and off:

```pgsql
COPY (
  SELECT gblur(a.video[1], 12, enable => 'between(t,0.5,1.5)')
  FROM input(:'source') a
) TO 'out.mp4'
```

```
$ ffrwd compile -f query.sql -v source=clip.mp4
ffmpeg -i clip.mp4 -filter_complex \
  '[0:v:0]gblur=sigma=12:enable=between(t\,0.5\,1.5)[out0]' -map '[out0]' out.mp4
```

## 22. One decode, several outputs

A `CREATE VIEW` is a named, shared piece of the graph, and each `COPY` after it is one output file - the whole script is a single ffmpeg run, so the watermarking happens once no matter how many files consume it. (The classic version of this, the ABR rendition ladder, is in the README.)

```pgsql
CREATE VIEW branded AS
  SELECT overlay(f.video[1], logo.video[1], 'W-w-20', 20) AS v, f.audio[1] AS a
  FROM input(:'main') f, input(:'overlay', loop => true) logo;

COPY (SELECT scale(b.v, 1280, -2) AS v, b.a FROM branded b)
TO :'web' WITH (video_codec 'libx264', crf 21, audio_codec 'aac');

COPY (SELECT b.a FROM branded b)
TO :'podcast' WITH (audio_codec 'aac', audio_bitrate '128k')
```

```
$ ffrwd compile -f query.sql -v main=film.mp4 -v overlay=watermark.png -v web=web.mp4 -v podcast=podcast.m4a
ffmpeg -i film.mp4 -loop 1 -i watermark.png -filter_complex \
  '[0:v:0][1:v:0]overlay=x=W-w-20:y=20,scale=width=1280:height=-2[out0]' -map '[out0]' \
  -map 0:a:0 -c:0 libx264 -crf:0 21 -c:1 aac web.mp4 -map 0:a:0 -c:0 aac -b:0 128k \
  podcast.m4a
```

## 23. Pick a track by what it is, not where it sits

`unnest` turns a track array into rows - one per track, with the probed metadata as real columns - and a `WHERE` over those columns is track selection that says what you mean. The row IS the track: a bare `t` where a stream is expected selects it, filters it, or gathers it, and the columns are the metadata about it. No more counting streams in ffprobe output to learn that English is `[2]` this time:

```pgsql
COPY (
  SELECT t
  FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) t
  WHERE t.tags.language = 'eng'
) TO 'eng.m4a'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -map 0:a:0 -c:0 copy -metadata:s:0 language=eng eng.m4a
```

Audio rows carry `tags` (read by path: `t.tags.language`, `t.tags.title`, any key), `codec`, `channels`, `channel_layout`, `sample_rate`, `bitrate` and `duration`; video rows carry `width`, `height`, `fps` and friends instead. A track nobody probed has NULL in every metadata column, and NULL matches nothing - standard SQL, no new rules.

## 26. Mix everything the files have, missing tracks count as silence

An outer join keeps the rows only one side has, and `COALESCE` fills the gap - for audio, with generated silence:

```pgsql
COPY (
  SELECT array_agg(amix(a, COALESCE(b, ffmpeg.anullsrc(duration => 4))))
  FROM input('tests/fixtures/av2.mp4') f, input('tests/fixtures/av-eng.mp4') g,
       unnest(f.audio) a FULL OUTER JOIN unnest(g.audio) b ON a.tags.language = b.tags.language
) TO 'full.mka'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -i tests/fixtures/av-eng.mp4 -filter_complex \
  'anullsrc=duration=4[n1];[0:a:0][1:a:0]amix=inputs=2[out0];'\
'[0:a:1][n1]amix=inputs=2[out1]' -map '[out0]' -metadata:s:0 language=eng -map '[out1]' \
  -metadata:s:1 language=fra full.mka
```

The second file has no French, so the French mix gets silence in that slot - and keeps its `fra` tag, because the tag came from the side that existed.

## 27. Concatenate files with different track counts

The founding case. `concat` demands identical segment shapes, so the file that lacks a French track needs a silent stand-in - which is the same outer join, once per branch, each branch selecting its own side and gathering its rows into that segment. (Aliases respell in the second branch because alias names are script-wide.)

```pgsql
COPY (
  SELECT f.video[1], array_agg(a)
  FROM input('tests/fixtures/av2.mp4') f, input('tests/fixtures/av-eng.mp4') g,
       unnest(f.audio) a FULL OUTER JOIN unnest(g.audio) b ON a.tags.language = b.tags.language
  GROUP BY f.video[1]
  UNION ALL
  SELECT g2.video[1], array_agg(COALESCE(b2, ffmpeg.anullsrc(duration => 4)))
  FROM input('tests/fixtures/av2.mp4') f2, input('tests/fixtures/av-eng.mp4') g2,
       unnest(f2.audio) a2 FULL OUTER JOIN unnest(g2.audio) b2 ON a2.tags.language = b2.tags.language
  GROUP BY g2.video[1]
) TO 'joined.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -i tests/fixtures/av-eng.mp4 -filter_complex \
  'anullsrc=duration=4[n1];'\
'[0:v:0][0:a:0][0:a:1][1:v:0][1:a:0][n1]concat=n=2:v=1:a=2[out0][out1][out2]' -map \
  '[out0]' -map '[out1]' -metadata:s:1 language=eng -map '[out2]' -metadata:s:2 \
  language=fra joined.mp4
```

Both branches share one join shape, so both agree on track order, and eng concatenates with eng. Each file appears in two branches but gets ONE `-i`: untrimmed aliases over the same path share an input.

## 30. Look at a file's tracks as a table

A SELECT with no COPY is a table query: `run` (the default subcommand, so no subcommand at all) prints the result set and executes nothing - the whole answer was known at compile time. The columns are the probed metadata, so this is ffprobe you can read:

```pgsql
SELECT t.index, t.tags.language, t.codec, t.channel_layout
FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) t
```

```
$ ffrwd -f query.sql
 index | language | codec | channel_layout
-------+----------+-------+----------------
 1     | eng      | aac   | mono
 2     | fra      | aac   | mono
(2 rows)
```

`SELECT t.*` prints the row's whole scalar shape instead - `index`, `codec`, and whatever else that stream type carries. The map columns stay out of the star: one `disposition` cell is every flag ffmpeg knows. Name them - `t.tags.language`, `t.disposition.forced` - to print them. `SELECT *` over the input alias `f` prints its array columns, one cell each: `video`, `audio`, `subtitle`, `data`, `chapters`.

## 35. Hit a delivery spec

Device and platform specs name a profile, a level, and a rate-control ceiling; they map straight onto sink options. `-t`-style output limiting rides along as `duration`:

```pgsql
COPY (
  SELECT f.video[1], f.audio[1] FROM input(:'source') f
) TO :'dest' WITH (
  video_codec 'libx264', profile 'baseline', level '3.1',
  maxrate '2675k', bufsize '5350k', audio_codec 'aac', duration 30
)
```

```
$ ffrwd compile -f query.sql -v source=in.mkv -v dest=out.mp4
ffmpeg -i in.mkv -map 0:v:0 -map 0:a:0 -c:0 libx264 -profile:0 baseline -level:0 3.1 \
  -maxrate:0 2675k -bufsize:0 5350k -c:1 aac -t 30 out.mp4
```

## 41. Flag the default track

`disposition` is a field of the row, not a tag: its value is ffmpeg's disposition spec ('default', 'forced', 'default+forced'; '0' clears), and it says what the track's whole flag map is. Read it back by path, `t.disposition.default`. Players open the default track first, so this decides what people hear. Same two levels as recipe 38: flag the rows in the `WITH`, gather them outside it:

```pgsql
COPY (
  WITH flagged AS (
    SELECT t AS track,
           CASE WHEN t.tags.language = 'eng' THEN 'default' ELSE '0' END AS disposition
    FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) t
  )
  SELECT array_agg(flagged.track) FROM flagged
) TO 'flagged.mka'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -map 0:a:0 -c:0 copy -metadata:s:0 language=eng \
  -disposition:0 default -map 0:a:1 -c:1 copy -metadata:s:1 language=fra -disposition:1 \
  0 flagged.mka
```

Reading the flags back is the same path form the tags take, one key at a time, and each one is a boolean:

```pgsql
SELECT t.index, t.tags.language, t.disposition.default, t.disposition.forced
FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) t
```

```
$ ffrwd -f query.sql
 index | language | default | forced
-------+----------+---------+--------
 1     | eng      | true    | false
 2     | fra      | false   | false
(2 rows)
```

## 54. Gather rows into one file

A single destination takes exactly one row, so a multi-row query says how its rows combine: `array_agg` gathers streams in row order, `GROUP BY` names what stays constant. (Without them, a multi-row query into one path is a compile error naming both ways out.)

```pgsql
COPY (
  SELECT f.video, array_agg(a)
  FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) a
  GROUP BY f.video
) TO 'out.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -metadata:s:1 \
  language=eng -map 0:a:1 -c:2 copy -metadata:s:2 language=fra out.mp4
```

## 55. One file per language, all its tracks inside

Explicit grouping unlocks what the plain fan-out rejects as a collision: rows that SHARE a destination. `GROUP BY` a row column, aggregate the tracks, and fan out over the key - the group key doubles as each file's title:

```pgsql
COPY (
  SELECT array_agg(a), STRUCT(a.tags.language AS title) AS tags
  FROM input('tests/fixtures/av-2eng.mp4') f, unnest(f.audio) a
  GROUP BY a.tags.language
) TO (a.tags.language || '.mka')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av-2eng.mp4 -map 0:a:0 -c:0 copy -metadata:s:0 language=eng \
  -map 0:a:1 -c:1 copy -metadata:s:1 language=eng -metadata title=eng eng.mka -map 0:a:2 \
  -c:0 copy -metadata:s:0 language=fra -metadata title=fra fra.mka
```

## 67. Write a function and reuse it

`CREATE FUNCTION` defines a reusable expression. It takes typed parameters, returns one of the dialect's types, and its body is a single `SELECT` in a `$$` block. Calls are expanded at compile time, so a function is exactly the query you could have typed by hand:

```pgsql
CREATE FUNCTION normalize_lang(raw text) RETURNS text AS $$
  SELECT CASE WHEN raw IN ('en', 'eng', 'english') THEN 'eng' ELSE raw END
$$ LANGUAGE sql;

COPY (
  SELECT t, STRUCT(normalize_lang(t.tags.language) AS language) AS tags
  FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) t
) TO (normalize_lang(t.tags.language) || '.mka')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -map 0:a:0 -c:0 copy -metadata:s:0 language=eng eng.mka \
  -map 0:a:1 -c:0 copy -metadata:s:0 language=fra fra.mka
```

One definition, two uses - the tag it writes and the filename it picks. A function is legal anywhere a value of its return type is: a `tags` field, `WHERE`, a fan-out destination.

## 74. Cut a file into N clips, one file each

Any compile-time row may key a fan-out `TO`, not just an `unnest` - so a series turns "how many clips" into a parameter. Each row binds its own window, and its own filename:

```sql
COPY (
  SELECT f.video[1], f.audio[1]
  FROM input(:'source') f, generate_series(1, :count) i
  WHERE f.t >= (i.i - 1) * :len AND f.t <= i.i * :len
) TO ('clip' || i.i::text || '.mp4')
```

```
$ ffrwd compile -f query.sql -v source=film.mp4 -v count=3 -v len=5
ffmpeg -ss 0 -to 5 -i film.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy clip1.mp4 && \
  ffmpeg -ss 5 -to 10 -i film.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy clip2.mp4 && \
  ffmpeg -ss 10 -to 15 -i film.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy clip3.mp4
```

Raise `:count` and the same query writes more files; nothing else moves. Every bound is still a compile-time number, so the seeks are decided before ffmpeg starts.

## 79. Give a parameter a default

A signature parameter may declare `DEFAULT <literal>`: a trailing argument left out of the call takes it, same as Postgres. One deviation from Postgres - NULL takes the default too, not the literal NULL, because NULL is absence everywhere else in the dialect (an unset variable substitutes to it) and a caller writing `label(:'prefix')` needs an unset `:prefix` to mean "not given" here as well:

```sql
CREATE FUNCTION label(prefix text DEFAULT 'clip') RETURNS text AS $$
  SELECT prefix
$$ LANGUAGE sql;

COPY (
  SELECT f.video[1], f.audio[1], STRUCT(label() AS title) AS tags
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mp4 -v dest=out.mp4
ffmpeg -i film.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -metadata title=clip \
  out.mp4
```

A parameter with no `DEFAULT` still passes NULL through unchanged - that stays the way to write NULL itself, since a parameter that always defaults it away can never receive it.

`inputs` is written for you, same as `hstack`/`vstack`/`amix` - it is `xstack`'s own first option, so a positional argument right after the streams binds to it, not to `grid`; `grid` needs `=>` to reach past it.

## 87. Run a wasm module over the picture

`LANGUAGE wasm` declares a filter ffmpeg does not have. The two-part `AS` is Postgres's own spelling for a function implemented outside SQL - the module file, then the export in it - and the signature says what it does to streams: one `video_stream` in, one out. At compile time ffrwd asks the sidecar what the module declares, so a wrong export name, a world this ffrwd does not host, or a pixel format the wire cannot carry is a rejection before ffmpeg runs:

```pgsql
CREATE FUNCTION invert(v video_stream) RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/invert.wasm', 'invert'
  LANGUAGE wasm;

COPY (
  SELECT invert(f.video[1])
  FROM input('tests/fixtures/testsrc.mp4') f
) TO 'inverted.mp4' WITH (video_codec 'libx264', crf 20)
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/testsrc.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 rgba -f nut \
  pipe:1 | ffrwd-wasm -f nut -i pipe:0 -m \
  ../sidecar/modules/target/wasm32-wasip2/release/invert.wasm -f nut pipe:1 | ffmpeg \
  -copyts -f nut -analyzeduration 0 -fpsprobesize 3 -i pipe:0 -map 0:v:0 -c:0 libx264 \
  -crf:0 20 inverted.mp4
```

A module cannot be a link in one ffmpeg's filter graph, so the query compiles to three processes joined by pipes rather than one command: an ffmpeg that decodes, the sidecar hosting the module, and an ffmpeg that encodes what comes back. The frames travel as NUT, and the pixel format on both seams is the one the module and the wire agree on - `rgba` here, because that is what `invert` accepts. `ffrwd run` executes the whole pipeline itself; the printed form is for reading and pasting.

## 139. Rewrite a stream's packets without decoding it

`RETURNS packets` declares a module that reads a stream's ENCODED bytes and hands the same stream back - the interface a module weaving data into a file uses. Write the call as a column of the COPY's SELECT and the compiler defers it past the encoder the destination would have placed:

```pgsql
CREATE FUNCTION hand_on(v video_stream) RETURNS packets
  AS '../sidecar/modules/target/wasm32-wasip2/release/packet_passthrough.wasm',
  'packet_passthrough' LANGUAGE wasm;

COPY (
  SELECT hand_on(f.video[1]), f.audio[1]
  FROM input('tests/fixtures/av.mp4') f
) TO 'woven.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -map 0:v:0 -c:0 copy -f nut pipe:1 | ffrwd-wasm -f nut \
  -i pipe:0 -m ../sidecar/modules/target/wasm32-wasip2/release/packet_passthrough.wasm \
  -f nut pipe:1 | ffmpeg -i tests/fixtures/av.mp4 -f nut -analyzeduration 0 \
  -fpsprobesize 3 -i pipe:0 -map 1:v:0 -c:0 copy -map 0:a:0 -c:1 copy woven.mp4
```

Nothing here decodes a picture. The video already arrives in a codec the module accepts, so the first ffmpeg copies it onto the pipe, the sidecar hands the packets through, and the muxer copies them into the file - which is what makes this the cheap way to add data to something already encoded. The audio never touches the pipe at all: the muxer opens the source itself for it, so both streams reach the file with the times they had. That holds whatever else the query does with the audio - a module reading it for rows of its own puts a `split` in front of the column, and the muxer maps the source through it rather than taking the module's decoded samples off a pipe.

Name an encoder and the same shape re-encodes instead - `WITH (video_codec 'libx264', crf 20)` puts `-c:0 libx264 -crf:0 20` on the first ffmpeg, one process ahead of where it would otherwise have been, and the destination still copies what the filter wrote.

## 94. Blur the people, and only the people

`segment` finds objects and hands back two things at once: an index map -
each pixel holds the id of the object owning it - and a row per object.
The gather's WHERE is checked against the row record at compile time and
runs at runtime, inside the sidecar, as a filter on the rows; `mask_select`
turns the surviving ids into a binary mask, and `blur_mask` does the rest:

```pgsql
CREATE FUNCTION segment(v video_stream)
RETURNS STRUCT(map video_stream, objects STRUCT(id number, class text, score number,
                                                x number, y number, w number, h number)[])
  AS '../sidecar/modules/target/wasm32-wasip2/release/segment.wasm', 'segment' LANGUAGE wasm;

CREATE FUNCTION mask_select(map video_stream,
                            objects STRUCT(id number, class text, score number,
                                           x number, y number, w number, h number)[])
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/mask_select.wasm', 'mask_select' LANGUAGE wasm;

CREATE FUNCTION blur_mask(v video_stream, mask video_stream,
                          max_radius number DEFAULT 16, invert boolean DEFAULT FALSE)
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/blur_mask.wasm', 'blur_mask' LANGUAGE wasm;

COPY (
  SELECT blur_mask(s.video[1],
                   mask_select(segment(s.video[1]).map,
                               ARRAY(SELECT o FROM unnest(segment(s.video[1]).objects) o
                                     WHERE o.class = 'person')),
                   24), s.audio
  FROM input('tests/fixtures/av.mp4') s
) TO 'blurred.mp4' WITH (video_codec 'libx264', crf 20)
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 yuv420p -f nut \
  pipe:1 | ffrwd-wasm -f nut -i pipe:0 -nn \
  segment=../sidecar/modules/target/wasm32-wasip2/release/segment.onnx -m \
  segment=../sidecar/modules/target/wasm32-wasip2/release/segment.wasm -m \
  mask_select=../sidecar/modules/target/wasm32-wasip2/release/mask_select.wasm -m \
  blur_mask=../sidecar/modules/target/wasm32-wasip2/release/blur_mask.wasm \
  -filter_complex \
  '[0:v]segment[n1];'\
'[n1]rowfilter=pred={"eq"\\:\[{"field"\\:"class"}\,{"lit"\\:"person"}\]}[n2];'\
'[n2]mask_select[n3];[0:v][n3]blur_mask=max_radius=24:invert=0[out0]' -map '[out0]' -f \
  nut pipe:1 | ffmpeg -i tests/fixtures/av.mp4 -f nut -analyzeduration 0 -fpsprobesize 3 \
  -i pipe:0 -map 1:v:0 -map 0:a:0 -c:1 copy -c:0 libx264 -crf:0 20 blurred.mp4
```

`rowfilter` is the WHERE, compiled: not a module, but a node the host
provides, with the predicate carried in the graph. Swap `'person'` for any
COCO class, or write `AND o.score >= 0.5` to trim low-confidence matches -
the predicate grammar is comparisons, `LIKE`/`ILIKE` on a text field and
their `ANY (ARRAY[...])` and `IN (...)` spellings, AND, OR, NOT and
parentheses over the row's own fields. Both `segment(...)` spellings are one call: the map and
the rows leave the same node, and the rows ride the map's frames through
the filter.

## 98. Post what a module found, as it is found

A module can END the graph. `RETURNS sink` declares a function that is a
COPY destination: the SELECT list carries the streams it reads - and the
rows riding them - and the call after `TO` carries the values it is
configured with. Nothing comes back out; the sink's own effects are the
output. `post_rows` POSTs each annotation row to an HTTP endpoint, one
request per row, the row's JSON as the body:

```pgsql
CREATE FUNCTION detect_faces(v video_stream)
RETURNS STRUCT(v video_stream, faces STRUCT(x number, y number, w number, h number)[])
  AS '../sidecar/modules/target/wasm32-wasip2/release/facebox.wasm', 'facebox'
  LANGUAGE wasm;

CREATE FUNCTION post_rows(v video_stream,
                          faces STRUCT(x number, y number, w number, h number)[],
                          url text)
RETURNS sink
  AS '../sidecar/modules/target/wasm32-wasip2/release/post_rows.wasm', 'post_rows'
  LANGUAGE wasm;

COPY (
  SELECT detect_faces(s.video[1])
  FROM input('tests/fixtures/av.mp4') s
) TO post_rows('http://127.0.0.1:8123/detections')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 yuv420p -f nut \
  pipe:1 | ffrwd-wasm -f nut -i pipe:0 -http \
  ../sidecar/modules/target/wasm32-wasip2/release/post_rows.wasm -m \
  facebox=../sidecar/modules/target/wasm32-wasip2/release/facebox.wasm -m \
  post_rows=../sidecar/modules/target/wasm32-wasip2/release/post_rows.wasm \
  -filter_complex \
  '[0:v]facebox[n1];[n1]post_rows=url=http\\://127.0.0.1\\:8123/detections[out0]' -map \
  '[out0]' -f null -
```

The module imports `wasi:http`, and the compiler read that off its
describe: `-http <module>` on the sidecar's command line is the grant,
per module and deny-by-default - a module the argv never names cannot
reach the network at all. The sink's pad maps to `-f null -`, an output
that opens nothing; the pipeline ends when the input drains, and the
module gets one final call that says so. The URL is a value argument
like any other - which also means no secret belongs in it: the query
text is the command line.

## 104. Publish the ladder as HLS

The same rows, one written name: `format 'hls'` makes the destination a
manifest, so the multi-row relation is accepted - each row is one entry
of the variant map. A video row is a rung, an audio row a rendition,
and a row carrying both is a muxed variant; a `FULL JOIN` with disjoint
keys is how the demuxed shape spells its rows. The compiler derives the
keyframe discipline from `hls_time` and the frame rate, lays variant
playlists and segments out beside the master, and writes
`var_stream_map` by transcribing the rows - a hand-written one is
refused, naming what the compiler would write:

```pgsql
COPY (
  WITH vid AS (
    SELECT scale(fps(f.video[1], 15), ARRAY[320, 160][i.i], -2) AS v, i.i AS rung
    FROM input('tests/fixtures/av.mp4') f, generate_series(1, 2) i
  ),
  aud AS (
    SELECT a AS t, 2 + a.index AS rung
    FROM input('tests/fixtures/av.mp4') g, unnest(g.audio) a
  )
  SELECT vid.v, aud.t
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO 'out/master.m3u8'
  WITH (format 'hls', hls_time 2, hls_playlist_type 'vod',
        video_codec 'libx264', video_bitrate ARRAY['800k', '300k'][vid.rung],
        audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -filter_complex \
  '[0:v:0]fps=fps=15,split=2[n1_split0][n1_split1];'\
'[n1_split0]scale=width=320:height=-2[out0];[n1_split1]scale=width=160:height=-2[out1]' \
  -map '[out0]' -map '[out1]' -map 0:a:0 -f hls -hls_time 2 -hls_playlist_type vod -c:0 \
  libx264 -c:1 libx264 -b:0 800k -b:1 300k -c:2 aac -g:0 30 -g:1 30 -keyint_min:0 30 \
  -keyint_min:1 30 -sc_threshold:0 0 -sc_threshold:1 0 -var_stream_map \
  'v:0,agroup:aud,name:240p v:1,agroup:aud,name:120p a:0,agroup:aud,name:a0,default:yes' \
  -master_pl_name master.m3u8 -hls_segment_filename out/%v/segment_%d.ts \
  out/%v/index.m3u8
```

`format 'dash'` is the same plan wearing different words: `seg_duration`
for `hls_time`, adaptation sets for the variant map, and the `.mpd` as
the written name.

## 140. Ship the ladder as a function

The rungs above are spelled out in the query: a `generate_series` to
count them, an array for their widths, another for their bitrates, and
a `[vid.rung]` on each to line the two up. They do not have to be. A
`TABLE`-returning function's arguments read the FROM items written to
its left, so a stream can be handed to one, and a rung becomes a row
the function returns - its width, and the encoder settings that belong
with that width:

```pgsql
CREATE FUNCTION ladder(v video_stream)
RETURNS TABLE(v video_stream, rung number, bitrate text, bufsize text) AS $$
  SELECT scale(v, r.width, -2), r.rung, r.bitrate, r.bufsize
  FROM unnest(ARRAY[STRUCT(1 AS rung, 320 AS width, '800k' AS bitrate, '1600k' AS bufsize),
                    STRUCT(2 AS rung, 160 AS width, '300k' AS bitrate,  '600k' AS bufsize)]) r
$$ LANGUAGE sql;

COPY (
  WITH vid AS (
    SELECT l.v AS v, l.bitrate AS bitrate, l.bufsize AS bufsize, l.rung AS rung
    FROM input('tests/fixtures/av.mp4') f, ladder(fps(f.video[1], 15)) l
  ),
  aud AS (
    SELECT a AS t, 2 + a.index AS rung
    FROM input('tests/fixtures/av.mp4') g, unnest(g.audio) a
  )
  SELECT vid.v, aud.t
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO 'out/master.m3u8'
  WITH (format 'hls', hls_time 2, hls_playlist_type 'vod',
        video_codec 'libx264', video_bitrate vid.bitrate,
        maxrate vid.bitrate, bufsize vid.bufsize,
        audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -filter_complex \
  '[0:v:0]fps=fps=15,split=2[n1_split0][n1_split1];'\
'[n1_split0]scale=width=320:height=-2[out0];[n1_split1]scale=width=160:height=-2[out1]' \
  -map '[out0]' -map '[out1]' -map 0:a:0 -f hls -hls_time 2 -hls_playlist_type vod -c:0 \
  libx264 -c:1 libx264 -b:0 800k -b:1 300k -maxrate:0 800k -maxrate:1 300k -bufsize:0 \
  1600k -bufsize:1 600k -c:2 aac -g:0 30 -g:1 30 -keyint_min:0 30 -keyint_min:1 30 \
  -sc_threshold:0 0 -sc_threshold:1 0 -var_stream_map \
  'v:0,agroup:aud,name:240p v:1,agroup:aud,name:120p a:0,agroup:aud,name:a0,default:yes' \
  -master_pl_name master.m3u8 -hls_segment_filename out/%v/segment_%d.ts \
  out/%v/index.m3u8
```

Recipe 104's command with a rate control block added, and the function
is inlined, so what runs is still the query you could have typed by
hand. What changed is who owns the rungs. `ladder` is a definition a
package ships and a query installs; the caller writes one call in place
of a row table, and reads the settings back off the alias instead of
keeping a parallel array in step with it.

A `WITH` option is settled before ffmpeg runs, and a row column is: it
binds once per row, in row order, and is typed exactly as the literal
in its place would be. `video_bitrate ARRAY['800k', '300k'][vid.rung]`
is the same option written the other way, and compiles to the same
command.

The stream argument belongs to the outer row, so `fps(f.video[1], 15)`
runs once and splits into both rungs rather than once per rung.

## 105. Pick a rung from an ABR ladder

`input()` on a manifest - an HLS master playlist or a DASH MPD - yields
one row per rendition instead of one row for the file: `bandwidth`,
`width`, `height`, `codecs`, `name`, `language`, and `video`/`audio`
arrays sized to that rendition's own streams. `WHERE r.height = 720`
keeps the rung already sized right, so re-encoding it costs one decode,
not every rung's:

```pgsql
COPY (
  SELECT r.video[1], r.audio[1]
  FROM input(:'ladder') r
  WHERE r.height = 720
) TO :'dest' WITH (video_codec 'libx264', crf 20, audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql -v ladder=tests/fixtures/ladder/master.m3u8 -v dest=rung720.mp4
ffmpeg -i tests/fixtures/ladder/master.m3u8 -map 0:v:1 -map 0:a:1 -c:0 libx264 -crf:0 20 \
  -c:1 aac rung720.mp4
```

Reach for this when a source publishes an ABR ladder and the job wants
one rendition re-encoded, not the whole set.

## 113. Translate captions as they are produced

A rows function reads rows and writes rows, with no stream anywhere: `RETURNS cue[]` over one `cue[]` parameter says so, and the module it names has to be one the sidecar runs on rows alone. Written over the annotation column another module produces, it becomes a second node beside that module, fed by a rows edge rather than by frames - so the cues never leave the sidecar between the two:

```pgsql
CREATE FUNCTION captions(v video_stream)
RETURNS STRUCT(v video_stream, cues cue[])
  AS '../sidecar/modules/target/wasm32-wasip2/release/captions.wasm', 'captions'
  LANGUAGE wasm;

CREATE FUNCTION fauxlate(cues cue[]) RETURNS cue[]
  AS '../sidecar/modules/target/wasm32-wasip2/release/fauxlate.wasm', 'fauxlate'
  LANGUAGE wasm;

COPY (
  SELECT f.video[1], f.audio[1], fauxlate(captions(f.video[1]).cues)
  FROM input('tests/fixtures/av.mp4') f
) TO 'fauxlated.mkv'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 rgba -f nut pipe:1 | \
  ffrwd-wasm -f nut -i pipe:0 -m \
  ../sidecar/modules/target/wasm32-wasip2/release/captions.wasm -m \
  ../sidecar/modules/target/wasm32-wasip2/release/fauxlate.wasm -rows-from 0 -f webvtt \
  pipe:1 | ffmpeg -i tests/fixtures/av.mp4 -f webvtt -i pipe:0 -map 0:v:0 -c:0 copy -map \
  0:a:0 -c:1 copy -map 1:s:0 -c:2 copy fauxlated.mkv
```

`-rows-from 0` is the edge: it names the module whose rows arrive here by its position in the `-m` table, and nothing about it is a pad, so the filtergraph never mentions it. `fauxlate` is a stand-in for a real translator - every word gains `-a` and `-o` in turn, so `Cue one.` comes back `Cue-a one-o.` - and the result is the producer's column read one module later: selected here it is a subtitle track, and at a `.ndjson` destination it is the rows themselves.

One module, two forms: `fauxlate.wasm` also exports the value function `translate`, declared `RETURNS text` and run once per call at compile time, which is how the same word rule reaches a caption FILE's cues - the shape [recipe 112](#112-a-function-over-a-caption-files-cues) writes with `upper`. Rows a module produces at run time are this recipe's; rows the compiler already holds are that one's, and a rows function over them is refused by name.

## 115. Stitch the rows you keep, then re-encode them to a ladder

`WHERE` over the module's own column keeps rows the same way it keeps rendition rows from a manifest; a CTE holds the concat, two more CTEs read it apart, and a series cross join makes the rungs. `format 'hls'` accepts the result the same as any other row source. `testsrc.mp4` has no audio; it is row 2 and the `WHERE` drops it, so both kept rows carry audio:

```pgsql
CREATE FUNCTION files(paths text) RETURNS source
  AS '../sidecar/modules/target/wasm32-wasip2/release/source_files.wasm', 'files'
  LANGUAGE wasm;

COPY (
  WITH pod AS (
    SELECT concat(VARIADIC array_agg(scale(s.video[1], 320, -2))) AS v,
           concat(VARIADIC array_agg(s.audio[1])) AS a
    FROM files('tests/fixtures/av.mp4,tests/fixtures/testsrc.mp4,tests/fixtures/av2.mp4') s
    WHERE s.sequence != 2
  ),
  vid AS (
    SELECT scale(pod.v, ARRAY[320, 160][i.i], -2) AS v, i.i AS rung
    FROM pod, generate_series(1, 2) i
  ),
  aud AS (
    SELECT pod.a AS t, 3 AS rung
    FROM pod
  )
  SELECT vid.v, aud.t
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO 'out/master.m3u8'
  WITH (format 'hls', hls_time 2, hls_playlist_type 'vod',
        video_codec 'libx264', video_bitrate ARRAY['800k', '300k'][vid.rung],
        audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -i tests/fixtures/av2.mp4 -filter_complex \
  '[0:v:0]scale=width=320:height=-2[n1];[1:v:0]scale=width=320:height=-2[n2];'\
'[n1][n2]concat=n=2:v=1:a=0[n3];[0:a:0][1:a:0]concat=n=2:v=0:a=1[out2];'\
'[n3]split=2[n3_split0][n3_split1];[n3_split0]scale=width=320:height=-2[out0];'\
'[n3_split1]scale=width=160:height=-2[out1]' -map '[out0]' -map '[out1]' -map '[out2]' \
  -f hls -hls_time 2 -hls_playlist_type vod -c:0 libx264 -c:1 libx264 -b:0 800k -b:1 \
  300k -c:2 aac -g:0 30 -g:1 30 -keyint_min:0 30 -keyint_min:1 30 -sc_threshold:0 0 \
  -sc_threshold:1 0 -var_stream_map \
  'v:0,agroup:aud,name:v0 v:1,agroup:aud,name:v1 a:0,agroup:aud,name:a0,default:yes' \
  -master_pl_name master.m3u8 -hls_segment_filename out/%v/segment_%d.ts \
  out/%v/index.m3u8
```

Reach for this when each segment of a sequence needs its own best-fit
file stitched in, then the whole result re-encoded to a fixed ladder,
all in one query. `testsrc.mp4` never reaches the printed command: it
is row 2, the `WHERE` drops it before the `concat`, and a row a query
drops leaves the command entirely rather than showing up as a gap.

## 117. Rank rows by a vector

`vector` is a value function's fourth type, alongside `text`, `number` and `boolean`: `embed_text` runs once per row, at compile time, and its result is a row column like any other - printed capped as `[0.12, -0.03, ...]`, but not compared, concatenated or cast to text, since a vector has no order or text form of its own. `cos_similarity(vector, vector) -> number` is the one thing it is for, and its result sorts like any other computed value:

```pgsql
CREATE FUNCTION embed_text(prompt text) RETURNS vector
  AS '../sidecar/modules/target/wasm32-wasip2/release/fauxlate.wasm', 'embed_text'
  LANGUAGE wasm;

SELECT r.label, round(cos_similarity(embed_text(r.blurb), embed_text('a small pet')), 4) AS score
FROM unnest(ARRAY[
  STRUCT('cat' AS label, 'a cat sat on the mat' AS blurb),
  STRUCT('dog' AS label, 'a dog ran in the yard' AS blurb),
  STRUCT('car' AS label, 'a car drove down the road' AS blurb)
]) r
ORDER BY score DESC
LIMIT 2
```

```
$ ffrwd -f query.sql
 label | score
-------+--------
 car   | 0.8099
 dog   | 0.7841
(2 rows)
```

There is no vector literal - `ARRAY[0.1, 0.2]` names an array of numbers, not the dialect's `vector`, so the only way to a vector value is a value function's own RETURNS, over a row column or a literal alike, the same per-row footing every value function stands on. `r.blurb` feeds `embed_text` once per row, memoized on its argument the way any other value call is; `embed_text('a small pet')` runs once, since its argument is a literal. `ORDER BY score` is Postgres's own rule: a bare name in `ORDER BY` that matches a `SELECT` alias sorts by that alias's expression, so `cos_similarity(...)` is written once, not repeated.

`embed_text` here is `fauxlate.wasm`'s third export - a stand-in for a real embedder, same as `translate` stands in for a real translator ([recipe 113](#113-translate-captions-as-they-are-produced)). It counts each blurb's letters into eight buckets and L2-normalizes, so a blurb closer in cosine to `'a small pet'` is one that shares more letters with it, not one that means anything like it - which is why `car` and `dog` outrank `cat` above.

## 118. Write rows as titled tracks

A cue array in a stream position is a subtitle track; an ALIAS on it is that track's title, and several such columns are several tracks. The rows can come from anywhere the compiler can count at compile time - here a written row table, one span per line:

```pgsql
COPY (
  SELECT f.video[1], f.audio[1],
         array_agg(STRUCT(r.line AS text, r.start_t AS start_t,
                          r.end_t AS end_t)::cue) AS speech
  FROM input('tests/fixtures/av.mp4') f,
       unnest(ARRAY[
         STRUCT('a cat sat on the mat' AS line, 0 AS start_t, 1.5 AS end_t),
         STRUCT('a dog ran in the yard' AS line, 1.5 AS start_t, 3 AS end_t)
       ]) r
  GROUP BY f.video[1], f.audio[1]
) TO 'described.mkv'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -f webvtt -i \
  'data:text/vtt;'\
'base64,'\
'V0VCVlRUCgowMDowMDowMC4wMDAgLS0+IDAwOjAwOjAxLjUwMAphIGNhdCBzYXQgb24gdGhlIG1hdAoKMDA6MDA6MDEuNTAwIC0tPiAwMDowMDowMy4wMDAKYSBkb2cgcmFuIGluIHRoZSB5YXJkCg==' \
  -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -map 1:s:0 -c:2 copy -metadata:s:2 \
  title=speech described.mkv
```

`-metadata:s:2 title=speech` is the alias, and it is the name `f.cues['speech']` finds the track by afterwards. The row's other fields are not in the payload: a block's bounds are its span, and the title is the column's.

Matroska only. A titled track written anywhere else comes back under a different name or under none, so a `.mp4` destination is refused by name and the hint says `.mkv`. An UNTITLED cue array still writes wherever captions do ([recipe 65](#65-turn-chapters-into-a-subtitle-track-and-back)).

Vectors are not a track. A vector belonging to a span of a file goes into the file's own encoded packets, written by a `RETURNS packets` filter the destination places behind its encoder ([recipe 139](#139-rewrite-a-streams-packets-without-decoding-it)) and read back by a packet sink called in `FROM` ([corpus 138](corpus.md#138-read-a-streams-own-packets-while-compiling)). `ffrwd/index` is the package that does both for embedding vectors.

## 123. Re-lay a muxed ladder as a demuxed one

A muxed rendition carries its own audio; a demuxed master names the audio
once and points every video variant at it. Two CTEs say that - one row per
video rung, one row for the audio - joined on rung keys that never meet, so
the result is the rows of both sides and nothing paired. A CTE's stream
column is a row column like any other, one cell per row of its own relation,
and the join's unmatched rows are the NULLs the variant map reads as an
absent kind:

```pgsql
COPY (
  WITH vid AS (
    SELECT r.video[1] AS v, r.height AS rung
    FROM input(:'ladder') r
  ),
  aud AS (
    SELECT s.audio[1] AS t, 1000 + s.bandwidth AS rung
    FROM input(:'ladder') s
    WHERE s.height = 720
  )
  SELECT vid.v, aud.t
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO :'dest' WITH (
  format 'hls', hls_time 2, hls_playlist_type 'vod', hls_segment_type 'fmp4',
  video_codec 'libx264', audio_codec 'aac'
)
```

```
$ ffrwd compile -f query.sql -v ladder=tests/fixtures/ladder/master.m3u8 -v dest=out/master.m3u8
ffmpeg -i tests/fixtures/ladder/master.m3u8 -map 0:v:0 -map 0:v:1 -map 0:a:1 -f hls \
  -hls_time 2 -hls_playlist_type vod -hls_segment_type fmp4 -c:0 libx264 -c:1 libx264 \
  -c:2 aac -g:0 30 -g:1 30 -keyint_min:0 30 -keyint_min:1 30 -sc_threshold:0 0 \
  -sc_threshold:1 0 -var_stream_map \
  'v:0,agroup:aud,name:1080p v:1,agroup:aud,name:v1 a:0,agroup:aud,name:a0,default:yes' \
  -master_pl_name master.m3u8 -hls_segment_filename out/%v/segment_%d.m4s \
  -hls_fmp4_init_filename init.mp4 out/%v/index.m3u8
```

`WHERE s.height = 720` is an unmodified read of the 720p variant's own
audio cell, so it carries that rendition's identity - `720p` - into the
audio-only row it becomes here, the same way its video cell would. But the
720p rung's OWN video row is also named `720p`, and `%v` is one directory
namespace regardless of kind, so the two would collide (`out/720p/`) -
both fall back to their position (`v1`, `a0`) instead, exactly as two same-
named rows of one kind already did before this recipe existed.

Reach for this to publish someone else's muxed ladder as a demuxed one -
one audio rendition instead of a copy per rung - without writing the
variant map by hand. `WHERE s.height = 720` is which rung's audio the
group takes; drop it and the ladder's every rung would contribute one.

## 125. A hybrid master: muxed variants and an audio group

A hybrid master is both shapes at once: variants that mux their own audio,
plus an audio rendition a player can switch to. `COALESCE` over two stream
columns takes the first non-NULL per row - the muxed rows keep their own
audio, and the audio-only row, which has none, takes the group's:

```pgsql
COPY (
  WITH vid AS (
    SELECT v.video[1] AS v, a.audio[1] AS a, v.height AS rung
    FROM input(:'ladder') v, input(:'ladder') a
    WHERE v.height IS NOT NULL AND a.height IS NULL
  ),
  aud AS (
    SELECT b.audio[1] AS t, 1000 + b.bandwidth AS rung
    FROM input(:'ladder') b
    WHERE b.height IS NULL
  )
  SELECT vid.v, COALESCE(vid.a, aud.t)
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO :'dest' WITH (
  format 'hls', hls_time 2, hls_playlist_type 'vod', hls_segment_type 'fmp4',
  video_codec 'libx264', audio_codec 'aac'
)
```

```
$ ffrwd compile -f query.sql -v ladder=tests/fixtures/ladder-demuxed/master.mpd -v dest=out/master.m3u8
ffmpeg -i tests/fixtures/ladder-demuxed/master.mpd -filter_complex \
  '[0:a:0]asplit=2[out2][out3]' -map 0:v:0 -map 0:v:1 -map '[out2]' -map '[out3]' -map \
  0:a:0 -f hls -hls_time 2 -hls_playlist_type vod -hls_segment_type fmp4 -c:0 libx264 \
  -c:1 libx264 -c:2 aac -c:3 aac -c:4 aac -g:0 30 -g:1 30 -keyint_min:0 30 -keyint_min:1 \
  30 -sc_threshold:0 0 -sc_threshold:1 0 -var_stream_map \
  'v:0,a:0,agroup:aud,name:0 v:1,a:1,agroup:aud,name:1 a:2,agroup:aud,name:2,'\
'default:yes' -master_pl_name master.m3u8 -hls_segment_filename out/%v/segment_%d.m4s \
  -hls_fmp4_init_filename init.mp4 out/%v/index.m3u8
```

Reach for this when a player has to be given both - a variant it can play
alone and an audio rendition it can switch to. Every COALESCE argument is
one kind of track, and a row where all of them are NULL is a row with no
stream of that kind, which the variant map writes as an absent one rather
than a rejection. Names come from the source ladder's own DASH
Representation ids (`0`, `1`, `2`, this MPD's own numbering, not a
computed `1080p`) - each row's video cell (or, on the audio-only row, its
audio cell) is an unmodified read of a rendition row, so it carries that
rendition's own name rather than one derived from height or language.


## 144. Hand one module's rows to a module reading two streams

A module reading several streams takes an annotation column too, after all of its streams, and the rows it reads ride its FIRST stream. `(notes(...)).*` in a WITH body names both halves of one call, the stream as `v` and its rows as `notes`, and the reader takes them back as its first stream and its column - here by name:

```pgsql
CREATE FUNCTION notes(v video_stream, label text DEFAULT 'note',
                      every number DEFAULT 5)
RETURNS STRUCT(v video_stream, notes STRUCT(pts number, note text)[])
  AS '../sidecar/modules/target/wasm32-wasip2/release/note_rows.wasm', 'note_rows'
  LANGUAGE wasm;

CREATE FUNCTION pad_rows(a video_stream, b video_stream,
                         notes STRUCT(pts number, note text)[])
RETURNS STRUCT(v video_stream,
               seen STRUCT(pts number, pad0 number, others number, notes text)[])
  AS '../sidecar/modules/target/wasm32-wasip2/release/pad_rows.wasm', 'pad_rows'
  LANGUAGE wasm;

COPY (
  WITH src AS (SELECT f.video[1] AS v FROM input('tests/fixtures/av.mp4') f),
       n AS (SELECT (notes(src.v, 'first', 2)).* FROM src),
       o AS (SELECT (notes(src.v, 'other', 1)).* FROM src)
  SELECT pad_rows(n.v, o.v, notes => n.notes).seen FROM n, o
) TO 'seen.ndjson'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 rgba -f nut pipe:1 | \
  ffrwd-wasm -f nut -i pipe:0 -m \
  note_rows=../sidecar/modules/target/wasm32-wasip2/release/note_rows.wasm -m \
  pad_rows=../sidecar/modules/target/wasm32-wasip2/release/pad_rows.wasm -filter_complex \
  '[0:v]note_rows=label=first:every=2[n1];[0:v]note_rows=label=other:every=1[n2];'\
'[n1][n2]pad_rows[out0]' -map '[out0]' -f ndjson seen.ndjson
```

Both `notes` calls and `pad_rows` run in one sidecar, and the rows never leave it: `n`'s notes ride the frames on `pad_rows`' first pad, and `o`'s, on the second, are dropped before the module is called. `pad_rows` is a stand-in that writes one row per call saying what reached it, so `seen.ndjson` holds sixty rows, `"first-0"` on frame 0, nothing on frame 1, `"first-2"` on frame 2, and never a note from `o`.

The first stream has to be the one those rows ride, untouched: `pad_rows(o.v, n.v, notes => n.notes)` is refused, and so is an ffmpeg filter between `n.v` and the call, since ffmpeg carries the frames on and drops the rows. Selecting both halves is a WITH body's business alone; a SELECT that writes its columns reads `.notes` on its own, as a track.

## 145. A detector returns its rows, and the picture stays where it was

A module reading one stream need not hand it back. `spot` finds a mark in each frame and returns rows alone - `RETURNS STRUCT(...)[]` with no stream beside it is a data stream the query reads while it runs - and `ring` draws them, reading the picture from the source and the rows from the detector. Both arguments descend from `f.video[1]`, so the rows pair with the frames by pts, frame for frame, and nothing is copied that the detector did not make:

```pgsql
CREATE FUNCTION spot(v video_stream, every number DEFAULT 30)
RETURNS STRUCT(start_t number, id number, x number, y number, w number, h number)[]
  AS '../sidecar/modules/target/wasm32-wasip2/release/spot.wasm', 'spot'
  LANGUAGE wasm;

CREATE FUNCTION ring(v video_stream,
                     spots STRUCT(start_t number, id number,
                                  x number, y number, w number, h number)[])
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/ring.wasm', 'ring'
  LANGUAGE wasm;

COPY (
  SELECT ring(f.video[1], spot(f.video[1]))
  FROM input('tests/fixtures/testsrc.mp4') f
) TO 'ringed.mp4' WITH (video_codec 'libx264', crf 20)
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/testsrc.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 rgba -f nut \
  pipe:1 | ffrwd-wasm -f nut -i pipe:0 -pad \
  '{"color": {"range": "pc", "primaries": "unknown", "trc": "unknown", "space": "gbr"}}' \
  -m spot=../sidecar/modules/target/wasm32-wasip2/release/spot.wasm -m \
  ring=../sidecar/modules/target/wasm32-wasip2/release/ring.wasm -filter_complex \
  '[v=0:v]spot=every=30[spots=n1];[v=0:v][spots=n1]ring[v=out0]' -bound \
  'spot=[{"input":"v","streams":[{"rate":{"num":15,"den":1}}]}]' -bound \
  'ring=[{"input":"v","streams":[{"rate":{"num":15,"den":1}}]},{"input":"spots",'\
'"streams":[{"rate":null}]}]' -map '[out0]' -f nut pipe:1 | ffmpeg -copyts -f nut \
  -analyzeduration 0 -fpsprobesize 3 -i pipe:0 -map 0:v:0 -c:0 libx264 -crf:0 20 \
  ringed.mp4
```

`spot` writes one row per frame for as long as the mark is in view, and every row of one mark carries the pts it was first seen at as `start_t`: the row for frame t says what is true at t, so a reader needs no look-ahead and a run split across workers agrees on the ids. The old spelling, a module returning `STRUCT(v video_stream, spots ...)` with the picture untouched, is what a package keeps when its own module has not moved: inside the sidecar the rows ride the frames exactly as before. A migrated package that wants the old reading back writes it in SQL - `CREATE FUNCTION spotted(v video_stream) RETURNS STRUCT(v video_stream, spots STRUCT(...)[]) AS $$ SELECT v, spot(v) AS spots $$ LANGUAGE sql` - and `ring(spotted(v))` reads the record as the stream and the rows, as a call over a two-part result always has.

## 146. A reader names only the fields it reads

`dim` wants a box; it does not care who found it or when. Its parameter declares the four fields it reads, and `spot`'s six-field rows are accepted because every field `dim` names is there with the type it names - the others pass through untouched. Row matching is structural, so one reader serves every detector whose rows carry a box:

```pgsql
CREATE FUNCTION spot(v video_stream, every number DEFAULT 30)
RETURNS STRUCT(start_t number, id number, x number, y number, w number, h number)[]
  AS '../sidecar/modules/target/wasm32-wasip2/release/spot.wasm', 'spot'
  LANGUAGE wasm;

CREATE FUNCTION dim(v video_stream, boxes STRUCT(x number, y number, w number, h number)[],
                    amount number DEFAULT 0.5)
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/dim.wasm', 'dim'
  LANGUAGE wasm;

COPY (
  SELECT dim(f.video[1],
             ARRAY(SELECT s FROM unnest(spot(f.video[1])) s WHERE s.w >= 20))
  FROM input('tests/fixtures/testsrc.mp4') f
) TO 'dimmed.mp4' WITH (video_codec 'libx264', crf 20)
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/testsrc.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 rgba -f nut \
  pipe:1 | ffrwd-wasm -f nut -i pipe:0 -pad \
  '{"color": {"range": "pc", "primaries": "unknown", "trc": "unknown", "space": "gbr"}}' \
  -m spot=../sidecar/modules/target/wasm32-wasip2/release/spot.wasm -m \
  dim=../sidecar/modules/target/wasm32-wasip2/release/dim.wasm -filter_complex \
  '[v=0:v]spot=every=30[spots=n1];'\
'[n1]rowfilter=pred={"ge"\\:\[{"field"\\:"w"}\,{"lit"\\:20}\]}[n2];'\
'[v=0:v][boxes=n2]dim=amount=0.5[v=out0]' -bound \
  'spot=[{"input":"v","streams":[{"rate":{"num":15,"den":1}}]}]' -bound \
  'dim=[{"input":"v","streams":[{"rate":{"num":15,"den":1}}]},{"input":"boxes",'\
'"streams":[{"rate":null}]}]' -map '[out0]' -f nut pipe:1 | ffmpeg -copyts -f nut \
  -analyzeduration 0 -fpsprobesize 3 -i pipe:0 -map 0:v:0 -c:0 libx264 -crf:0 20 \
  dimmed.mp4
```

A field the reader names that the producer's rows lack, or carries with another type, is still refused at compile time, naming both. The WHERE runs inside the sidecar as before, and it may test fields the reader never sees.

## 147. One call, however many places read it

`hear` listens to the sound and writes a cue per window. The query reads it twice, as a caption track of the file and as the words `burn` paints onto the picture, and that is one node: a call with the same arguments written anywhere in one query is evaluated once, and its data output is split to every reader. Today those were two runs of the model, since a caption track feeds the muxer and a module's rows feed a stage of its own:

```pgsql
CREATE FUNCTION hear(a audio_stream) RETURNS cue[]
  AS '../sidecar/modules/target/wasm32-wasip2/release/hear.wasm', 'hear'
  LANGUAGE wasm;

CREATE FUNCTION burn(v video_stream, a audio_stream DEFAULT NULL,
                     words cue[] DEFAULT NULL)
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/burn.wasm', 'burn'
  LANGUAGE wasm;

COPY (
  SELECT burn(f.video[1], words => hear(f.audio[1])), f.audio[1], hear(f.audio[1])
  FROM input('tests/fixtures/av.mp4') f
) TO 'heard.mkv' WITH (video_codec 'libx264', crf 20, audio_codec 'copy')
```

```
$ ffrwd compile -f query.sql
# named pipes: ffmpeg0 reads sidecar0, sidecar0; sidecar0 feeds ffmpeg0, ffmpeg0
1. ffmpeg: ffmpeg -i tests/fixtures/av.mp4 -f webvtt -i \
  '<named pipe sidecar0-ffmpeg0 ffrwd.cues#2 read>' -f nut -analyzeduration 0 \
  -fpsprobesize 3 -i '<named pipe sidecar0-ffmpeg0 n2 read>' -map 2:v:0 -map 0:a:0 -map \
  1:s:0 -c:2 copy -c:0 libx264 -crf:0 20 -c:1 copy heard.mkv
2. ffmpeg: ffmpeg -i tests/fixtures/av.mp4 -map 0:a:0 -map 0:v:0 -ar:0 48000 -c:0 \
  pcm_f32le -c:1 rawvideo -pix_fmt:1 rgba -f nut pipe:1
3. sidecar: ffrwd-wasm -f nut -i pipe:0 -pad \
  '{"color": {"range": "pc", "primaries": "unknown", "trc": "unknown", "space": "gbr"}}' \
  -m hear=../sidecar/modules/target/wasm32-wasip2/release/hear.wasm -m \
  burn=../sidecar/modules/target/wasm32-wasip2/release/burn.wasm -filter_complex \
  '[a=0:a]hear[cues=out1];[v=0:v][words=out1]burn[v=out0]' -bound \
  'hear=[{"input":"a","streams":[{"rate":{"num":48000,"den":1}}]}]' -bound \
  'burn=[{"input":"v","streams":[{"rate":{"num":15,"den":1}}]},{"input":"words",'\
'"streams":[{"rate":null}]}]' -map '[out0]' -f nut \
  '<named pipe sidecar0-ffmpeg0 n2 write>' -map '[out1]' -f webvtt \
  '<named pipe sidecar0-ffmpeg0 ffrwd.cues#2 write>'
# this listing is not a shell command -- run the plan with `ffrwd run`
```

Two calls whose arguments differ are still two nodes. The split is of a data edge, so what it costs is a second copy of each row, not a second run.

## 148. A node reads the picture, the sound and the words at once

A module's inputs are any mix of kinds in any order: `burn` takes the picture as its clock, the sound beside it frame for frame, and the words by their time. Each is a port of its own with its own pairing, declared by the module and read by the compiler, so the query writes the call and nothing about how the three are lined up:

```pgsql
CREATE FUNCTION hear(a audio_stream) RETURNS cue[]
  AS '../sidecar/modules/target/wasm32-wasip2/release/hear.wasm', 'hear'
  LANGUAGE wasm;

CREATE FUNCTION burn(v video_stream, a audio_stream DEFAULT NULL,
                     words cue[] DEFAULT NULL)
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/burn.wasm', 'burn'
  LANGUAGE wasm;

COPY (
  SELECT burn(f.video[1], f.audio[1], hear(f.audio[1])), f.audio[1]
  FROM input('tests/fixtures/av.mp4') f
) TO 'burned.mp4' WITH (video_codec 'libx264', crf 20, audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -filter_complex '[0:a:0]asplit=2[out0][out2]' -map \
  '[out0]' -map 0:v:0 -map '[out2]' -ar:0 48000 -c:0 pcm_f32le -c:2 pcm_f32le -c:1 \
  rawvideo -pix_fmt:1 rgba -f nut pipe:1 | ffrwd-wasm -f nut -i pipe:0 -pad \
  '{"color": {"range": "pc", "primaries": "unknown", "trc": "unknown", "space": "gbr"}}' \
  -m hear=../sidecar/modules/target/wasm32-wasip2/release/hear.wasm -m \
  burn=../sidecar/modules/target/wasm32-wasip2/release/burn.wasm -filter_complex \
  '[a=0:a]hear[cues=n1];[v=0:v][a=0:a:1][words=n1]burn[v=out0]' -bound \
  'hear=[{"input":"a","streams":[{"rate":{"num":48000,"den":1}}]}]' -bound \
  'burn=[{"input":"v","streams":[{"rate":{"num":15,"den":1}}]},{"input":"a",'\
'"streams":[{"rate":{"num":44100,"den":1}}]},{"input":"words",'\
'"streams":[{"rate":null}]}]' -map '[out0]' -f nut pipe:1 | ffmpeg -i \
  tests/fixtures/av.mp4 -f nut -analyzeduration 0 -fpsprobesize 3 -i pipe:0 -map 1:v:0 \
  -map 0:a:0 -c:0 libx264 -crf:0 20 -c:1 aac burned.mp4
```

`hear` works two seconds of sound at a time and says so, and a window's cues leave with the window. `burn` reads them by interval, so the host holds each picture until the window holding its time is done, and the picture leaves `burn` two seconds behind the sound that enters it. [Recipe 153](#153-see-what-each-node-waits-for) shows where that number is printed.

## 149. Leave an input out

Any stream parameter may carry `DEFAULT NULL`, and a call leaves it off or writes `NULL`: the port is unbound and the module is told so. `burn` with the picture alone paints nothing; with the sound and no words it paints a level meter; `inset` shows a feed that connects later, by port, over the picture, and runs on the picture alone until something connects:

```pgsql
CREATE FUNCTION burn(v video_stream, a audio_stream DEFAULT NULL,
                     words cue[] DEFAULT NULL)
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/burn.wasm', 'burn'
  LANGUAGE wasm;

CREATE FUNCTION inset(v video_stream, feed video_stream DEFAULT NULL,
                      port number DEFAULT 9000, lead number DEFAULT 0.5)
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/inset.wasm', 'inset'
  LANGUAGE wasm;

COPY (
  SELECT inset(burn(f.video[1], f.audio[1]), port => 9100), f.audio[1]
  FROM input('tests/fixtures/av.mp4') f
) TO 'inset.mp4' WITH (video_codec 'libx264', crf 20, audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -map 0:v:0 -map 0:a:0 -c:0 rawvideo -pix_fmt:0 rgba -c:1 \
  pcm_f32le -f nut pipe:1 | ffrwd-wasm -f nut -i pipe:0 -pad \
  '{"color": {"range": "pc", "primaries": "unknown", "trc": "unknown", "space": "gbr"}}' \
  -m burn=../sidecar/modules/target/wasm32-wasip2/release/burn.wasm -m \
  inset=../sidecar/modules/target/wasm32-wasip2/release/inset.wasm -filter_complex \
  '[v=0:v][a=0:a]burn[v=n1];[v=n1]inset=port=9100:lead=0.5[v=out0]' -bound \
  'burn=[{"input":"v","streams":[{"rate":{"num":15,"den":1}}]},{"input":"a",'\
'"streams":[{"rate":{"num":44100,"den":1}}]}]' -bound \
  'inset=[{"input":"v","streams":[{"rate":{"num":15,"den":1}}]}]' -map '[out0]' -f nut \
  pipe:1 | ffmpeg -i tests/fixtures/av.mp4 -f nut -analyzeduration 0 -fpsprobesize 3 -i \
  pipe:0 -map 1:v:0 -map 0:a:0 -c:0 libx264 -crf:0 20 -c:1 aac inset.mp4
# listens: sidecar0 at tcp://127.0.0.1:9100 for inset(feed)
```

`feed` is a hold input: whatever connects to port 9100 is shown at the picture's pace from `lead` seconds after its first frame arrives, the last frame held while it runs late, and the picture alone again when it ends. The compile listing names the port and the process that owns it, as it does for a switch's feeders. An input with no `DEFAULT NULL` is required, and a call that leaves it off is refused.

## 150. Tile any number of pictures

`video_stream[]` declares a port that takes as many streams as the call gives it, each arriving with its own size, rate and tags. `tile` lays them out in a grid. It ticks at the first picture's rate, or at `fps` when the call says so, and at every tick shows each stream's newest frame, so the pictures need not share a rate or a start:

```pgsql
CREATE FUNCTION tile(v video_stream[], columns number DEFAULT 2, fps number DEFAULT NULL)
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/tile.wasm', 'tile'
  LANGUAGE wasm;

COPY (
  SELECT tile(ARRAY[a.video[1], b.video[1], c.video[1]], 3)
  FROM input('tests/fixtures/av.mp4') a,
       input('tests/fixtures/av2.mp4') b,
       input('tests/fixtures/testsrc.mp4') c
) TO 'tiled.mp4' WITH (video_codec 'libx264', crf 20)
```

```
$ ffrwd compile -f query.sql
# named pipes: sidecar0 reads ffmpeg1, ffmpeg2, ffmpeg3
1. ffmpeg: ffmpeg -copyts -f nut -analyzeduration 0 -fpsprobesize 3 -i pipe:0 -map 0:v:0 \
  -c:0 libx264 -crf:0 20 tiled.mp4
2. ffmpeg: ffmpeg -i tests/fixtures/av.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 rgba -f \
  nut '<named pipe ffmpeg1-sidecar0 src:a:v:0 write>'
3. ffmpeg: ffmpeg -i tests/fixtures/av2.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 rgba -f \
  nut '<named pipe ffmpeg2-sidecar0 src:b:v:0 write>'
4. ffmpeg: ffmpeg -i tests/fixtures/testsrc.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 rgba \
  -f nut '<named pipe ffmpeg3-sidecar0 src:c:v:0 write>'
5. sidecar: ffrwd-wasm -f nut -i '<named pipe ffmpeg1-sidecar0 src:a:v:0 read>' -pad \
  '{"color": {"range": "pc", "primaries": "unknown", "trc": "unknown", "space": "gbr"}}' \
  -f nut -i '<named pipe ffmpeg2-sidecar0 src:b:v:0 read>' -pad \
  '{"color": {"range": "pc", "primaries": "unknown", "trc": "unknown", "space": "gbr"}}' \
  -f nut -i '<named pipe ffmpeg3-sidecar0 src:c:v:0 read>' -pad \
  '{"color": {"range": "pc", "primaries": "unknown", "trc": "unknown", "space": "gbr"}}' \
  -m tile=../sidecar/modules/target/wasm32-wasip2/release/tile.wasm -filter_complex \
  '[v=0:v][v=1:v][v=2:v]tile=columns=3[v=out0]' -bound \
  'tile=[{"input":"v","streams":[{"rate":{"num":15,"den":1}},{"rate":{"num":15,'\
'"den":1}},{"rate":{"num":15,"den":1}}]}]' -map '[out0]' -f nut pipe:1
# this listing is not a shell command -- run the plan with `ffrwd run`
```

A bare array column broadcasts over a filter, one call per element; over a module port declared as an array it is the port's whole list, and a module that wants one call per element is called under `unnest`. `audio_stream[]` and a rows parameter with `[]` on the record work the same way. A port that takes several streams cannot be the module's clock, which is why `tile` keeps time itself; the rate it keeps is read off the first picture by the compiler, which knows every stream's rate before anything runs.

## 151. A node makes a matte and the rows that go with it

A module that produces two things returns a record naming both: `matte` makes a gray matte of the mark it finds and a row per mark, and both leave the one node. Read either field off the call, or every field at once with `.*` in a WITH body; however the fields are read, the call is one instance. Here the rows dim the picture where the marks are and the matte is written beside it:

```pgsql
CREATE FUNCTION matte(v video_stream, every number DEFAULT 30)
RETURNS STRUCT(mask video_stream,
               spots STRUCT(start_t number, id number,
                            x number, y number, w number, h number)[])
  AS '../sidecar/modules/target/wasm32-wasip2/release/matte.wasm', 'matte'
  LANGUAGE wasm;

CREATE FUNCTION dim(v video_stream, boxes STRUCT(x number, y number, w number, h number)[],
                    amount number DEFAULT 0.5)
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/dim.wasm', 'dim'
  LANGUAGE wasm;

COPY (
  WITH m AS (SELECT f.video[1] AS v, (matte(f.video[1])).*
             FROM input('tests/fixtures/testsrc.mp4') f)
  SELECT dim(m.v, m.spots), m.mask
  FROM m
) TO 'matte.mkv' WITH (video_codec 'ffv1')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/testsrc.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 rgba -f nut \
  pipe:1 | ffrwd-wasm -f nut -i pipe:0 -pad \
  '{"color": {"range": "pc", "primaries": "unknown", "trc": "unknown", "space": "gbr"}}' \
  -m matte=../sidecar/modules/target/wasm32-wasip2/release/matte.wasm -m \
  dim=../sidecar/modules/target/wasm32-wasip2/release/dim.wasm -filter_complex \
  '[v=0:v]matte=every=30[mask=out1][spots=n11];[v=0:v][boxes=n11]dim=amount=0.5[v=out0]' \
  -bound 'matte=[{"input":"v","streams":[{"rate":{"num":15,"den":1}}]}]' -bound \
  'dim=[{"input":"v","streams":[{"rate":{"num":15,"den":1}}]},{"input":"boxes",'\
'"streams":[{"rate":null}]}]' -map '[out0]' -map '[out1]' -f nut pipe:1 | ffmpeg -copyts \
  -f nut -analyzeduration 0 -fpsprobesize 3 -i pipe:0 -map 0:v:0 -map 0:v:1 -c:0 ffv1 \
  -c:1 ffv1 matte.mkv
```

Each field is an output port with a format and a time base of its own, declared by the module for the call's parameters. [Recipe 94](#94-blur-the-people-and-only-the-people)'s `segment` is this shape, and its rows no longer ride the map's frames: they are a data stream beside it, which is why `mask_select` can read them from a call `segment` is not part of.

## 152. Spans from the rows that said so, frame by frame

A row per frame is the honest shape for a thing known as it happens, and a span is what a subtitle track or a record wants. `ffrwd.merge_spans` is the reducer between them, a node the host provides: it groups rows by `start_t`, keeps the last row's fields, and ends each span at the last frame that carried it plus that frame's duration. A frame that misses a row is a gap inside the span, not the end of it:

```pgsql
CREATE FUNCTION spot(v video_stream, every number DEFAULT 30)
RETURNS STRUCT(start_t number, id number, x number, y number, w number, h number)[]
  AS '../sidecar/modules/target/wasm32-wasip2/release/spot.wasm', 'spot'
  LANGUAGE wasm;

COPY (
  SELECT ffrwd.merge_spans(spot(f.video[1]), max_span => 10)
  FROM input('tests/fixtures/testsrc.mp4') f
) TO 'spots.ndjson'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/testsrc.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 rgba -f nut \
  pipe:1 | ffrwd-wasm -f nut -i pipe:0 -pad \
  '{"color": {"range": "pc", "primaries": "unknown", "trc": "unknown", "space": "gbr"}}' \
  -m spot=../sidecar/modules/target/wasm32-wasip2/release/spot.wasm -filter_complex \
  '[v=0:v]spot=every=30[spots=n1];[n1]rowmerge=max_span=10[out0]' -bound \
  'spot=[{"input":"v","streams":[{"rate":{"num":15,"den":1}}]}]' -map '[out0]' -f ndjson \
  spots.ndjson
```

The rows out carry `start_t` and `end_t` beside the fields in, one row per span, so `spots.ndjson` holds one line per mark rather than one per frame. Rows whose fields are `start_t` and `text` reduce to cues, and selecting them beside a picture writes a subtitle track. A span row leaves when its span ends, so a span is as late as it is long; `max_span` bounds that, and it is what a reader pairing by time waits for. A span still open after ten seconds is written as it stands and goes on as a new one. The reducer closes a span on its producer's progress, not on the next row, so the last span of a run ends where the rows did.

## 153. See what each node waits for

Every node declares the window it works in and how late its rows may leave, and the compiler adds the waits up along each path. `explain` prints them: per node, its window in streaming SQL's words (per-frame, tumbling, hopping, sliding), and per output, how far behind the source it runs. The query is [recipe 148](#148-a-node-reads-the-picture-the-sound-and-the-words-at-once)'s:

```pgsql
CREATE FUNCTION hear(a audio_stream) RETURNS cue[]
  AS '../sidecar/modules/target/wasm32-wasip2/release/hear.wasm', 'hear'
  LANGUAGE wasm;

CREATE FUNCTION burn(v video_stream, a audio_stream DEFAULT NULL,
                     words cue[] DEFAULT NULL)
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/burn.wasm', 'burn'
  LANGUAGE wasm;

COPY (
  SELECT burn(f.video[1], f.audio[1], hear(f.audio[1])), f.audio[1]
  FROM input('tests/fixtures/av.mp4') f
) TO 'burned.mp4' WITH (video_codec 'libx264', crf 20, audio_codec 'aac')
```

```
$ ffrwd explain --delays -f query.sql
hear: tumbling 2 s
burn: per-frame; words by interval, no bound
burned.mp4 stream 0 (video): 2 s behind the source
burned.mp4 stream 1 (audio): 0 s behind the source, waits 2 s
```

`hear` is a tumbling window of 2 s, so its cues trail the sound by up to 2 s and nothing more; `burn`'s picture is 2 s behind the source, and the sound written beside it, read straight from the file, waits those 2 s at the muxer, which `compile` sizes. On a live input the same sums decide whether a query can run at all: a node that must act ahead of time (an ad decision that needs `announce_before_s`, a playout that needs `lead_s`) fed by a path later than that lead is refused at compile time as `LIVE_LEAD`, naming the node, the lead it needs and the delay of the path feeding it. A file run has no such rule, since nothing there is late.

## 154. A page with no inputs is a source

A module with no stream parameters is a source: it declares its outputs for its parameters, ticks at the rate it declares, and reads nothing. `ticker` draws a line of text crossing a canvas. `RETURNS source` puts it in FROM, where the alias carries the stream columns the module declared, and the query reads them as it reads a file's:

```pgsql
CREATE FUNCTION ticker(text text, width number DEFAULT 1280, height number DEFAULT 720,
                       fps number DEFAULT 30) RETURNS source
  AS '../sidecar/modules/target/wasm32-wasip2/release/ticker.wasm', 'ticker'
  LANGUAGE wasm;

COPY (
  SELECT s.video[1]
  FROM ticker('Nothing to see here') s
  WHERE s.t < 10
) TO 'ticker.mp4' WITH (video_codec 'libx264', crf 20)
```

```
$ ffrwd compile -f query.sql
ffrwd-wasm -m ticker=../sidecar/modules/target/wasm32-wasip2/release/ticker.wasm \
  -filter_complex \
  'ticker=text=Nothing\ to\ see\ here:width=1280:height=720:fps=30[video=out0]' -bound \
  'ticker=[]' -map '[out0]' -f nut pipe:1 | ffmpeg -copyts -f nut -analyzeduration 0 \
  -fpsprobesize 3 -to 10 -i pipe:0 -map 0:v:0 -c:0 libx264 -crf:0 20 ticker.mp4
```

In a file run the source runs as fast as its reader drains it; in a live run it is paced to the wall clock. `WHERE s.t < 10` ends it after ten seconds, as it would any source. A network source is the same shape with a clock of its own: it emits when it has something, and `shape` may reach the network at compile time to learn its outputs, as a manifest is probed. `ffrwd.blitz.compose` with no streams, a page that animates on its own, is this recipe's shape too.
