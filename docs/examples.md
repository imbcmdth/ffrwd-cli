# Cookbook

Real tasks. Every shown output on this page is real - a test reruns all of them and diffs the resulting ffmpeg commands, so if a recipe is here, it works.

Most recipes are parameterized (`:'source'`-style variables, filled by the `-v` flags in the shown command): swap the `-v` values and they run against your files. Recipe 33 explains the mechanism; [packages/ffrwd/](../packages/ffrwd/) collects ready-made ones as installable packages.

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

## 2. Remux into another container without re-encoding

`SELECT *` means keep everything: the container's stream arrays - video, audio, subtitle, data, in that order - untouched, tags intact, and chapters riding through as ffmpeg's own default. Nothing decodes; this runs as fast as the disk. The one wrinkle is captions - mp4 only carries `mov_text`, so the subtitle track transcodes while video and audio copy:

```pgsql
COPY (
  SELECT * FROM input('tests/fixtures/avs.mkv') a
) TO 'film.mp4' WITH (subtitle_codec 'mov_text')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/avs.mkv -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -map 0:s:0 \
  -metadata:s:2 language=eng -c:2 mov_text film.mp4
```

## 3. Extract the audio track to its own file

The SELECT list is the output. Select only the audio and that's the whole file - stream-copied, no generation loss:

```sql
COPY (
  SELECT f.audio[1]
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=soundtrack.m4a
ffmpeg -i film.mkv -map 0:a:0 -c:0 copy soundtrack.m4a
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

## 6. Rotate a phone video 90 degrees

For quarter turns, ffmpeg's `transpose` is the right tool (it swaps the axes rather than resampling). For arbitrary angles there's `rotate`, whose angle is an expression in radians - `rotate(f.video[1], '7*PI/180')` leans a clip seven degrees:

```pgsql
COPY (
  SELECT transpose(v.video[1], dir => 'clock'), v.audio[1]
  FROM input(:'source') v
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=phone.mp4 -v dest=upright.mp4
ffmpeg -i phone.mp4 -filter_complex '[0:v:0]transpose=dir=clock[out0]' -map '[out0]' \
  -map 0:a:0 -c:1 copy upright.mp4
```

## 7. Sharpen a soft-looking video

Any of your ffmpeg's filters is callable directly, options by name, checked against what the binary actually supports. (The one-knob version, if you don't need the fine control: `unsharp(f.video[1], 5, 5, 1.5)`, matrix sizes then amount, positionally in unsharp's own order.)

```pgsql
COPY (
  SELECT unsharp(a.video[1], luma_msize_x => 7, luma_amount => 1.5), a.audio[1]
  FROM input(:'source') a
) TO 'out.mp4'
```

```
$ ffrwd compile -f query.sql -v source=clip.mp4
ffmpeg -i clip.mp4 -filter_complex '[0:v:0]unsharp=luma_msize_x=7:luma_amount=1.5[out0]' \
  -map '[out0]' -map 0:a:0 -c:1 copy out.mp4
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

## 9. Watermark a video

`loop => true` keeps a still image alive for the whole duration, and the position is an ffmpeg expression - `(W-w)/2` centers it without you knowing either file's dimensions:

```pgsql
COPY (
  SELECT overlay(f.video[1], logo.video[1], '(W-w)/2', '(H-h)/2'), f.audio[1]
  FROM input(:'main') f, input(:'overlay', loop => true) logo
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v main=film.mp4 -v overlay=watermark.png -v dest=branded.mp4
ffmpeg -i film.mp4 -loop 1 -i watermark.png -filter_complex \
  '[0:v:0][1:v:0]overlay=x=(W-w)/2:y=(H-h)/2[out0]' -map '[out0]' -map 0:a:0 -c:1 copy \
  branded.mp4
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

## 11. Burn subtitles into the picture

Different from muxing a track: `subtitles()` is a video filter that renders the cues into the pixels. The subtitle file is read when ffmpeg runs, so it needs to exist then, not now:

```pgsql
COPY (
  SELECT subtitles(f.video[1], 'subs.en.srt'), f.audio[1]
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mp4 -v dest=burned.mp4
ffmpeg -i film.mp4 -filter_complex '[0:v:0]subtitles=filename=subs.en.srt[out0]' -map \
  '[out0]' -map 0:a:0 -c:1 copy burned.mp4
```

## 12. Speed a clip up 2x, picture and sound together

Two functions because the two stream types speed up differently: `ffrwd.speed` restamps video frames, `atempo` resamples audio while keeping the pitch (so nobody turns into a chipmunk):

```pgsql
COPY (
  SELECT ffrwd.speed(f.video[1], :factor), atempo(f.audio[1], :factor)
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mp4 -v factor=2 -v dest=fast.mp4
ffmpeg -i film.mp4 -filter_complex \
  '[0:v:0]setpts=PTS/2[out0];[0:a:0]atempo=tempo=2[out1]' -map '[out0]' -map '[out1]' \
  fast.mp4
```

## 13. Crossfade between two clips

`xfade` takes both clips, then `duration` and `offset` by name (its first option is the transition style, which defaults to a plain dissolve) - the offset is seconds into the FIRST clip where the fade begins, so a 10-second clip with a 1-second fade starts dissolving at 9. `acrossfade` does the same for the sound:

```pgsql
COPY (
  SELECT xfade(a.video[1], b.video[1], duration => 1, offset => 9),
         acrossfade(a.audio[1], b.audio[1], duration => 1)
  FROM input(:'first') a, input(:'second') b
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v first=one.mp4 -v second=two.mp4 -v dest=dissolve.mp4
ffmpeg -i one.mp4 -i two.mp4 -filter_complex \
  '[0:v:0][1:v:0]xfade=duration=1:offset=9[out0];'\
'[0:a:0][1:a:0]acrossfade=duration=1[out1]' -map '[out0]' -map '[out1]' dissolve.mp4
```

## 14. Turn a clip into a GIF

The good-looking way needs two passes over the frames - one to build a palette, one to use it. Write it as a CTE consumed twice; the compiler inserts the split:

```pgsql
COPY (
  WITH small AS (
    SELECT fps(scale(v.video[1], 480, -2), 12) AS frame
    FROM input(:'source') v
  )
  SELECT paletteuse(small.frame, palettegen(small.frame))
  FROM small
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=clip.mp4 -v dest=clip.gif
ffmpeg -i clip.mp4 -filter_complex \
  '[0:v:0]scale=width=480:height=-2,fps=fps=12,split=2[n2_split0][n2_split1];'\
'[n2_split0]palettegen[n3];[n2_split1][n3]paletteuse[out0]' -map '[out0]' clip.gif
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

## 16. Picture-in-picture

A quarter-size camera in the bottom-right corner, 20 pixels off each edge - the expressions mean the position holds whatever the two resolutions are. (The dual-language version, with the audio mixed per language, is the README's opening demo.)

```pgsql
COPY (
  SELECT overlay(f.video[1], scale(c.video[1], 'iw/4', -2), 'W-w-20', 'H-h-20'), f.audio[1]
  FROM input(:'main') f, input(:'overlay') c
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v main=film.mp4 -v overlay=camera.mp4 -v dest=pip.mp4
ffmpeg -i film.mp4 -i camera.mp4 -filter_complex \
  '[1:v:0]scale=width=iw/4:height=-2[n1];[0:v:0][n1]overlay=x=W-w-20:y=H-h-20[out0]' \
  -map '[out0]' -map 0:a:0 -c:1 copy pip.mp4
```

## 17. Insert a clip at a timestamp

The splice: cut away to the insert, then resume. The same file appears under two aliases with two windows, and the tail's `>= 120` means "to the end" with no made-up end time:

```sql
COPY (
  SELECT f.video[1], f.audio[1] FROM input(:'main') f WHERE f.t <= :cut
  UNION ALL
  SELECT ad.video[1], ad.audio[1] FROM input(:'insert') ad
  UNION ALL
  SELECT g.video[1], g.audio[1] FROM input(:'main') g WHERE g.t >= :cut
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v main=film.mp4 -v insert=promo.mp4 -v cut=120 -v dest=spliced.mp4
ffmpeg -to 120 -i film.mp4 -i promo.mp4 -ss 120 -i film.mp4 -filter_complex \
  '[0:v:0][0:a:0][1:v:0][1:a:0][2:v:0][2:a:0]concat=n=3:v=1:a=1[out0][out1]' -map \
  '[out0]' -map '[out1]' spliced.mp4
```

Or keep the main video playing and overlay the insert on top: a delayed video stream is transparent until its start time (and after it ends), so it composes with a plain `overlay` - no timeline bookkeeping:

```pgsql
COPY (
  SELECT overlay(f.video[1], ffrwd.delay(promo.video[1], 120), 20, 20), f.audio[1]
  FROM input(:'main') f, input(:'insert') promo
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v main=film.mp4 -v insert=promo.mp4 -v dest=overlaid.mp4
ffmpeg -i film.mp4 -i promo.mp4 -filter_complex \
  '[1:v:0]format=pix_fmts=yuva420p,tpad=start_duration=120:stop=1:color=black@0[n2];'\
'[0:v:0][n2]overlay=x=20:y=20[out0]' -map '[out0]' -map 0:a:0 -c:1 copy overlaid.mp4
```

## 18. Normalize loudness on every language track at once

A bare `.audio` is the whole track array; handing it to a filter broadcasts, one node per language, and every output keeps its language tag. (`ffmpeg.loudnorm` rather than bare `loudnorm` only out of habit here - the bare name works too; the namespace is the spelling that never collides with Postgres grammar. `I` is EBU R128 integrated loudness, and yes, it's a capital I.)

```pgsql
COPY (
  SELECT f.video[1], ffmpeg.loudnorm(f.audio, I => -23)
  FROM input('tests/fixtures/av2.mp4') f
) TO 'broadcast.mkv'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -filter_complex \
  '[0:a:0]loudnorm=I=-23[out1];[0:a:1]loudnorm=I=-23[out2]' -map 0:v:0 -c:0 copy -map \
  '[out1]' -metadata:s:1 language=eng -map '[out2]' -metadata:s:2 language=fra \
  broadcast.mkv
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

## 20. Generate test media

Sources live in FROM and consume no input file at all - note the command below has no `-i`:

```pgsql
COPY (
  SELECT t.video[1], s.audio[1]
  FROM ffmpeg.testsrc2(duration => 10, size => '1280x720', rate => 30) t,
       ffmpeg.sine(frequency => 440, duration => 10) s
) TO 'bars.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -filter_complex \
  'testsrc2=duration=10:size=1280x720:rate=30[out0];'\
'sine=frequency=440:duration=10[out1]' -map '[out0]' -map '[out1]' bars.mp4
```

They also solve a quieter problem: `UNION ALL` branches must match column for column, so appending a slate to a clip needs a silent audio track from somewhere. `anullsrc` is that somewhere:

```pgsql
COPY (
  SELECT f.video[1], f.audio[1] FROM input(:'source') f
  UNION ALL
  SELECT t.video[1], s.audio[1]
  FROM ffmpeg.color(color => 'black', duration => 3, size => '1280x720', rate => 30) t,
       ffmpeg.anullsrc(duration => 3) s
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=clip.mp4 -v dest=with-slate.mp4
ffmpeg -i clip.mp4 -filter_complex \
  'color=color=black:duration=3:size=1280x720:rate=30[n1];anullsrc=duration=3[n2];'\
'[0:v:0][0:a:0][n1][n2]concat=n=2:v=1:a=1[out0][out1]' -map '[out0]' -map '[out1]' \
  with-slate.mp4
```

## 21. Split a stereo track, or compress it in bands

A few filters return a whole array, sized by one of their own options. `channelsplit` turns one stereo track into two mono streams; splatted into the SELECT list, each becomes its own output:

```pgsql
COPY (
  SELECT ffmpeg.channelsplit(a.audio[1])
  FROM input(:'source') a
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=stereo.mp4 -v dest=channels.mkv
ffmpeg -i stereo.mp4 -filter_complex '[0:a:0]channelsplit[out0][out1]' -map '[out0]' \
  -map '[out1]' channels.mkv
```

`acrossover` splits by frequency instead - two split points make three bands - and that's the shape of multiband compression: split, compress each band on its own settings, mix back:

```pgsql
COPY (
  WITH bands AS (
    SELECT ffmpeg.acrossover(a.audio[1], split => '300 3000') AS b
    FROM input(:'source') a
  )
  SELECT amix(amix(acompressor(bands.b[1], threshold => 0.1, ratio => 4),
                   acompressor(bands.b[2], threshold => 0.05, ratio => 6)),
              acompressor(bands.b[3], threshold => 0.1, ratio => 4))
  FROM bands
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=song.m4a -v dest=mastered.m4a
ffmpeg -i song.m4a -filter_complex \
  '[0:a:0]acrossover=split=300\ 3000[n10][n11][n12];'\
'[n10]acompressor=threshold=0.1:ratio=4[n2];'\
'[n11]acompressor=threshold=0.05:ratio=6[n3];[n2][n3]amix=inputs=2[n4];'\
'[n12]acompressor=threshold=0.1:ratio=4[n5];[n4][n5]amix=inputs=2[out0]' -map '[out0]' \
  mastered.m4a
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

## 24. Extract captions by language

Caption arrays unnest the same way (columns: `tags`, `codec`), so pulling the English subtitles out of a many-language file is a `WHERE`, not a subscript:

```pgsql
COPY (
  SELECT s
  FROM input('tests/fixtures/avs.mkv') f, unnest(f.subtitle) s
  WHERE s.tags.language = 'eng'
) TO 'subs.srt'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/avs.mkv -map 0:s:0 -c:0 copy -metadata:s:0 language=eng \
  subs.srt
```

## 25. Mix two files' tracks pairwise, matched by language

Two multi-language files, and every track should mix with its counterpart - English with English, French with French, whatever order each file stores them in. That is a JOIN, written exactly the way Postgres writes it, evaluated entirely at compile time (the metadata is probed, so ffmpeg only ever sees the wiring the join decided). The join leaves one row per pair, and `array_agg` is what puts all those pairs in one file:

```pgsql
COPY (
  SELECT array_agg(amix(a, b))
  FROM input('tests/fixtures/av2.mp4') f, input('tests/fixtures/av3.mp4') g,
       unnest(f.audio) a JOIN unnest(g.audio) b ON a.tags.language = b.tags.language
) TO 'mixed.mka'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -i tests/fixtures/av3.mp4 -filter_complex \
  '[0:a:0][1:a:0]amix=inputs=2[out0];[0:a:1][1:a:1]amix=inputs=2[out1]' -map '[out0]' \
  -metadata:s:0 language=eng -map '[out1]' -metadata:s:1 language=fra mixed.mka
```

Result rows follow the LEFT side's track order, so the output track order is `f`'s - track order is player-visible surface, and nothing here resorts it. And when one file carries two English tracks (a 5.1 and a stereo, say), that's not an error, it's two pairs - real join semantics - and the fix is a wider key: `ON a.tags.language = b.tags.language AND a.channel_layout = b.channel_layout`.

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

## 28. Side by side, matched by resolution

Video arrays unnest too - `width`, `height`, `fps`, `codec`, `bitrate` are the columns - so pairing renditions for a comparison strip is a join on the numbers that matter:

```pgsql
COPY (
  SELECT hstack(a, b)
  FROM input('tests/fixtures/testsrc.mp4') f, input('tests/fixtures/smptebars.mp4') g,
       unnest(f.video) a JOIN unnest(g.video) b
         ON a.width = b.width AND a.height = b.height
) TO 'sxs.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/testsrc.mp4 -i tests/fixtures/smptebars.mp4 -filter_complex \
  '[0:v:0][1:v:0]hstack=inputs=2[out0]' -map '[out0]' sxs.mp4
```

A video gap in an outer join fills with `COALESCE(b, ffmpeg.color())` - black by default, size, rate and duration inherited from the paired row. A caption gap fills with `COALESCE(b, ffrwd.empty_captions())`: the track exists and takes its language tag, it just contains zero cues - nobody generates your subtitles for you.

## 29. Assert what you're shipping

A subscripted track has the same metadata columns a row does: `f.audio[1].tags.language` is the first track's tag, right there in a `WHERE`. Since the predicate evaluates at compile time, this is an assertion - if track 1 isn't English, the script refuses to compile instead of quietly shipping the wrong language. (The strictly-Postgres spelling `(f.audio[1]).tags.language` works too.)

```pgsql
COPY (
  SELECT f.audio[1] FROM input('tests/fixtures/av2.mp4') f
  WHERE f.audio[1].tags.language = 'eng'
) TO 'eng.m4a'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -map 0:a:0 -c:0 copy -metadata:s:0 language=eng eng.m4a
```

Recipe 23 answers "give me whichever track is English"; this one answers "I believe track 1 is English - stop me if I'm wrong". Same wiring out the other end, different contract.

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

## 31. Inspect a join before you trust it

Stream-valued cells print as placeholders carrying the stream spec, so a table query over a join shows exactly which track paired with which - and an empty cell is an outer join's gap, before you've committed to a fill:

```pgsql
SELECT a.tags.language, a AS film, b AS promo
FROM input('tests/fixtures/av2.mp4') f, input('tests/fixtures/av-eng.mp4') g,
     unnest(f.audio) a FULL OUTER JOIN unnest(g.audio) b ON a.tags.language = b.tags.language
```

```
$ ffrwd -f query.sql
 language | film          | promo
----------+---------------+---------------
 eng      | <audio 0:a:0> | <audio 1:a:0>
 fra      | <audio 0:a:1> |
(2 rows)
```

## 32. Export track metadata as CSV

`COPY ... TO STDOUT WITH (FORMAT csv)` is stock Postgres, and here it makes the table query scriptable - pipe it wherever your inventory lives. `TO '<path>.csv'` writes a file instead; `header true` adds the column row:

```pgsql
COPY (
  SELECT t.tags.language, t.codec, t.channel_layout
  FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) t
) TO STDOUT WITH (format 'csv', header true)
```

```
$ ffrwd -f query.sql
language,codec,channel_layout
eng,aac,mono
fra,aac,mono
```

## 33. One query, many files

A query file with variables is a recipe. `-v name=value` is psql's own flag and `:'name'` is psql's own interpolation - the value lands as a properly escaped string literal (bare `:name` substitutes raw, for numbers), and an undefined variable is a compile-time error, not a surprise:

```sql
COPY (SELECT f.video[1], f.audio[1] FROM input(:'source') f)
TO :'dest' WITH (video_codec 'libx264', crf 20, audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=out.mkv
ffmpeg -i film.mkv -map 0:v:0 -map 0:a:0 -c:0 libx264 -crf:0 20 -c:1 aac out.mkv
```

Swap the `-v` values and the same file transcodes anything. The [packages/imbcmdth/](../packages/imbcmdth/) packages collect ready-to-run recipes built this way.

## 34. Grab a poster frame

`frames 1` stops the output after one frame, and `video_codec 'png'` forces the decode a PNG needs (an unfiltered stream would otherwise try to stream-copy). The seek puts the frame where you want it:

```pgsql
COPY (
  SELECT f.video[1] FROM input(:'source') f WHERE f.t >= :at
) TO :'dest' WITH (video_codec 'png', frames 1)
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v at=90 -v dest=poster.png
ffmpeg -ss 90 -i film.mkv -map 0:v:0 -c:0 png -frames:0 1 poster.png
```

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

## 36. Keep the last minute

`seek_end` seeks from the END of the file - no need to know its length. Stream copy applies, keyframe snapping included, same as any input seek:

```pgsql
COPY (
  SELECT a.video[1], a.audio[1]
  FROM input(:'source', seek_end => 60) a
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=clip.mp4 -v dest=tail.mp4
ffmpeg -sseof -60 -i clip.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy tail.mp4
```

## 37. Retitle tracks from their own metadata

A non-stream column in a media query sets a tag on that row's output. The alias names the tag, the value is any compile-time expression over the row - here a title built from the language tag with `||`:

```pgsql
COPY (
  SELECT t, STRUCT('Audio (' || t.tags.language || ')' AS title) AS tags
  FROM input('tests/fixtures/av-eng.mp4') f, unnest(f.audio) t
) TO 'titled.mka'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av-eng.mp4 -map 0:a:0 -c:0 copy -metadata:s:0 language=eng \
  -metadata:s:0 'title=Audio (eng)' titled.mka
```

## 38. Normalize language tags

CASE makes the edit conditional, and it runs over every row - one expression fixes the whole file. A `tags` column sets the keys it names and leaves the rest alone; a `NULL` field clears its key. Rows are tracks inside the `WITH`, which is where a `tags` column lands on one; the outer `array_agg` puts the tagged tracks in the file:

```pgsql
COPY (
  WITH retagged AS (
    SELECT t AS track,
           STRUCT(CASE WHEN t.tags.language = 'fra' THEN 'fre'
                       ELSE t.tags.language END AS language) AS tags
    FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) t
  )
  SELECT array_agg(retagged.track) FROM retagged
) TO 'retagged.mka'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -map 0:a:0 -c:0 copy -metadata:s:0 language=eng -map \
  0:a:1 -c:1 copy -metadata:s:1 language=fre retagged.mka
```

## 39. List a file's chapters

`chapters` is an array column of the input, like the stream arrays: unnest it and each chapter is a row, straight from the container. Like every metadata query, no COPY means it prints and nothing runs:

```pgsql
SELECT c.index, c.title, c.start_t, c.end_t
FROM input('tests/fixtures/av-chapters.mkv') f, unnest(f.chapters) c
```

```
$ ffrwd -f query.sql
 index | title     | start_t | end_t
-------+-----------+---------+-------
 1     | Intro     | 0.0     | 1.0
 2     | Chapter 1 | 1.0     | 2.0
 3     | Chapter 2 | 2.0     | 3.0
 4     | Credits   | 3.0     | 4.0
(4 rows)
```

## 40. Write chapters

A `chapters` column IS the file's chapter list, the same shape `unnest(f.chapters)` reads. Build it from `chapter` records; it compiles to one extra self-contained input - no file on disk:

```pgsql
COPY (
  SELECT f.video[1], f.audio[1],
         ARRAY[STRUCT('Intro' AS title, 0 AS start_t, 60 AS end_t)::chapter,
               STRUCT('Act One' AS title, 60 AS start_t, 300 AS end_t)::chapter]
           AS chapters
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=chaptered.mkv
ffmpeg -i film.mkv -f ffmetadata -i \
  'data:text/plain;'\
'base64,'\
'O0ZGTUVUQURBVEExCltDSEFQVEVSXQpUSU1FQkFTRT0xLzEKU1RBUlQ9MApFTkQ9NjAKdGl0bGU9SW50cm8KW0NIQVBURVJdClRJTUVCQVNFPTEvMQpTVEFSVD02MApFTkQ9MzAwCnRpdGxlPUFjdCBPbmUK' \
  -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -map_chapters 1 chaptered.mkv
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

## 42. Title the file and keep its global tags

In a query without track rows a `tags` column is the container's map. Naming an input's own `tags` on the left of `||` copies that input's globals through, and the keys on the right override theirs:

```pgsql
COPY (
  SELECT f.video[1], f.audio[1], f.tags || STRUCT('Director Cut' AS title) AS tags
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=cut.mkv
ffmpeg -i film.mkv -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -map_metadata 0 -metadata \
  'title=Director Cut' cut.mkv
```

## 43. Two-pass encode to a target bitrate

`two_pass true` compiles to TWO chained ffmpeg commands: pass 1 encodes video only into ffmpeg's stats file and discards the output, pass 2 reads the stats and writes the file. `run` executes both in order; requires `video_bitrate` (two-pass exists to hit a bitrate):

```pgsql
COPY (SELECT f.video[1], f.audio[1] FROM input(:'source') f)
TO :'dest' WITH (video_codec 'libx264', video_bitrate '2500k', two_pass true, audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql -v source=in.mkv -v dest=out.mp4
ffmpeg -i in.mkv -map 0:v:0 -c:0 libx264 -b:0 2500k -pass 1 -passlogfile out.mp4 -f null \
  - && ffmpeg -i in.mkv -map 0:v:0 -map 0:a:0 -c:0 libx264 -b:0 2500k -pass 2 \
  -passlogfile out.mp4 -c:1 aac out.mp4
```

## 44. Merge two audio tracks into one

`amerge` combines tracks into a single multichannel stream (unlike `amix`, which sums them):

```pgsql
COPY (
  SELECT amerge(a.audio[1], b.audio[1])
  FROM input(:'first') a, input(:'second') b
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v first=one.mp4 -v second=two.mp4 -v dest=merged.mka
ffmpeg -i one.mp4 -i two.mp4 -filter_complex '[0:a:0][1:a:0]amerge=inputs=2[out0]' -map \
  '[out0]' merged.mka
```

## 45. Scale each track relative to itself

A filter argument over a row table's columns is computed per row, at compile time - each rendition scaled against its own probed width:

```pgsql
COPY (
  SELECT scale(t, t.width / 2, -2)
  FROM input('tests/fixtures/av2.mp4') f, unnest(f.video) t
) TO 'half.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -filter_complex \
  '[0:v:0]scale=width=160:height=-2[out0]' -map '[out0]' half.mp4
```

## 46. Keep everything but the end

`f.duration` is the probed container duration, and trim bounds take arithmetic - so "all but the last half second" needs no known length:

```pgsql
COPY (
  SELECT f.video[1], f.audio[1]
  FROM input('tests/fixtures/av2.mp4') f
  WHERE f.t <= f.duration - 0.5
) TO 'trimmed.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -to 3.5 -i tests/fixtures/av2.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy \
  -metadata:s:1 language=eng trimmed.mp4
```

## 47. Split a file by its chapters

A `TO` expression over a row table's columns means one output file per row - the chapters drive the seeks and the filenames. Two ways to cut, and the trade matters:

**Stream copy** - fastest, nothing decodes, but each cut snaps back to the previous keyframe, so pieces can start early. Copied trims need one ffmpeg command per piece, chained with `&&`:

```pgsql
COPY (
  SELECT f.video, f.audio
  FROM input('tests/fixtures/av-chapters.mkv') f, unnest(f.chapters) c
  WHERE f.t BETWEEN c.start_t AND c.end_t
) TO ('ch-' || c.title || '.mkv')
```

```
$ ffrwd compile -f query.sql
ffmpeg -ss 0.0 -to 1.0 -i tests/fixtures/av-chapters.mkv -map 0:v:0 -c:0 copy -map 0:a:0 \
  -c:1 copy -metadata:s:1 language=eng -map 0:a:1 -c:2 copy -metadata:s:2 language=fra \
  ch-Intro.mkv && ffmpeg -ss 1.0 -to 2.0 -i tests/fixtures/av-chapters.mkv -map 0:v:0 \
  -c:0 copy -map 0:a:0 -c:1 copy -metadata:s:1 language=eng -map 0:a:1 -c:2 copy \
  -metadata:s:2 language=fra 'ch-Chapter 1.mkv' && ffmpeg -ss 2.0 -to 3.0 -i \
  tests/fixtures/av-chapters.mkv -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -metadata:s:1 \
  language=eng -map 0:a:1 -c:2 copy -metadata:s:2 language=fra 'ch-Chapter 2.mkv' && \
  ffmpeg -ss 3.0 -to 4.0 -i tests/fixtures/av-chapters.mkv -map 0:v:0 -c:0 copy -map \
  0:a:0 -c:1 copy -metadata:s:1 language=eng -map 0:a:1 -c:2 copy -metadata:s:2 \
  language=fra ch-Credits.mkv
```

**Re-encode** - frame-accurate cuts, and the whole split is ONE command: the source decodes once no matter how many chapters, with each output taking its own `-ss`/`-to`:

```pgsql
COPY (
  SELECT f.video, f.audio
  FROM input('tests/fixtures/av-chapters.mkv') f, unnest(f.chapters) c
  WHERE f.t BETWEEN c.start_t AND c.end_t
) TO ('ch-' || c.title || '.mkv') WITH (video_codec 'libx264', crf 18, audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av-chapters.mkv -ss 0.0 -to 1.0 -map 0:v:0 -map 0:a:0 \
  -metadata:s:1 language=eng -map 0:a:1 -metadata:s:2 language=fra -c:0 libx264 -crf:0 \
  18 -c:1 aac -c:2 aac ch-Intro.mkv -ss 1.0 -to 2.0 -map 0:v:0 -map 0:a:0 -metadata:s:1 \
  language=eng -map 0:a:1 -metadata:s:2 language=fra -c:0 libx264 -crf:0 18 -c:1 aac \
  -c:2 aac 'ch-Chapter 1.mkv' -ss 2.0 -to 3.0 -map 0:v:0 -map 0:a:0 -metadata:s:1 \
  language=eng -map 0:a:1 -metadata:s:2 language=fra -c:0 libx264 -crf:0 18 -c:1 aac \
  -c:2 aac 'ch-Chapter 2.mkv' -ss 3.0 -to 4.0 -map 0:v:0 -map 0:a:0 -metadata:s:1 \
  language=eng -map 0:a:1 -metadata:s:2 language=fra -c:0 libx264 -crf:0 18 -c:1 aac \
  -c:2 aac ch-Credits.mkv
```

The chain is the exception, not the rule: it survives only while EVERY stream of every piece is a stream copy. Name one codec, or wrap one column in a filter, and the whole split becomes the single invocation above - the streams you left alone go along with it, taking the container's default encoder instead of `-c copy`.

## 48. Extract every language to its own file

The same rule over track rows: each row's stream goes to a filename built from its own metadata:

```pgsql
COPY (SELECT t FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) t)
TO (t.tags.language || '.m4a')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -map 0:a:0 -c:0 copy -metadata:s:0 language=eng eng.m4a \
  -map 0:a:1 -c:0 copy -metadata:s:0 language=fra fra.m4a
```

## 49. Normalize loudness properly (two-pass)

`ffrwd.loudnorm2` measures first and corrects second - the broadcast-compliant way. It compiles to a shell chain: pass 1 prints measurements, `ffrwd loudnorm2env` turns them into environment variables, and pass 2 splices them into its filter. (POSIX shells only; `run` does the substitution itself and works everywhere. This is the one command line the cookbook shows unwrapped - its quoting cannot be split.)

```pgsql
COPY (
  SELECT ffrwd.loudnorm2(f.audio[1], I => -16, TP => -1.5, LRA => 11)
  FROM input(:'source') f
) TO :'dest' WITH (audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=out.m4a
eval "$(ffmpeg -i film.mkv -filter_complex '[0:a:0]loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json[out0]' -map '[out0]' -f null - 2>&1 | ffrwd loudnorm2env)" && ffmpeg -i film.mkv -filter_complex '[0:a:0]loudnorm=I=-16:TP=-1.5:LRA=11:measured_I='"${FFRWD_LN_I}"':measured_TP='"${FFRWD_LN_TP}"':measured_LRA='"${FFRWD_LN_LRA}"':measured_thresh='"${FFRWD_LN_THRESH}"':offset='"${FFRWD_LN_OFFSET}"':linear=true[out0]' -map '[out0]' -c:0 aac out.m4a
```

## 50. Stream instead of writing a file

A sink path can be a protocol URL - rtmp, srt, udp - and ffmpeg owns it from there. Name the muxer with `format` (a URL has no extension to infer from):

```pgsql
COPY (SELECT f.video[1], f.audio[1] FROM input(:'source') f)
TO :'dest' WITH (format 'flv', video_codec 'libx264', audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=rtmp://live.example.com/app/streamkey
ffmpeg -i film.mkv -map 0:v:0 -map 0:a:0 -f flv -c:0 libx264 -c:1 aac \
  rtmp://live.example.com/app/streamkey
```

For SRT use `format 'mpegts'` with an `srt://` destination; UDP the same. Verified end to end: a query streamed over `udp://` to a listening receiver arrives intact, video and audio.

## 51. Set the container's title, clear its artist

In a query without track rows, a `tags` column is the CONTAINER's map - each field name is a key, free-form, same as track-row tags. A `NULL` field clears the key in the output (ffmpeg copies input globals by default, so clearing is explicit):

```pgsql
COPY (
  SELECT f.video[1], f.audio[1],
         STRUCT('Remastered 2026' AS title, NULL AS artist) AS tags
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=out.mkv
ffmpeg -i film.mkv -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -metadata artist= -metadata \
  'title=Remastered 2026' out.mkv
```

## 52. Read the container's tags, rewrite them with CASE

Container tags are a map on the input alias, read by path - `f.tags.title`, `f.tags.artist`, `f.tags.comment`, any key the file carries - NULL when it doesn't carry them. So the full CASE toolkit works: fill missing tags, build new ones from old:

```pgsql
COPY (
  SELECT f.video[1], f.audio[1],
    STRUCT(f.tags.title || ' (restored)' AS title,
           CASE WHEN f.tags.comment IS NULL THEN 'no notes'
                ELSE f.tags.comment END AS comment) AS tags
  FROM input('tests/fixtures/tagged.mp4') f
) TO 'restored.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/tagged.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -metadata \
  'comment=no notes' -metadata 'title=Angel One (restored)' restored.mp4
```

Reading needs the probe (the values live in the file), so this one is fixture-bound. The same paths work in table queries: `select f.tags.title, f.tags.artist, f.duration from input('movie.mp4') f` prints them, and a bare `f.tags` prints the whole map.

## 53. Tag the tracks and the container in one query

Two levels, two scopes, visible in the query text: inside the `WITH`, rows are tracks, so the `tags` column titles each stream; outside it, the CTE's streams are just streams, so the `tags` column titles the container:

```pgsql
COPY (
  WITH tagged AS (
    SELECT a AS track, STRUCT('Audio (' || a.tags.language || ')' AS title) AS tags
    FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) a
  )
  SELECT g.video, array_agg(tagged.track), STRUCT('Director Cut' AS title) AS tags
  FROM input('tests/fixtures/av2.mp4') g, tagged
  GROUP BY g.video
) TO 'out.mkv'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -metadata:s:1 \
  language=eng -metadata:s:1 'title=Audio (eng)' -map 0:a:1 -c:2 copy -metadata:s:2 \
  language=fra -metadata:s:2 'title=Audio (fra)' -metadata 'title=Director Cut' out.mkv
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

## 56. Preview a grouped shape as a table

Grouping works in table queries too - drop the COPY and the same relation prints instead of writing, one row per group, arrays in braces. The single-group form shows what a one-file COPY would carry:

```pgsql
SELECT f.video, array_agg(a)
FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) a
GROUP BY f.video
```

```
$ ffrwd -f query.sql
 video           | array_agg
-----------------+-------------------------------
 {<video 0:v:0>} | {<audio 0:a:0>,<audio 0:a:1>}
(1 row)
```

And grouping by a row column previews a fan-out's partitions before any file is written - here, recipe 55's per-language split:

```pgsql
SELECT a.tags.language, array_agg(a)
FROM input('tests/fixtures/av-2eng.mp4') f, unnest(f.audio) a
GROUP BY a.tags.language
```

```
$ ffrwd -f query.sql
 language | array_agg
----------+-------------------------------
 eng      | {<audio 0:a:0>,<audio 0:a:1>}
 fra      | {<audio 0:a:2>}
(2 rows)
```

## 57. Combine tracks selected by separate CTEs

Each CTE picks its tracks with its own WHERE; the outer query is plain SQL over their rows - `FROM vid, aud` is a cross join, so gather the audio with `array_agg` and group by the video to get one row. The table form previews it; wrap it in COPY and the same relation becomes the file:

```pgsql
WITH vid AS (
  SELECT v AS track FROM input('tests/fixtures/av-2eng.mp4') i1, unnest(i1.video) v
),
aud AS (
  SELECT a AS track FROM input('tests/fixtures/av-2eng.mp4') i2, unnest(i2.audio) a
  WHERE a.tags.language = 'eng'
)
SELECT vid.track, array_agg(aud.track) FROM vid, aud GROUP BY vid.track
```

```
$ ffrwd -f query.sql
 track         | array_agg
---------------+-------------------------------
 <video 0:v:0> | {<audio 0:a:0>,<audio 0:a:1>}
(1 row)
```

The same SELECT inside `COPY (...) TO 'combo.mkv'` compiles to:

```pgsql
COPY (
  WITH vid AS (
    SELECT v AS track FROM input('tests/fixtures/av-2eng.mp4') i1, unnest(i1.video) v
  ),
  aud AS (
    SELECT a AS track FROM input('tests/fixtures/av-2eng.mp4') i2, unnest(i2.audio) a
    WHERE a.tags.language = 'eng'
  )
  SELECT vid.track, array_agg(aud.track) FROM vid, aud GROUP BY vid.track
) TO 'combo.mkv'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av-2eng.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy \
  -metadata:s:1 language=eng -map 0:a:1 -c:2 copy -metadata:s:2 language=eng combo.mkv
```

## 58. Burn a title onto the picture

`drawtext` works out of the box; the font is an option like any other, so name one - fontconfig fallbacks vary by build, and a named file is the same everywhere:

```pgsql
COPY (
  SELECT drawtext(f.video[1], text => :'text', fontfile => :'font', fontsize => 48, x => 20, y => 20, fontcolor => 'white'),
         f.audio[1]
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v text=Hello -v font=arial.ttf -v source=film.mkv -v dest=titled.mp4
ffmpeg -i film.mkv -filter_complex \
  '[0:v:0]drawtext=text=Hello:fontfile=arial.ttf:fontsize=48:x=20:y=20:fontcolor=white[out0]' \
  -map '[out0]' -map 0:a:0 -c:1 copy titled.mp4
```

## 59. Turn an image sequence into a video, and back

An image sequence is an input like any other - ffmpeg's `%04d` pattern names the files, and `framerate` says how fast to play them:

```sql
COPY (SELECT f.video[1] FROM input(:'frames', framerate => 24) f)
TO :'dest' WITH (video_codec 'libx264', crf 18)
```

```
$ ffrwd compile -f query.sql -v frames=frames/%04d.png -v dest=out.mp4
ffmpeg -framerate 24 -i frames/%04d.png -map 0:v:0 -c:0 libx264 -crf:0 18 out.mp4
```

The reverse is a pattern in the destination - here one frame per second:

```pgsql
COPY (SELECT fps(f.video[1], 1) FROM input(:'source') f) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=frame-%04d.png
ffmpeg -i film.mkv -filter_complex '[0:v:0]fps=fps=1[out0]' -map '[out0]' frame-%04d.png
```

## 60. Draw a waveform for an audio file

`showwaves` is an audio-to-video filter: it takes the track and returns a picture. Select the same track again as audio and the result is a video with sound - the compiler splits the stream for you:

```pgsql
COPY (
  SELECT showwaves(f.audio[1], size => '1280x240', mode => 'line'), f.audio[1]
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=song.mp3 -v dest=waves.mp4
ffmpeg -i song.mp3 -filter_complex \
  '[0:a:0]asplit=2[src_f_a_0_split0][out1];'\
'[src_f_a_0_split0]showwaves=size=1280x240:mode=line[out0]' -map '[out0]' -map '[out1]' \
  waves.mp4
```

## 61. Record a stream

A URL is an input path; ffmpeg owns the protocol. Stream-copy a live HLS or RTMP source straight to disk:

```sql
COPY (SELECT f.video[1], f.audio[1] FROM input(:'url') f) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v url=https://example.com/live/stream.m3u8 -v dest=capture.mp4
ffmpeg -i https://example.com/live/stream.m3u8 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy \
  capture.mp4
```

Add `WITH (duration 60)` to stop after a minute; per-protocol options (headers, transports) are not expressible yet - see [known_gaps.md](known_gaps.md).

## 62. Use a plugin filter

`frei0r` loads effect plugins at runtime (most builds ship it enabled); its options name the plugin and pass its parameters, and it compiles like any other filter:

```pgsql
COPY (
  SELECT frei0r(f.video[1], filter_name => 'glow', filter_params => '0.5'), f.audio[1]
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=glow.mp4
ffmpeg -i film.mkv -filter_complex \
  '[0:v:0]frei0r=filter_name=glow:filter_params=0.5[out0]' -map '[out0]' -map 0:a:0 -c:1 \
  copy glow.mp4
```

ffmpeg finds plugins through the `FREI0R_PATH` environment variable. Audio plugins go through `ladspa` the same way.

## 63. Copy or rebuild a chapter list

`g.chapters AS chapters` takes another file's chapters wholesale, and `NULL AS chapters` writes none at all:

```pgsql
COPY (
  SELECT f.video[1], f.audio[1], g.chapters AS chapters
  FROM input('tests/fixtures/av2.mp4') f, input('tests/fixtures/av-chapters.mkv') g
) TO 'borrowed.mkv'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -i tests/fixtures/av-chapters.mkv -map 0:v:0 -c:0 copy \
  -map 0:a:0 -c:1 copy -metadata:s:1 language=eng -map_chapters 1 borrowed.mkv
```

Gathering rows builds one instead. A written row table is just another row source, so this compiles to exactly the same command as [recipe 40](#40-write-chapters) - two spellings, one file:

```sql
COPY (
  SELECT f.video[1], f.audio[1],
         array_agg(STRUCT(m.title AS title, m.start_t AS start_t,
                          m.end_t AS end_t)::chapter) AS chapters
  FROM input(:'source') f,
       unnest(ARRAY[STRUCT(0 AS start_t, 60 AS end_t, 'Intro' AS title),
                    STRUCT(60 AS start_t, 300 AS end_t, 'Act One' AS title)]) m
  GROUP BY f.video[1], f.audio[1]
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=chaptered.mkv
ffmpeg -i film.mkv -f ffmetadata -i \
  'data:text/plain;'\
'base64,'\
'O0ZGTUVUQURBVEExCltDSEFQVEVSXQpUSU1FQkFTRT0xLzEKU1RBUlQ9MApFTkQ9NjAKdGl0bGU9SW50cm8KW0NIQVBURVJdClRJTUVCQVNFPTEvMQpTVEFSVD02MApFTkQ9MzAwCnRpdGxlPUFjdCBPbmUK' \
  -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -map_chapters 1 chaptered.mkv
```

## 64. Read a subtitle file's cues

A WebVTT track's cues are rows, the way a container's chapters are - `index`, `start_t`, `end_t` and the cue `text`. ffprobe does not enumerate them, so ffrwd reads the file itself:

```pgsql
SELECT c.index, c.start_t, c.end_t, c.text
FROM input('tests/fixtures/subs.en.vtt') v, unnest(v.cues) c
```

```
$ ffrwd -f query.sql
 index | start_t | end_t | text
-------+---------+-------+------------
 1     | 0.0     | 0.6   | Cue one.
 2     | 0.7     | 1.3   | Cue two.
 3     | 1.4     | 2.0   | Cue three.
(3 rows)
```

## 65. Turn chapters into a subtitle track, and back

Cues and chapters are the same shape - a title over a time span - so converting either way is an `array_agg` over the other's rows. WebVTT is what HLS uses for chapter metadata, so this is the canonical export:

```pgsql
COPY (
  SELECT f.video[1], f.audio[1],
         array_agg(STRUCT(c.title AS text, c.start_t AS start_t,
                          c.end_t AS end_t)::cue)
  FROM input('tests/fixtures/av-chapters.mkv') f, unnest(f.chapters) c
  GROUP BY f.video[1], f.audio[1]
) TO 'with-cues.mkv'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av-chapters.mkv -f webvtt -i \
  'data:text/vtt;'\
'base64,'\
'V0VCVlRUCgowMDowMDowMC4wMDAgLS0+IDAwOjAwOjAxLjAwMApJbnRybwoKMDA6MDA6MDEuMDAwIC0tPiAwMDowMDowMi4wMDAKQ2hhcHRlciAxCgowMDowMDowMi4wMDAgLS0+IDAwOjAwOjAzLjAwMApDaGFwdGVyIDIKCjAwOjAwOjAzLjAwMCAtLT4gMDA6MDA6MDQuMDAwCkNyZWRpdHMK' \
  -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -metadata:s:1 language=eng -map 1:s:0 -c:2 \
  copy with-cues.mkv
```

The reverse - a `.vtt` file's cues becoming a chapter list - is the same expression with the types swapped: `array_agg(STRUCT(c.text AS title, c.start_t AS start_t, c.end_t AS end_t)::chapter) AS chapters` over `unnest(v.cues) c`.

## 66. Attach a font, and list what a file carries

An attachment is a file riding inside the container - a subtitle font, cover art, a script. Build the list from `attachment` records: filename, MIME type, and the file to read:

```pgsql
COPY (
  SELECT f.video[1], f.audio[1],
         ARRAY[STRUCT('font.ttf' AS filename,
                      'application/x-truetype-font' AS mimetype,
                      'tests/fixtures/font.ttf' AS path)::attachment]
           AS attachments
  FROM input('tests/fixtures/av2.mp4') f
) TO 'fonted.mkv'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -attach tests/fixtures/font.ttf -map 0:v:0 -c:0 copy \
  -map 0:a:0 -c:1 copy -metadata:s:1 language=eng -metadata:s:2 \
  mimetype=application/x-truetype-font -metadata:s:2 filename=font.ttf fonted.mkv
```

Reading is the mirror - attachments are rows like chapters and cues, so a table query lists what a file carries:

```pgsql
SELECT a.index, a.filename, a.mimetype
FROM input('tests/fixtures/attached.mkv') f, unnest(f.attachments) a
```

```
$ ffrwd -f query.sql
 index | filename | mimetype
-------+----------+-----------------------------
 1     | font.ttf | application/x-truetype-font
(1 row)
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

## 68. A function that returns rows

`RETURNS TABLE(...)` makes a function a row source: call it in `FROM`, give it an alias, and read its columns off that alias. It is a view that takes arguments - the thing a view cannot be:

```pgsql
CREATE FUNCTION spoken(file text, lang text) RETURNS TABLE(track audio_stream) AS $$
  SELECT a FROM input(file) f, unnest(f.audio) a WHERE a.tags.language = lang
$$ LANGUAGE sql;

COPY (
  SELECT v.video[1], array_agg(t.track)
  FROM input('tests/fixtures/av2.mp4') v,
       spoken('tests/fixtures/av-2eng.mp4', 'eng') AS t
  GROUP BY v.video[1]
) TO 'out.mkv'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av-2eng.mp4 -i tests/fixtures/av2.mp4 -map 1:v:0 -c:0 copy -map \
  0:a:0 -c:1 copy -metadata:s:1 language=eng -map 0:a:1 -c:2 copy -metadata:s:2 \
  language=eng out.mkv
```

The call contributes its body's rows, so everything that applies to a CTE applies here - cross joins, `WHERE`, grouping, the one-row rule, and a `tags` column in the body tagging the streams it returns (it is not a declared column, so it stays out of `RETURNS TABLE` and off the caller's alias). Calling a table-returning function in the `SELECT` list is rejected: reading a field off the call would read it once per field, minting one input per read.

## 69. Leave a knob unset

An unset variable is `NULL`, and NULL in an option position means the option is not written - ffmpeg's own default applies. One query serves every combination of knobs: here only the height is set, so `scale` gets no `width` and nothing shifts, the positions hold:

```pgsql
COPY (
  SELECT scale(f.video[1], :w, :h), f.audio[1]
  FROM input(:'source') f
) TO :'dest'
```
$ ffrwd compile -f query.sql -v h=480 -v source=film.mp4 -v dest=small.mp4
ffmpeg -i film.mp4 -filter_complex '[0:v:0]scale=height=480[out0]' -map '[out0]' -map 0:a:0 -c:1 copy small.mp4
```
$ ffrwd compile -f query.sql -v h=480 -v source=film.mp4 -v dest=small.mp4
ffmpeg -i film.mp4 -filter_complex '[0:v:0]scale=height=480[out0]' -map '[out0]' -map \
  0:a:0 -c:1 copy small.mp4
```

The same rule at the sink: an unset `crf` writes no `-crf`, the encoder decides, while the options that were set are written as usual:

```sql
COPY (
  SELECT f.video[1], f.audio[1]
  FROM input(:'source') f
) TO :'dest' WITH (video_codec 'libx264', crf :crf, preset :'preset')
```
$ ffrwd compile -f query.sql -v preset=fast -v source=film.mkv -v dest=out.mkv
ffmpeg -i film.mkv -map 0:v:0 -map 0:a:0 -c:1 copy -c:0 libx264 -preset:0 fast out.mkv
```
$ ffrwd compile -f query.sql -v preset=fast -v source=film.mkv -v dest=out.mkv
ffmpeg -i film.mkv -map 0:v:0 -map 0:a:0 -c:1 copy -c:0 libx264 -preset:0 fast out.mkv
```

Tags are the one place absence acts: a NULL field of a `tags` column clears that key, so `:'artist' AS artist` unset clears the artist. "Keep unless told otherwise" is ordinary SQL - `COALESCE(:'title', f.tags.title)` falls back to the file's own title when `:title` is unset:

```pgsql
COPY (
  SELECT f.video[1], f.audio[1],
         STRUCT(:'artist' AS artist,
                COALESCE(:'title', f.tags.title) AS title) AS tags
  FROM input('tests/fixtures/tagged.mp4') f
) TO :'dest'
```
$ ffrwd compile -f query.sql -v dest=out.mkv
ffmpeg -i tests/fixtures/tagged.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -metadata artist= -metadata 'title=Angel One' out.mkv
```
$ ffrwd compile -f query.sql -v dest=out.mkv
ffmpeg -i tests/fixtures/tagged.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -metadata \
  artist= -metadata 'title=Angel One' out.mkv
```

Where a value is required - `input()`'s path, `COPY`'s destination, a stream position, `subtitles`' `filename` - an unset variable is a compile-time rejection that names it: `':source' was not set`.

## 70. Join however many tracks a file has, with concat

`concat` takes any number of segments, so plain SQL never had a way to call it - a filter whose input count isn't in its signature isn't callable at all. `VARIADIC` fixes that: the array it spreads IS the argument list, so `array_agg(t)` over a file's own track rows becomes as many concat inputs as the file actually has, two tracks or ten:

```pgsql
COPY (
  SELECT ffmpeg.concat(VARIADIC array_agg(t))
  FROM input('tests/fixtures/av-chapters.mkv') f, unnest(f.audio) t
) TO 'joined.mka'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av-chapters.mkv -filter_complex \
  '[0:a:0][0:a:1]concat=n=2:v=0:a=1[out0]' -map '[out0]' joined.mka
```

Called without `VARIADIC`, `concat(a, b)` is still `UNKNOWN_FUNCTION` - the hint says to add it. A bare array stays a broadcast (one node per element, same as ever); only the `VARIADIC` spelling means "spread this array into the pads", so the two never collide.

## 71. Mix however many tracks a file has

`amix` already took a written-out list of streams; `VARIADIC` adds a second way to reach it, spreading an already-countable array instead of listing positions by hand. No `GROUP BY` is needed for the common case - `f.audio` is already an array the moment the file is read:

```pgsql
COPY (
  SELECT amix(VARIADIC f.audio)
  FROM input('tests/fixtures/av2.mp4') f
) TO 'mixed.mka'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -filter_complex '[0:a:0][0:a:1]amix=inputs=2[out0]' \
  -map '[out0]' mixed.mka
```

`inputs` still writes itself from the count, so `amix(VARIADIC xs, inputs => 3)` over a two-element array is a rejection naming both numbers, exactly as `amix(a, b, inputs => 3)` already was.
## 72. Write evenly-spaced chapters, however many you want

`generate_series(start, stop[, step])` in `FROM` is a row source that is a count rather than a file - a struct row table with its cells computed instead of written. The alias names both the table and its one column, so `generate_series(1, :count) i` reads back as `i.i`; gather it into a chapter list the same way any row source builds one, and the count becomes a parameter instead of a wall of copy-pasted `STRUCT(...)`s:

```sql
COPY (
  SELECT f.video[1], f.audio[1],
         array_agg(STRUCT('Chapter ' || i.i::text AS title,
                          (i.i - 1) * 10 AS start_t,
                          i.i * 10 AS end_t)::chapter) AS chapters
  FROM input(:'source') f, generate_series(1, :count) i
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v count=4 -v dest=chaptered.mkv
ffmpeg -i film.mkv -f ffmetadata -i \
  'data:text/plain;'\
'base64,'\
'O0ZGTUVUQURBVEExCltDSEFQVEVSXQpUSU1FQkFTRT0xLzEKU1RBUlQ9MApFTkQ9MTAKdGl0bGU9Q2hhcHRlciAxCltDSEFQVEVSXQpUSU1FQkFTRT0xLzEKU1RBUlQ9MTAKRU5EPTIwCnRpdGxlPUNoYXB0ZXIgMgpbQ0hBUFRFUl0KVElNRUJBU0U9MS8xClNUQVJUPTIwCkVORD0zMAp0aXRsZT1DaGFwdGVyIDMKW0NIQVBURVJdClRJTUVCQVNFPTEvMQpTVEFSVD0zMApFTkQ9NDAKdGl0bGU9Q2hhcHRlciA0Cg==' \
  -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -map_chapters 1 chaptered.mkv
```

`i.i` runs 1 to `:count`, so `(i.i - 1) * 10` and `i.i * 10` lay out four back-to-back ten-second chapters here - swap `:count` and every chapter boundary moves with it, no query edit. Bounds and step are checked at compile time (an integer literal or a substituted variable, never a column), so the row count - and therefore how many chapters get written - is known before anything runs.

## 73. Join a series against a file's tracks

A series row is an ordinary compile-time row, so it cross-joins with any other row source exactly like a second `unnest`: each track pairs with each series value, real multiplicity, previewable as a table before it feeds anything real:

```pgsql
SELECT t.tags.language, i.i AS pass
FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) t, generate_series(1, 2) i
```

```
$ ffrwd -f query.sql
 language | pass
----------+------
 eng      | 1
 eng      | 2
 fra      | 1
 fra      | 2
(4 rows)
```

Two audio tracks times two passes is four rows, in the left side's order - the whole point of a compile-time row model: what a fan-out or a gather will do is inspectable before it writes anything. The next two recipes put those rows to work: a series row can key a fan-out `TO`, and it can bound a trim window.

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

## 75. Gather N clips from one file into one

Drop the `TO` expression and the same windows mean something else: the rows have no destination of their own, so they have to be combined. Each row takes its own `-i` of the file with its own seek, and `VARIADIC array_agg` spreads the lot into one `concat` - a contact sheet of a long file, sampled at even intervals:

```pgsql
COPY (
  WITH shots AS (
    SELECT f.video AS frame
    FROM input('tests/fixtures/av.mp4') f, generate_series(1, 3) i
    WHERE f.t >= (i.i - 1) * 2 AND f.t <= (i.i - 1) * 2 + 1
  )
  SELECT ffmpeg.concat(VARIADIC array_agg(shots.frame))
  FROM shots
) TO 'sampled.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -ss 0 -to 1 -i tests/fixtures/av.mp4 -ss 2 -to 3 -i tests/fixtures/av.mp4 -ss 4 \
  -to 5 -i tests/fixtures/av.mp4 -filter_complex \
  '[0:v:0][1:v:0][2:v:0]concat=n=3:v=1:a=0[out0]' -map '[out0]' sampled.mp4
```

A row-bounded window needs one of the two: a `TO` expression to give each row a file, or an aggregate to gather the rows into one. Neither, and the query is a multi-row query into a single destination - the compile error names both ways out.

## 76. Trim a filter call to a computed window

`WHERE <alias>.t` isn't the only way to bound a clip - `ffmpeg.trim`/`ffmpeg.atrim`'s own `starti`/`endi`/`durationi` options take a number of seconds too, and that number may be a compile-time expression over `<alias>.duration`, not just a literal: half the file to its own end, without knowing the file's length up front.

```pgsql
COPY (
  SELECT ffmpeg.trim(a.video[1], starti => a.duration / 4, endi => a.duration)
  FROM input('tests/fixtures/testsrc.mp4') a
) TO 'second_part.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/testsrc.mp4 -filter_complex \
  '[0:v:0]trim=starti=1.0:endi=4.0,setpts=PTS-STARTPTS[out0]' -map '[out0]' \
  second_part.mp4
```

`trim` preserves the source's own timestamps, so the compiler adds `setpts=PTS-STARTPTS` right after it - the query never wrote that filter. See recipe 77 for why that matters.

## 77. Concatenate two windows of the same file

A `trim`/`atrim` call leaves the source's timestamps alone - each clip still carries whatever PTS it had in the original file. Concatenated as-is, the second clip's own offset would show up as a gap. ffrwd resets it for you, once per trim, right before whatever consumes it:

```pgsql
COPY (
  SELECT ffmpeg.trim(a.video[1], starti => 0, endi => 1)
  FROM input('tests/fixtures/testsrc.mp4') a
  UNION ALL
  SELECT ffmpeg.trim(b.video[1], starti => 2, endi => 3)
  FROM input('tests/fixtures/testsrc.mp4') b
) TO 'joined.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/testsrc.mp4 -filter_complex \
  '[0:v:0]trim=starti=0:endi=1[n1];[0:v:0]trim=starti=2:endi=3[n2];'\
'[n1]setpts=PTS-STARTPTS[n1_pts];[n2]setpts=PTS-STARTPTS[n2_pts];'\
'[n1_pts][n2_pts]concat=n=2:v=1:a=0[out0]' -map '[out0]' joined.mp4
```

Two `setpts` nodes, `n1_pts` and `n2_pts` - one per trim, neither written by the query - each rebasing its own clip to start at zero before `concat` joins them end to end. Writing your own `setpts`/`asetpts` right after a trim (or calling `ffrwd.speed`, which expands to one) takes over timing for that stream, and the compiler leaves it alone rather than stacking a second reset on top.

## 78. Lay four windows into a grid

`ffmpeg.xstack` takes however many video streams you give it and lays them into a `grid` you name - each row mints its own window the same way recipe 74's clips do, and `VARIADIC array_agg` gathers them the same way recipe 75's `concat` does. Four one-second windows of one file become a 2x2 contact sheet:

```pgsql
COPY (
  WITH windows AS (
    SELECT f.video AS frame
    FROM input('tests/fixtures/av.mp4') f, generate_series(1, 4) i
    WHERE f.t >= i.i - 1 AND f.t <= i.i
  )
  SELECT ffmpeg.xstack(VARIADIC array_agg(windows.frame), grid => '2x2')
  FROM windows
) TO 'grid.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -ss 0 -to 1 -i tests/fixtures/av.mp4 -ss 1 -to 2 -i tests/fixtures/av.mp4 -ss 2 \
  -to 3 -i tests/fixtures/av.mp4 -ss 3 -to 4 -i tests/fixtures/av.mp4 -filter_complex \
  '[0:v:0][1:v:0][2:v:0][3:v:0]xstack=grid=2x2:inputs=4[out0]' -map '[out0]' grid.mp4
```

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

## 80. Fade in, video or audio, with one name

`fade` is ffmpeg's video filter; `afade` does the same for audio. Write `fade` once and call it on both a video track and an audio track - ffrwd reads each argument's type and resolves the call itself, no need to spell `afade` separately:

```pgsql
COPY (
  SELECT fade(f.video[1], type => 'in', duration => 1),
         fade(f.audio[1], type => 'in', duration => 1)
  FROM input(:'source') f
) TO :'dest'
```

```
$ ffrwd compile -f query.sql -v source=film.mp4 -v dest=faded.mp4
ffmpeg -i film.mp4 -filter_complex \
  '[0:v:0]fade=type=in:duration=1[out0];[0:a:0]afade=type=in:duration=1[out1]' -map \
  '[out0]' -map '[out1]' faded.mp4
```

## 81. Grab N evenly-spaced frames, one file each

A row's own value can name the file it writes. Cross a file against a series, bound each row's window to its share of the duration, and select the series value as a column of the rows - `i.i AS n` - so the fan-out `TO` has something to build a name from. `:count` decides how many frames come out; nothing else moves:

```pgsql
COPY (
  WITH shots AS (
    SELECT v AS frame, i.i AS n
    FROM input(:'source') f, unnest(f.video) v, generate_series(1, :count) i
    WHERE v.index = 1 AND f.t >= f.duration * (i.i - 0.5) / :count
  )
  SELECT shots.frame FROM shots
) TO ('shot' || shots.n::text || '.png') WITH (video_codec 'png', frames 1)
```

```
$ ffrwd compile -f query.sql -v source=tests/fixtures/av2.mp4 -v count=4
ffmpeg -ss 0.5 -i tests/fixtures/av2.mp4 -ss 1.5 -i tests/fixtures/av2.mp4 -ss 2.5 -i \
  tests/fixtures/av2.mp4 -ss 3.5 -i tests/fixtures/av2.mp4 -map 0:v:0 -c:0 png -frames:0 \
  1 shot1.png -map 1:v:0 -c:0 png -frames:0 1 shot2.png -map 2:v:0 -c:0 png -frames:0 1 \
  shot3.png -map 3:v:0 -c:0 png -frames:0 1 shot4.png
```

The window rides on the row, so each frame arrives already seeked, and `n` is an ordinary value column: readable in `WHERE`, in a `GROUP BY`, in a further CTE, or - as here - in the destination.

## 82. Keep a file's tags and change one

`tags` is a map, and `||` merges two of them: whatever the right side names wins, everything else on the left survives. Naming an input's own `tags` on the left is what copies its globals through - without it, ffmpeg's default copying still applies and only the keys written land:

```pgsql
COPY (
  SELECT f.video[1], f.audio[1],
         f.tags || STRUCT('Angel One (restored)' AS title, NULL AS comment) AS tags
  FROM input('tests/fixtures/tagged.mp4') f
) TO 'restored.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/tagged.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy \
  -map_metadata 0 -metadata comment= -metadata 'title=Angel One (restored)' restored.mp4
```

A `NULL` field clears exactly its key. `STRUCT() AS tags` on its own writes no globals at all, which is how a file ships without the tags it was built from.

## 83. Re-encode one track and carry the rest

`* REPLACE(<expr> AS <name>)` keeps the star's whole expansion, in order, with one slot's stream swapped for a computed one - the everyday shape of "change one thing, keep everything else" without writing every column out:

```pgsql
COPY (
  SELECT * REPLACE(scale(a.video[1], 1280, -2) AS video)
  FROM input('tests/fixtures/avs.mkv') a
) TO 'film.mp4' WITH (subtitle_codec 'mov_text')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/avs.mkv -filter_complex \
  '[0:v:0]scale=width=1280:height=-2[out0]' -map '[out0]' -map 0:a:0 -c:1 copy -map \
  0:s:0 -metadata:s:2 language=eng -c:2 mov_text film.mp4
```

`video` names the slot `*` would otherwise fill with `a.video[1]` untouched; audio and the subtitle track pass through exactly as `SELECT *` alone would leave them.

## 84. Drop a kind

`* EXCEPT(<name>, ...)` keeps everything `*` would select except the named kind - here, every subtitle stream, without listing the video and audio columns by hand:

```pgsql
COPY (
  SELECT * EXCEPT(subtitle) FROM input('tests/fixtures/avs.mkv') a
) TO 'film.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/avs.mkv -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy film.mp4
```

No subtitle stream, no `mov_text` transcode: dropping the track drops the reason recipe 2 needed one.

## 85. Key an encode ladder from written rows

`unnest(ARRAY[STRUCT(...), ...])` is an inline row table, its columns named by the STRUCT fields instead of a column list. Each row keys its own `TO` and its own `* REPLACE`, so an encode ladder is one row per rung rather than one query per rung:

```pgsql
COPY (
  SELECT f.* REPLACE(scale(f.video[1], r.w, -2) AS video)
  FROM input('tests/fixtures/av.mp4') f,
       unnest(ARRAY[STRUCT(1920 AS w, '1080p' AS name), STRUCT(1280 AS w, '720p' AS name)]) r
) TO (r.name || '.mp4')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -filter_complex \
  '[0:v:0]split=2[src_f_v_0_split0][src_f_v_0_split1];'\
'[src_f_v_0_split0]scale=width=1920:height=-2[out0];'\
'[src_f_v_0_split1]scale=width=1280:height=-2[out2]' -map '[out0]' -map 0:a:0 -c:1 copy \
  1080p.mp4 -map '[out2]' -map 0:a:0 -c:1 copy 720p.mp4
```

Add a rung by adding a row - nothing else about the query changes. A row's field is a compile-time value like any other: `r.w` reads as a number wherever one is wanted, the same as a track row's own metadata columns do.

## 86. Gather clips into one file without the CTE

`ARRAY(<select>)` is the converse of `unnest`: it gathers a countable subquery's rows into an array, in expression position, without a `WITH` and a `GROUP BY`-less `array_agg` to spell it. Recipe 75's contact sheet, written as one expression:

```pgsql
COPY (
  SELECT ffmpeg.concat(VARIADIC ARRAY(
    SELECT f.video FROM input('tests/fixtures/av.mp4') f, generate_series(1, 3) i
    WHERE f.t >= (i.i - 1) * 2 AND f.t <= (i.i - 1) * 2 + 1
  ))
) TO 'sampled.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -ss 0 -to 1 -i tests/fixtures/av.mp4 -ss 2 -to 3 -i tests/fixtures/av.mp4 -ss 4 \
  -to 5 -i tests/fixtures/av.mp4 -filter_complex \
  '[0:v:0][1:v:0][2:v:0]concat=n=3:v=1:a=0[out0]' -map '[out0]' sampled.mp4
```

Byte for byte recipe 75's command: `ARRAY(...)` reads its subquery's own FROM as this branch's row source, the same one `WITH shots AS (...) ... FROM shots` gives `array_agg` by hand. A multi-column subquery needs `SELECT AS STRUCT` to gather a struct array instead - the same shape `array_agg(STRUCT(...)::chapter)` builds by hand, for a `chapters`/`attachments` column that has nowhere to write the cast.

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
  ../sidecar/modules/target/wasm32-wasip2/release/invert.wasm -f nut pipe:1 | ffmpeg -f \
  nut -i pipe:0 -map 0:v:0 -c:0 libx264 -crf:0 20 inverted.mp4
```

A module cannot be a link in one ffmpeg's filter graph, so the query compiles to three processes joined by pipes rather than one command: an ffmpeg that decodes, the sidecar hosting the module, and an ffmpeg that encodes what comes back. The frames travel as NUT, and the pixel format on both seams is the one the module and the wire agree on - `rgba` here, because that is what `invert` accepts. `ffrwd run` executes the whole pipeline itself; the printed form is for reading and pasting.

## 88. Blur the faces a module found

A module can hand the next one more than frames. `RETURNS STRUCT(v video_stream, faces STRUCT(...)[])` says this module reads something off each frame and returns it beside the picture; a module that consumes those rows takes them right after its own stream. Writing one call inside the other passes both at once:

```pgsql
CREATE FUNCTION detect_faces(v video_stream)
RETURNS STRUCT(v video_stream, faces STRUCT(x number, y number, w number, h number)[])
  AS '../sidecar/modules/target/wasm32-wasip2/release/facebox.wasm', 'facebox'
  LANGUAGE wasm;

CREATE FUNCTION blur_boxes(v video_stream,
                           faces STRUCT(x number, y number, w number, h number)[])
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/blur_boxes.wasm', 'blur-boxes'
  LANGUAGE wasm;

COPY (
  SELECT blur_boxes(detect_faces(s.video[1])), s.audio
  FROM input('tests/fixtures/av.mp4') s
) TO 'blurred.mp4' WITH (video_codec 'libx264', crf 20)
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 yuv420p -f nut \
  pipe:1 | ffrwd-wasm -f nut -i pipe:0 -m \
  facebox=../sidecar/modules/target/wasm32-wasip2/release/facebox.wasm -m \
  blur_boxes=../sidecar/modules/target/wasm32-wasip2/release/blur_boxes.wasm \
  -filter_complex '[0:v]facebox[n1];[n1]blur_boxes[out0]' -map '[out0]' -f nut pipe:1 | \
  ffmpeg -i tests/fixtures/av.mp4 -f nut -i pipe:0 -map 1:v:0 -map 0:a:0 -c:1 copy -c:0 \
  libx264 -crf:0 20 blurred.mp4
```

The annotation record is checked twice before anything runs: against the row schema `facebox` publishes, so a misspelled field or a wrong type is a rejection at the declaration, and against `blur_boxes`'s own parameter, so the two ends of the composition have to agree. The names are the writer's - nothing carries them at run time - which is what makes this a type and not a protocol.

The two modules are adjacent, so ONE sidecar hosts both. `-m <name>=<path>` binds each module to a name, and the `-filter_complex` string wires those names together exactly as it wires ffmpeg's filters. The frames - and the rows riding on them - pass from one module to the next in memory, so nothing between them is a pipe, a NUT hop or a flag; only the two ffmpeg seams are.

## 89. Compute a tag with a wasm module

`RETURNS text` (or `number`, or `boolean`) makes a wasm function a VALUE, not a filter: it takes no stream, and the compiler runs it once per call, at compile time, the way it runs ffprobe. The result folds into the query as a literal - here, into the tags map beside a probed one:

```pgsql
CREATE FUNCTION brand(title text, suffix text) RETURNS text
  AS '../sidecar/modules/target/wasm32-wasip2/release/brand.wasm', 'append-brand'
  LANGUAGE wasm;

COPY (
  SELECT f.video[1], f.audio[1],
         f.tags || STRUCT(brand(f.tags.title, ' (restored)') AS title) AS tags
  FROM input('tests/fixtures/tagged.mp4') f
) TO 'restored.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/tagged.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy \
  -map_metadata 0 -metadata 'title=Angel One (restored)' restored.mp4
```

`tagged.mp4`'s own title, "Angel One", is what `f.tags.title` reads once probing runs; `brand()` appends the suffix and hands back "Angel One (restored)", already spelled out in `-metadata` by the time ffmpeg's command exists. Nothing about `brand` survives into the argv - no module path, no sidecar, no plan - because there is no stream for it to touch and nothing left to run at command time.

A parameter or a RETURNS the module's own schema does not hold - a struct, an array, a mismatched type - is a rejection at the declaration; an argument that does not resolve to a compile-time value (an undecoded frame, say) is a rejection at the call.

## 90. Crop to a rectangle

`crop` takes its rectangle by name - `out_w`/`out_h` for the size, `x`/`y` for the top-left corner it starts at - so nothing here depends on argument order. The audio rides along untouched:

```pgsql
COPY (
  SELECT crop(f.video[1], out_w => 160, out_h => 120, x => 80, y => 60), f.audio[1]
  FROM input('tests/fixtures/av2.mp4') f
) TO 'cropped.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -filter_complex \
  '[0:v:0]crop=out_w=160:out_h=120:x=80:y=60[out0]' -map '[out0]' -map 0:a:0 -c:1 copy \
  -metadata:s:1 language=eng cropped.mp4
```

Every option is an ffmpeg expression, so `out_w => 'iw/2'` and `x => '(iw-ow)/2'` crop a centered half-width window out of whatever the input turns out to be.

## 91. Keep the blur boxes steady across cuts

Three modules, still one sidecar. `shots` watches for hard cuts and hands each frame a `{"shot": n}` row; `facebox` remembers every box it finds for half a second and blurs the union of what it remembers - steadier boxes that survive frames the detector misses - and empties that memory the moment the shot index changes, so no box lingers across a cut:

```pgsql
CREATE FUNCTION shots(v video_stream)
RETURNS STRUCT(v video_stream, cuts STRUCT(shot number)[])
  AS '../sidecar/modules/target/wasm32-wasip2/release/shots.wasm', 'shots'
  LANGUAGE wasm;

CREATE FUNCTION detect_faces(v video_stream,
                             cuts STRUCT(shot number)[] DEFAULT NULL)
RETURNS STRUCT(v video_stream, faces STRUCT(x number, y number, w number, h number)[])
  AS '../sidecar/modules/target/wasm32-wasip2/release/facebox.wasm', 'facebox'
  LANGUAGE wasm;

CREATE FUNCTION blur_boxes(v video_stream,
                           faces STRUCT(x number, y number, w number, h number)[])
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/blur_boxes.wasm', 'blur-boxes'
  LANGUAGE wasm;

COPY (
  SELECT blur_boxes(detect_faces(shots(s.video[1]))), s.audio
  FROM input('tests/fixtures/av.mp4') s
) TO 'blurred.mp4' WITH (video_codec 'libx264', crf 20)
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 yuv420p -f nut \
  pipe:1 | ffrwd-wasm -f nut -i pipe:0 -m \
  shots=../sidecar/modules/target/wasm32-wasip2/release/shots.wasm -m \
  facebox=../sidecar/modules/target/wasm32-wasip2/release/facebox.wasm -m \
  blur_boxes=../sidecar/modules/target/wasm32-wasip2/release/blur_boxes.wasm \
  -filter_complex '[0:v]shots[n1];[n1]facebox[n2];[n2]blur_boxes[out0]' -map '[out0]' -f \
  nut pipe:1 | ffmpeg -i tests/fixtures/av.mp4 -f nut -i pipe:0 -map 1:v:0 -map 0:a:0 \
  -c:1 copy -c:0 libx264 -crf:0 20 blurred.mp4
```

The `DEFAULT NULL` on `cuts` makes the column optional: delete the `shots(...)` call and the same declaration still compiles, with no rows wired in - `detect_faces` then simply never clears what it remembers. Only a module that reads rows at its own option can default the column; one that exists to consume them, like `blur_boxes`, is refused a DEFAULT at the declaration.

## 92. Subtitle a film in a language it does not speak

`transcribe` listens in one language and can hand back another: `language` is what the dialogue is in, `language_to` (optional) what the rows come out in. Declaring the annotation column `cue[]` types it as cues, and selecting `.words` off the call is a subtitle track - the same rule that turns a compile-time cue array into one. The track's language is computed, not spelled: the first of `language_to`, `language` that is set at the call, so the output always says what it carries:

```pgsql
CREATE FUNCTION transcribe(a audio_stream,
                           language text,
                           language_to text DEFAULT NULL)
RETURNS STRUCT(a audio_stream, words cue[])
  AS '../sidecar/modules/target/wasm32-wasip2/release/transcribe.wasm', 'transcribe'
  LANGUAGE wasm;

COPY (
  SELECT s.video[1], s.audio[1],
         transcribe(s.audio[1], 'es', 'en').words
  FROM input('tests/fixtures/av.mp4') s
) TO 'subbed.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -map 0:a:0 -ar:0 16000 -ac:0 1 -c:0 pcm_f32le -f nut \
  pipe:1 | ffrwd-wasm -f nut -i pipe:0 -m \
  ../sidecar/modules/target/wasm32-wasip2/release/transcribe.wasm -params \
  '{"language": "es", "language_to": "en"}' -f webvtt pipe:1 | ffmpeg -i \
  tests/fixtures/av.mp4 -f webvtt -i pipe:0 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy \
  -map 1:s:0 -metadata:s:2 language=eng -c:2 mov_text subbed.mp4
```

The module asked for 16kHz mono, so the feeding ffmpeg conforms the stream before the pcm edge - nothing in the query says so. The sidecar writes the cues as a WebVTT document on its own output, and the far ffmpeg encodes it per container: `mov_text` here because the destination is mp4, copied through untouched for mkv. `-metadata:s:2 language=eng` is the computed tag - `language_to` first, `language` when no conversion is asked for. Swap the destination for `'words.ndjson'` and drop the streams to get the rows themselves instead of a track.

## 93. Depth of field, from the scene's own geometry

`depth` reads a frame and hands back its depth as a grayscale picture, a model
doing the looking; `blur_mask` takes a frame beside any grayscale mask and
blurs each pixel by the mask's brightness there. Composing them is fake bokeh
driven by real geometry - `invert` because depth models paint near bright, and
it is the far field that should melt:

```pgsql
CREATE FUNCTION depth(v video_stream) RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/depth.wasm', 'depth' LANGUAGE wasm;

CREATE FUNCTION blur_mask(v video_stream, mask video_stream,
                          max_radius number DEFAULT 16, invert boolean DEFAULT FALSE)
RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/blur_mask.wasm', 'blur_mask' LANGUAGE wasm;

COPY (
  SELECT blur_mask(f.video[1], depth(f.video[1]), 24, TRUE)
  FROM input('tests/fixtures/testsrc.mp4') f
) TO 'bokeh.mp4' WITH (video_codec 'libx264', crf 20)
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/testsrc.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 yuv420p -f nut \
  pipe:1 | ffrwd-wasm -f nut -i pipe:0 -nn \
  depth=../sidecar/modules/target/wasm32-wasip2/release/depth.onnx -m \
  depth=../sidecar/modules/target/wasm32-wasip2/release/depth.wasm -m \
  blur_mask=../sidecar/modules/target/wasm32-wasip2/release/blur_mask.wasm \
  -filter_complex '[0:v]depth[n1];[0:v][n1]blur_mask=max_radius=24:invert=1[out0]' -map \
  '[out0]' -f nut pipe:1 | ffmpeg -f nut -i pipe:0 -map 0:v:0 -c:0 libx264 -crf:0 20 \
  bokeh.mp4
```

The `-nn` binding is the model file beside the module, found by name; the
module runs it through the host's inference runtime, on the GPU when the
machine has one, and neither fact appears in the query. `mask` is any
grayscale stream - a segmentation matte or a hand-drawn gradient work the
same way.

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
  nut pipe:1 | ffmpeg -i tests/fixtures/av.mp4 -f nut -i pipe:0 -map 1:v:0 -map 0:a:0 \
  -c:1 copy -c:0 libx264 -crf:0 20 blurred.mp4
```

`rowfilter` is the WHERE, compiled: not a module, but a node the host
provides, with the predicate carried in the graph. Swap `'person'` for any
COCO class, or write `AND o.score >= 0.5` to trim low-confidence matches -
the predicate grammar is comparisons, AND, OR, NOT and parentheses over the
row's own fields. Both `segment(...)` spellings are one call: the map and
the rows leave the same node, and the rows ride the map's frames through
the filter.

## 95. Make a clip out of nothing

An `input()` takes named options after the path - ffmpeg's own per-input
flags, written in the order ffmpeg gets them, immediately before that
input's `-i`. `format` is the one that changes what the path means: name a
demuxer and the path is that demuxer's to read, not the filesystem's. With
`lavfi` it is a filter graph, so this compiles and runs anywhere ffmpeg
exists, with no file on disk:

```pgsql
COPY (
  SELECT gblur(a.video[1], sigma => 4)
  FROM input('testsrc2=size=640x360:rate=25:duration=2', format => 'lavfi') a
) TO 'synthetic.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -f lavfi -i testsrc2=size=640x360:rate=25:duration=2 -filter_complex \
  '[0:v:0]gblur=sigma=4[out0]' -map '[out0]' synthetic.mp4
```

`format` shapes what the demuxer reads, so it reaches the probe as well as
the decode - ffprobe is run with it too - and the synthetic source is as
readable as a file: `a.*` expands over it, `a.video` has a length, and the
durations come back. Two aliases naming one path with different options are
two `-i` entries and two probes; with the same options they fold onto one.

The same mechanism reaches a capture device - `input('video=Logitech BRIO',
format => 'dshow', framerate => 30, video_size => '960x540')` - and a
network stream, where `rtsp_transport => 'tcp'` and friends are protocol
options rather than device ones. A live source simply has no duration for
the probe to report.

## 96. Take the top row: ORDER BY ... LIMIT

`LIMIT` is legal exactly where `ORDER BY` is - over a compile-time row
table - and it narrows the resolved row count the same way `WHERE` does.
So "the widest video track" is a sort and a `LIMIT 1`, no self-join, no
aggregate; the same spelling picks the first track of a language:

```pgsql
COPY (
  WITH vid AS (
    SELECT t AS track
    FROM input('tests/fixtures/av2.mp4') f, unnest(f.video) t
    ORDER BY t.width DESC LIMIT 1
  ),
  aud AS (
    SELECT a AS track
    FROM input('tests/fixtures/av2.mp4') g, unnest(g.audio) a
    WHERE a.tags.language = 'eng'
    ORDER BY a.index LIMIT 1
  )
  SELECT vid.track, aud.track FROM vid, aud
) TO 'picked.mp4'
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av2.mp4 -map 0:v:0 -c:0 copy -map 0:a:0 -c:1 copy -metadata:s:1 \
  language=eng picked.mp4
```

Each CTE resolves to ONE row before the one-row rule looks, so a single
destination takes the pair without an `array_agg`. The count must stay
positive and knowable: `LIMIT 0` is rejected (a query that selects
nothing is a mistake worth naming), and the count is an integer literal
after `-v` substitution - `LIMIT :n` works, a column reference does not.
`OFFSET` rides along with the same rules, and skipping every row is the
same rejection `LIMIT 0` gets.

## 97. Key a ladder from list variables

`-v` takes a list: subscript the reference (`:widths[2]`, 1-based) and
the value splits on commas at substitution time; unsubscripted, it stays
the one raw text it always was. A literal subscript resolves before the
query even parses, and a row-column subscript picks per row - so a series
plus two lists is an encode ladder whose rungs live on the command line:

```pgsql
COPY (
  SELECT scale(f.video[1], :widths[i.i], -2) AS v, f.audio
  FROM input('tests/fixtures/av.mp4') f, generate_series(1, 3) i
) TO (:'names'[i.i] || '.mp4')
  WITH (video_codec 'libx264', audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql -v widths=1920,1280,640 -v names=1080p,720p,360p
ffmpeg -i tests/fixtures/av.mp4 -filter_complex \
  '[0:v:0]split=3[src_f_v_0_split0][src_f_v_0_split1][src_f_v_0_split2];'\
'[src_f_v_0_split0]scale=width=1920:height=-2[out0];'\
'[src_f_v_0_split1]scale=width=1280:height=-2[out2];'\
'[src_f_v_0_split2]scale=width=640:height=-2[out4]' -map '[out0]' -map 0:a:0 -c:0 \
  libx264 -c:1 aac 1080p.mp4 -map '[out2]' -map 0:a:0 -c:0 libx264 -c:1 aac 720p.mp4 \
  -map '[out4]' -map 0:a:0 -c:0 libx264 -c:1 aac 360p.mp4
```

`:widths[i.i]` substitutes to `ARRAY[1920,1280,640][i.i]` - a raw
reference splices its elements raw, `:'names'[i.i]` makes them string
literals - and the element is read per row during lowering, the same way
`r.w` reads off a written row table. A subscript past the end is a
rejection naming the list's length, whether it is written as a literal
or reached by a row. The whole variable unset stays NULL-is-absence,
subscripted or not.

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

## 99. Watch the frames go by

A sink needs no rows and no network - consuming the stream is enough.
`frame_stats` reads each frame, writes one stats line to stderr, and
emits nothing:

```pgsql
CREATE FUNCTION frame_stats(v video_stream) RETURNS sink
  AS '../sidecar/modules/target/wasm32-wasip2/release/frame_stats.wasm', 'frame_stats'
  LANGUAGE wasm;

COPY (
  SELECT s.video[1]
  FROM input('tests/fixtures/av.mp4') s
) TO frame_stats()
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -map 0:v:0 -c:0 rawvideo -pix_fmt:0 rgba -f nut pipe:1 | \
  ffrwd-wasm -f nut -i pipe:0 -m \
  ../sidecar/modules/target/wasm32-wasip2/release/frame_stats.wasm -f null -
```

One module, so the short `-m <path>` spelling holds, and the COPY names
no file anywhere: the stderr lines are the whole product.

## 100. Read the encoder's output, packet by packet

A packet sink is a sink module that consumes ENCODED packets - the
compressed bytes, pts, dts and keyframe flags - instead of decoded
frames. The compiler reads that off the module's describe and places
the sink after an encoder: the feeding ffmpeg encodes onto the pipe,
and the sidecar hands the packets through untouched. `packet_stats`
counts each group of pictures and emits its rows as NDJSON on stdout:

```pgsql
CREATE FUNCTION packet_stats(v video_stream) RETURNS sink
  AS '../sidecar/modules/target/wasm32-wasip2/release/packet_stats.wasm', 'packet_stats'
  LANGUAGE wasm;

COPY (
  SELECT s.video[1]
  FROM input('tests/fixtures/av.mp4') s
) TO packet_stats()
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -map 0:v:0 -c:0 libx264 -pix_fmt:0 yuv420p -f nut pipe:1 \
  | ffrwd-wasm -f nut -i pipe:0 -m \
  ../sidecar/modules/target/wasm32-wasip2/release/packet_stats.wasm -f ndjson pipe:1
```

No codec was named, so the compiler takes the module's preference -
`packet_stats` accepts every codec, and the default is h264 by
libx264. The encoder is shaped by the same WITH options a file sink
takes, the video ones: `video_codec`, `crf`, `preset` and kin. A codec
the module's describe does not accept is refused at compile time,
naming the ones it does:

```pgsql
CREATE FUNCTION packet_stats(v video_stream) RETURNS sink
  AS '../sidecar/modules/target/wasm32-wasip2/release/packet_stats.wasm', 'packet_stats'
  LANGUAGE wasm;

COPY (
  SELECT s.video[1]
  FROM input('tests/fixtures/av.mp4') s
) TO packet_stats() WITH (video_codec 'libx265', crf 30, preset 'fast')
```

```
$ ffrwd compile -f query.sql
ffmpeg -i tests/fixtures/av.mp4 -map 0:v:0 -c:0 libx265 -pix_fmt:0 yuv420p -crf:0 30 \
  -preset:0 fast -f nut pipe:1 | ffrwd-wasm -f nut -i pipe:0 -m \
  ../sidecar/modules/target/wasm32-wasip2/release/packet_stats.wasm -f ndjson pipe:1
```

The encode happens on the feeder's way out, after its filtergraph:
`scale` the stream in the SELECT and the resize runs on decoded
frames, with only the sink's own edge carrying the encode. That holds
whatever feeds the sink - a module's output gets an encoding stage of
its own on the way in. A sink declaring an audio parameter takes the
audio options too. The rows ride the sidecar's stdout, one JSON
object per line - redirect them to keep them.

An arrayed parameter reads every stream of its kind the SELECT
carries: gather a ladder and one instance reads every rung, each pad
behind its own encoder, and a `WITH` value read once per row shapes
each rung separately:

```pgsql
CREATE FUNCTION packet_tally(v video_stream[]) RETURNS sink
  AS '../sidecar/modules/target/wasm32-wasip2/release/packet_tally.wasm', 'packet_tally'
  LANGUAGE wasm;

COPY (
  SELECT array_agg(scale(s.video[1], ARRAY[640, 320][i.i], -2))
  FROM input('tests/fixtures/av.mp4') s, generate_series(1, 2) i
) TO packet_tally() WITH (video_bitrate ARRAY['1200k', '400k'][i.i])
```

```
$ ffrwd compile -f query.sql
# named pipes: sidecar0 reads ffmpeg0, ffmpeg0; ffmpeg0 feeds sidecar0, sidecar0
1. ffmpeg: ffmpeg -i tests/fixtures/av.mp4 -filter_complex \
  '[0:v:0]split=2[src_s_v_0_split0][src_s_v_0_split1];'\
'[src_s_v_0_split0]scale=width=640:height=-2[out0];'\
'[src_s_v_0_split1]scale=width=320:height=-2[out1]' -map '[out0]' -c:0 libx264 \
  -pix_fmt:0 yuv420p -b:0 1200k -f nut '<named pipe ffmpeg0-sidecar0 n1 write>' -map \
  '[out1]' -c:0 libx264 -pix_fmt:0 yuv420p -b:0 400k -f nut \
  '<named pipe ffmpeg0-sidecar0 n2 write>'
2. sidecar: ffrwd-wasm -f nut -i '<named pipe ffmpeg0-sidecar0 n1 read>' -f nut -i \
  '<named pipe ffmpeg0-sidecar0 n2 read>' -m \
  ../sidecar/modules/target/wasm32-wasip2/release/packet_tally.wasm -f ndjson pipe:1
# this listing is not a shell command -- run the plan with `ffrwd run`
```

One decode, one encoder per rung, one instance reading every pad.

## 101. Read a live source once, however many ways the query uses it

A file can be opened twice, so a query that sends one stream through a
module and merges the result with the original picture lets each
process decode the file for itself. A socket cannot: an `srt://` or
`udp://` URL, or an input whose `format =>` names a capture device,
binds once and refuses the second open. Such an input is read by
exactly ONE process, whatever the shape of the graph. Everything the
consumers need before the split - here the `split` itself - moves into
that reader, and each consumer gets a pipe of its own:

```pgsql
CREATE FUNCTION invert(v video_stream) RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/invert.wasm', 'invert'
  LANGUAGE wasm;

COPY (
  SELECT ffmpeg.hstack(a.video[1], invert(a.video[1]))
  FROM input('testsrc2=size=640x360:rate=30:duration=5',
             format => 'lavfi', realtime => true) a
) TO 'live.mp4' WITH (video_codec 'libx264', crf 22)
```

```
$ ffrwd compile -f query.sql
# named pipes: ffmpeg0 reads ffmpeg1, sidecar0; ffmpeg1 feeds sidecar0, ffmpeg0
1. ffmpeg: ffmpeg -f nut -i '<named pipe ffmpeg1-ffmpeg0 src_a_v_0_split:1 read>' -f nut \
  -i '<named pipe sidecar0-ffmpeg0 n1 read>' -filter_complex \
  '[0:v:0][1:v:0]hstack=inputs=2[out0]' -map '[out0]' -c:0 libx264 -crf:0 22 live.mp4
2. ffmpeg: ffmpeg -f lavfi -re -i testsrc2=size=640x360:rate=30:duration=5 \
  -filter_complex '[0:v:0]split=2[out0][out1]' -map '[out0]' -c:0 rawvideo -pix_fmt:0 \
  rgba -f nut '<named pipe ffmpeg1-sidecar0 src_a_v_0_split:0 write>' -map '[out1]' -c:0 \
  rawvideo -pix_fmt:0 yuv420p -f nut \
  '<named pipe ffmpeg1-ffmpeg0 src_a_v_0_split:1 write>'
3. sidecar: ffrwd-wasm -f nut -i pipe:0 -m \
  ../sidecar/modules/target/wasm32-wasip2/release/invert.wasm -f nut pipe:1
# this listing is not a shell command -- run the plan with `ffrwd run`
```

`testsrc2=...` appears in exactly one of the three, and process 2 has
two outputs where a file-backed query would have had two processes.
`realtime => true` paces the read at the source's own frame rate; a
real camera or listener paces itself and needs no flag, and `format
=>` alone is enough to make an input one-open. A stream nothing on the
far side filters - an audio track mapped straight through - crosses
its pipe as `-c copy`, so a passthrough stays a passthrough.

## 102. Size the buffer between a live source's two paths

The reader hands one frame to each of its pipes at once, and `hstack`
takes one from each at once - but the module's frame goes the long way
round, through a second process. So the direct pipe holds frames while
the module's catches up, and the compiler counts how many: one process
between the paths, holding the frame it is working on plus whatever
its modules declare they read ahead. That count is the edge's BOUND,
and doubling it is what the buffer is sized to.

Where those bytes fit comfortably in a pipe's own buffer - recipe 101,
at 640x360 - the pipe is simply made that big and the command line
shows nothing. Raise the frame size and the depth moves into ffmpeg's
own fifo muxer instead, which queues packets rather than kernel
memory:

```pgsql
CREATE FUNCTION invert(v video_stream) RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/invert.wasm', 'invert'
  LANGUAGE wasm;

COPY (
  SELECT ffmpeg.hstack(a.video[1], invert(a.video[1]))
  FROM input('testsrc2=size=1920x1080:rate=30:duration=5',
             format => 'lavfi', realtime => true) a
) TO 'live.mp4' WITH (video_codec 'libx264', crf 22)
```

```
$ ffrwd compile -f query.sql
# named pipes: ffmpeg0 reads ffmpeg1, sidecar0; ffmpeg1 feeds sidecar0, ffmpeg0
1. ffmpeg: ffmpeg -f nut -i '<named pipe ffmpeg1-ffmpeg0 src_a_v_0_split:1 read>' -f nut \
  -i '<named pipe sidecar0-ffmpeg0 n1 read>' -filter_complex \
  '[0:v:0][1:v:0]hstack=inputs=2[out0]' -map '[out0]' -c:0 libx264 -crf:0 22 live.mp4
2. ffmpeg: ffmpeg -f lavfi -re -i testsrc2=size=1920x1080:rate=30:duration=5 \
  -filter_complex '[0:v:0]split=2[out0][out1]' -map '[out0]' -c:0 rawvideo -pix_fmt:0 \
  rgba -f nut '<named pipe ffmpeg1-sidecar0 src_a_v_0_split:0 write>' -map '[out1]' -c:0 \
  rawvideo -pix_fmt:0 yuv420p -fifo_format nut -queue_size 2 -f fifo \
  '<named pipe ffmpeg1-ffmpeg0 src_a_v_0_split:1 write>'
3. sidecar: ffrwd-wasm -f nut -i pipe:0 -m \
  ../sidecar/modules/target/wasm32-wasip2/release/invert.wasm -f nut pipe:1
# this listing is not a shell command -- run the plan with `ffrwd run`
```

`-queue_size 2` is the bound of 1 frame doubled. A module declaring a
`window` raises it: nine frames of window is a bound of nine and a
queue of eighteen. `ffrwd explain` prints the number and the road each
edge took, under the plan's `edges`.

Both roads block when they fill, and neither drops a frame. A run
whose paths drift further apart than the bound counted stops with
`BUFFER_OVERFLOW` naming the edge and the depth it was given - see
[docs/errors.md](errors.md). And a query whose two paths have a stage
between them that hands on a different number of frames than it reads
- `fps`, `select`, a module that does not declare one frame out per
frame in - has no such count at all, and is refused at compile time
with `UNBOUNDED_LIVE_INPUT`.

## 103. Give each rung of the ladder its own encode

Recipe 97 varies the picture and the filename per rung and leaves every
rung encoded the same. A `WITH` option takes a subscripted list variable
too, so the encoder settings ride the command line with the rest of the
ladder - one bitrate per rung, read off the pinned row the same way the
`TO` expression is:

```pgsql
COPY (
  SELECT scale(f.video[1], :widths[i.i], -2) AS v, f.audio
  FROM input('tests/fixtures/av.mp4') f, generate_series(1, 3) i
) TO (:'names'[i.i] || '.mp4')
  WITH (video_codec 'libx264', video_bitrate :'rates'[i.i],
        audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql -v widths=1920,1280,640 -v names=1080p,720p,360p -v rates=6000k,3000k,1000k
ffmpeg -i tests/fixtures/av.mp4 -filter_complex \
  '[0:v:0]split=3[src_f_v_0_split0][src_f_v_0_split1][src_f_v_0_split2];'\
'[src_f_v_0_split0]scale=width=1920:height=-2[out0];'\
'[src_f_v_0_split1]scale=width=1280:height=-2[out2];'\
'[src_f_v_0_split2]scale=width=640:height=-2[out4]' -map '[out0]' -map 0:a:0 -c:0 \
  libx264 -b:0 6000k -c:1 aac 1080p.mp4 -map '[out2]' -map 0:a:0 -c:0 libx264 -b:0 3000k \
  -c:1 aac 720p.mp4 -map '[out4]' -map 0:a:0 -c:0 libx264 -b:0 1000k -c:1 aac 360p.mp4
```

An option value is settled before ffmpeg runs, so what may stand there
is a literal or a subscripted list variable and nothing else: a column
off the media itself is a rejection naming the option. A subscript past
the end of the list is the same rejection it is anywhere else, naming
the list's length - so a `rates` list shorter than the series says so
rather than writing a file at the wrong bitrate.

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
  -master_pl_name master.m3u8 -hls_segment_filename out/v%v/segment_%d.ts \
  out/v%v/index.m3u8
```

`format 'dash'` is the same plan wearing different words: `seg_duration`
for `hls_time`, adaptation sets for the variant map, and the `.mpd` as
the written name.

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

## 106. The widest rung of an ABR ladder

Same relation, no rung named up front: `ORDER BY r.bandwidth DESC LIMIT
1` picks the highest-bitrate rendition without knowing its resolution
in advance - the winning row still carries its own `video`/`audio`
arrays, same as any other:

```pgsql
COPY (
  SELECT r.video[1], r.audio[1]
  FROM input(:'ladder') r
  ORDER BY r.bandwidth DESC LIMIT 1
) TO :'dest' WITH (video_codec 'libx264', crf 20, audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql -v ladder=tests/fixtures/ladder/master.m3u8 -v dest=widest.mp4
ffmpeg -i tests/fixtures/ladder/master.m3u8 -map 0:v:0 -map 0:a:0 -c:0 libx264 -crf:0 20 \
  -c:1 aac widest.mp4
```

Reach for this when the ladder's own top rung is whatever the source
considers best, and the job just wants that one, whichever resolution
it turns out to be.

## 107. Re-encode an ABR ladder through to another ABR ladder

No `WHERE`, no `LIMIT`: every rendition row survives the `SELECT`, and
`format 'hls'` on the destination turns the multi-row relation back
into a manifest - the same acceptance rule recipe 104 uses for rows
built by hand:

```pgsql
COPY (
  SELECT r.video[1], r.audio[1]
  FROM input(:'ladder') r
) TO :'dest' WITH (
  format 'hls', hls_time 2, hls_playlist_type 'vod', hls_segment_type 'fmp4',
  video_codec 'libx264', audio_codec 'aac'
)
```

```
$ ffrwd compile -f query.sql -v ladder=tests/fixtures/ladder/master.m3u8 -v dest=out/master.m3u8
ffmpeg -i tests/fixtures/ladder/master.m3u8 -map 0:v:0 -map 0:v:1 -map 0:a:0 -map 0:a:1 \
  -f hls -hls_time 2 -hls_playlist_type vod -hls_segment_type fmp4 -c:0 libx264 -c:1 \
  libx264 -c:2 aac -c:3 aac -g:0 30 -g:1 30 -keyint_min:0 30 -keyint_min:1 30 \
  -sc_threshold:0 0 -sc_threshold:1 0 -var_stream_map \
  'v:0,a:0,name:1080p v:1,a:1,name:720p' -master_pl_name master.m3u8 \
  -hls_segment_filename out/v%v/segment_%d.m4s -hls_fmp4_init_filename init.mp4 \
  out/v%v/index.m3u8
```

Reach for this to repackage or re-encode someone else's ladder into
your own, rung for rung, without re-deriving the variant map by hand.

## 108. Stream a file as if it were live

Nothing paces a plain file input: ffmpeg reads it as fast as it can, so
pointing one at a live-shaped destination finishes in a fraction of the
file's own duration instead of running for it. `realtime => true` adds
`-re` ahead of that input's `-i`, reading it at its own frame rate
instead - the same flag a real live source would need none of:

```pgsql
COPY (SELECT f.video[1], f.audio[1] FROM input(:'source', realtime => true) f)
TO :'dest' WITH (format 'flv', video_codec 'libx264', audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql -v source=film.mkv -v dest=rtmp://live.example.com/app/streamkey
ffmpeg -re -i film.mkv -map 0:v:0 -map 0:a:0 -f flv -c:0 libx264 -c:1 aac \
  rtmp://live.example.com/app/streamkey
```

Reach for this to publish a VOD file to a live-shaped destination -
an ad break spliced into a stream, or a rung of an encode ladder sent
to a relay expecting a live push. `realtime` refuses on an input
already named by a socket (`srt://`, `udp://`, `rtmp://`, ...): it is
already paced by whatever is sending it, and pacing it a second time
is refused with a hint to drop the option - see `INPUT_OPTION_TYPE` in
[docs/errors.md](errors.md).

## 109. A ladder into one file with every rung as its own track

`input(:'ladder')` is one row per rendition, same as recipe 105's.
`array_agg(r.video[1])` collapses whichever rows survive down to a
single row whose one column is an N-element video array - and a
container that holds several video tracks (`mkv`) writes an N-element
array as N tracks in one file rather than N files:

```pgsql
COPY (
  SELECT array_agg(r.video[1])
  FROM input(:'ladder') r
  WHERE r.height >= 480
) TO :'dest' WITH (video_codec 'libx264', crf 20)
```

```
$ ffrwd compile -f query.sql -v ladder=tests/fixtures/ladder/master.m3u8 -v dest=archive.mkv
ffmpeg -i tests/fixtures/ladder/master.m3u8 -map 0:v:0 -map 0:v:1 -c:0 libx264 -c:1 \
  libx264 -crf:0 20 -crf:1 20 archive.mkv
```

Reach for this to archive a ladder as one file, or to hand a player
something it can switch tracks on instead of a set of separate
renditions.

## 110. Every video rung muxed with every audio rendition, one mp4 each

A self-join of the rendition table against itself pairs each video
rung with each audio rendition. `a.height IS NULL` is what tells an
audio-only row apart - an audio rendition has no video stream to
measure a height from - and the destination reads `a.bandwidth`,
since a rendition's `name` only ever comes from an HLS master, never
a DASH one. Both go through `::text` before `||`, the same cast
recipe 81's `n`-keyed names take, so each pairing lands in its own
file:

```pgsql
COPY (
  SELECT v.video[1], a.audio[1]
  FROM input(:'ladder') v, input(:'ladder') a
  WHERE v.height >= 480 AND a.height IS NULL
) TO (:'prefix' || v.height::text || 'p-' || a.bandwidth::text || '.mp4')
  WITH (video_codec 'libx264', crf 20, audio_codec 'aac')
```

```
$ ffrwd compile -f query.sql -v ladder=tests/fixtures/ladder-demuxed/master.mpd -v prefix=out-
ffmpeg -i tests/fixtures/ladder-demuxed/master.mpd -map 0:v:0 -map 0:a:0 -c:0 libx264 \
  -crf:0 20 -c:1 aac out-1080p-69000.mp4 -map 0:v:1 -map 0:a:0 -c:0 libx264 -crf:0 20 \
  -c:1 aac out-720p-69000.mp4
```

Reach for this to produce per-rendition downloads from a demuxed
ladder - one whose audio rides in its own renditions rather than
muxed into each video variant. An HLS master reads back one row per
`#EXT-X-STREAM-INF` variant only - a variant's own bound audio group
folds into that row rather than surfacing as a row of its own - so a
ladder with an audio-only row for `a` to pick up needs a DASH MPD,
whose `<Representation>`s stay one row apiece regardless of type.
