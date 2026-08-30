COPY (
  SELECT f.video[1], f.audio[1], s.subtitle[1]
  FROM input('film.mp4') f, input('subs.en.vtt') s
) TO 'out.mp4' WITH (
  subtitle_codec 'mov_text'
)
