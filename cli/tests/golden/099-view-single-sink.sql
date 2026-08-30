CREATE VIEW master AS
  SELECT scale(f.video[1], 1280, -2) AS v, volume(f.audio[1], 0.8) AS a
  FROM input('film.mp4') f;

COPY (SELECT gblur(master.v, 2), master.a FROM master)
TO '720.mp4' WITH (video_codec 'libx264', crf 20)
