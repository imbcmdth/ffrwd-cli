CREATE VIEW master AS
  SELECT scale(f.video[1], 1920, -2) AS v, volume(f.audio[1], 0.9) AS a
  FROM input('film.mkv') f;

COPY (SELECT scale(m.v, 1280, -2) AS v, m.a FROM master m)
TO '720.mp4' WITH (video_codec 'libx264', crf 21, audio_codec 'aac');

COPY (SELECT scale(m.v, 640, -2) AS v, m.a FROM master m)
TO '360.mp4' WITH (video_codec 'libx264', crf 26, audio_codec 'aac');

COPY (SELECT m.a FROM master m)
TO 'audio.m4a' WITH (audio_codec 'aac', audio_bitrate '128k')
