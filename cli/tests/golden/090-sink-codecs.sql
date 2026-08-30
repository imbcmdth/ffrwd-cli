COPY (
  SELECT a.video[1], a.audio[1]
  FROM input('foo.mp4') a
) TO 'out.mkv' WITH (
  video_codec 'libx264', crf 20, preset 'slow',
  audio_codec 'aac', audio_bitrate '192k', faststart true
)
