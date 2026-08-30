WITH c AS (
  SELECT scale(a.video[1], 'iw/2') AS frame
  FROM input('clip.mp4') a
)
SELECT overlay(c.frame, c.frame, 10, 10)
FROM c
