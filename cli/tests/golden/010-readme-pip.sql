WITH pip AS (
  SELECT scale(crop(b.video[1], 1200, 50, 600, 200), 'iw/2') AS frame
  FROM input('game.mp4') b
)
SELECT overlay(a.video[1], pip.frame, 20, 20)
FROM input('game.mp4') a, pip
