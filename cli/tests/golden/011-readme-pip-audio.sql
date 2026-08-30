WITH pip AS (
  SELECT scale(crop(b.video[1], 1200, 50, 600, 200), 'iw/2') AS frame,
         b.audio[1] AS sound
  FROM input('game.mp4') b
)
SELECT overlay(a.video[1], pip.frame, 20, 20),
       amix(volume(a.audio[1], 0.65), volume(pip.sound, 0.35))
FROM input('game.mp4') a, pip
