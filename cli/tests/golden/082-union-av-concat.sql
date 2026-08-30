SELECT a.video[1], a.audio[1]
FROM input('intro.mp4') a
UNION ALL
SELECT b.video[1], b.audio[1]
FROM input('main.mp4') b
