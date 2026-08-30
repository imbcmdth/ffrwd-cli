SELECT a.video[1]
FROM input('clip.mp4') a
GROUP BY a.video[1]
