SELECT a.video[1]
FROM input('one.mp4') a
UNION ALL
SELECT b.video[1]
FROM input('two.mp4') b
