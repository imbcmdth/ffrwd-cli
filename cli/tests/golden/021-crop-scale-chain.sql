SELECT scale(crop(a.video[1], 10, 20, 300, 400), 640, 480)
FROM input('clip.mp4') a
