SELECT overlay(a.video[1], scale(b.video[1], 'iw*0.3'), 50, 50)
FROM input('main.mp4') a, input('logo.mp4') b
