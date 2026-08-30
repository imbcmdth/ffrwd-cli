SELECT overlay(f.video[1], p.video[1], '(W-w)/2', '(H-h)/2')
FROM input('film.mp4') f, input('logo.png') p
