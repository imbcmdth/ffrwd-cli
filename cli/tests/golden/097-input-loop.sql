SELECT overlay(f.video[1], logo.video[1], 20, 20)
FROM input('film.mp4') f, input('logo.png', loop => true, framerate => 15) logo
