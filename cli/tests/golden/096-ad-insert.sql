SELECT overlay(f.video[1], ffrwd.delay(scale(a.video[1], 'iw*0.33', 'ih*0.33'), 1), 20, 20) AS frame,
       amix(f.audio[1], volume(adelay(a.audio[1], 1000), 0.5)) AS sound
FROM input('film.mp4') f, input('ad.mp4') a
