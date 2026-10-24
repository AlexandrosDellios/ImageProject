# Synthesis assets

Inputs of the synthetic-data generator (`uno_vision/synth`, `scripts/generate_synthetic.py`).

- `cards/<class>.png`: one RGBA cut-out per class, 56 in all (the 54 card faces plus
  `black_token` and `yellow_token`). File names are the class names of
  `uno_vision/classes.py`; `scripts/generate_synthetic.py` refuses any other name.
- `backgrounds/*.png`: patches of the two table surfaces, `white_*` (plain table) and
  `complexe_*` (leaf-pattern tablecloth). The generator crops random 336 x 336 squares from
  them at their native resolution.

The cut-outs and patches were cut by hand from photos of the course dataset with a free-form
screenshot tool. These are cleaned copies of those snips:

- Cards downscaled (Lanczos) so the longest side is at most 400 px. The generator draws a
  card at most about 200 px tall, so 400 px keeps twice the detail it needs. With the same
  seed, samples drawn from these copies and from the full-size snips (up to 1177 px) differ
  by under 0.3 grey levels on average (PSNR above 50 dB), far below the noise the generator
  adds; in about 1 sample in 10 the rounded card width moves by 1 px. `yellow_token`
  (339 x 337) was already smaller and is pixel-identical to its snip; `black_token` went from
  295 x 445 to 265 x 400.
- Backgrounds kept at native resolution, except `complexe_1` (1188 x 1285 -> 1109 x 1200),
  and saved as RGB (the snips were fully opaque RGBA).
- No metadata: the snipping tool's text chunks, sRGB, gamma and dpi chunks are dropped.
  Saved with Pillow `optimize=True`. Total about 15 MB.
