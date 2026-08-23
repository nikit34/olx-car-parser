# Damage segmentation dataset

Photos of damaged cars from OLX-PT, annotated for instance segmentation of the
damage itself (not of the car, not of the panel).

## Why it exists

The production photo classifier (`damage_classifier_v2.pt`, ResNet50) reports
`val_precision 0.933` in its own checkpoint and scores **0.20 precision in the
field** — 8 of its 10 most confident flags on a 2026-08-23 stratified sample
were undamaged cars. It was trained on an external binary dataset
(`/tmp/yolo_data/drbimmer_binary`), so it learned that distribution rather than
ours, and the flag it produces is a hard veto in `_blocking_deal_reason`: it
currently removes 571 of 20 497 active listings from the deal feed, more than
every text-based veto combined.

A detector trained on OUR photos, that localises the damage instead of
scoring the whole frame, is the honest replacement — localisation also gives
the buyer evidence rather than a number.

## How annotation works

A language model can say **where** damage is and **what** it is; it cannot
draw a pixel boundary. So the work is split:

1. `scripts/harvest_damage_seg.py` mines listings with Portuguese salvage
   vocabulary ("sinistrado", "batido", "para peças", …) — damage is a few
   percent of the corpus, so random sampling is hopeless — and stages the
   photos. The existing CLIP exterior filter drops interiors and engine bays.
2. Claude looks at each photo and writes boxes + a subtype into `ann/*.json`.
3. `scripts/damage_seg_masks.py` turns those into YOLO-seg labels and renders
   an overlay for every image so the annotation can be re-checked against what
   was actually written.

### The SAM rule, and why it is what it is

Prompted with a box, SAM segments the salient **object** inside it. Measured on
the first images:

- **Object-like damage** — a smashed lamp lens, a hanging bumper, a panel
  that is gone: the mask came back tight and correct. These get real masks.
- **Surface deformation** — a dent or a crease: a loose box returned the whole
  bumper (SAM found the panel), and a tight box returned a 1 695-px fragment of
  the dent's shadow, 14 % of the box. Both would teach a model the wrong thing.
  These keep their box as a rectangular polygon.
- **Absences with no silhouette** — an empty headlight cavity: SAM returned a
  thin strip of the shadow line. Box as well.

So `SAM_SUBTYPES` vs `BOX_SUBTYPES` in the script is not a style choice, it is
where SAM was measured to stop working. Every label records which path made it,
and overlays colour them differently (lime = SAM mask, magenta = box-derived)
so the two are never confused when reviewing.

### What counts as damage

Only what a buyer would pay to fix: dents, creases, crash damage, broken or
missing parts, cracked glass, exposed structure. **Not** dirt, faded lamps,
dull paint, or light bumper scuffs — those are age, and folding them into the
same class is how you get a detector that fires on every used car.

A photo of a damaged car that shows no damage in **that frame** is annotated
with an empty list, on purpose. Those are the hard negatives — a glossy black
flank from a "para peças" listing is exactly the image the ResNet scored 1.00.

## Layout

    raw/          originals as downloaded
    images/       exterior-filtered, resized to 900 px wide — the dataset images
    ann/*.json    {photo_id: [{"box": [x1,y1,x2,y2], "type": ..., "note": ...}]}
    labels/       generated YOLO-seg polygons (regenerate, don't hand-edit)
    overlays/     generated visual check
    subset/       train/val split of the annotated images only

`ann/` is the source of truth and is versioned. Everything else is derived or
bulky, and is ignored.
