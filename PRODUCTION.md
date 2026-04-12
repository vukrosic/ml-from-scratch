# Production Workflow

## Per-lesson checklist

1. **Code** — write all `.py` files in the lesson folder
2. **README** — write the article (same text goes to blog)
3. **Blog** — copy to `/root/opensuperintelligencelab/src/content/{slug}.mdx`
4. **Skool $49** — extended notebook/scripts (NOT in this repo)
5. **Record** — screen-record scrolling through blog article
6. **Upload** — YouTube with description linking to this repo's lesson folder
7. **Post** — X + LinkedIn with key insight from the lesson

## YouTube description template

```
{title} — ML From Scratch

Code: https://github.com/vukrosic/ml-from-scratch/tree/main/{series}/{lesson}/
Blog: https://vukrosic.vercel.app/blog/{slug}/
Extended materials: https://www.skool.com/opensuperintelligencelab

All code is free and open source.
```

## Naming conventions

- Series folders: lowercase-kebab, no numbers, NEVER rename
- Lesson folders: NNN-slug/, append-only, NEVER reorder
- Code files: lowercase_snake.py
- One README.md per lesson (this is the article)

## Publish order

Record in any order. Publish order = lesson number within series.
Series can interleave on YouTube (e.g. Mon=pytorch, Tue=transformer, Wed=training).
