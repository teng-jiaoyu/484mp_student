# ECE 484 MP1 Headless Capture Dataset

This archive contains 2,200 synchronized raw samples collected by driving the
GEM vehicle in the Silverstone Gazebo environment. The final 200 samples
(IDs 2000 through 2199) cover one complete 1,369 m lap and are spaced by
approximately 6.84 m.

Each sample includes:

- `images/N.png`: 800x600 BGR camera image
- `poses/N.json`: synchronized vehicle position and orientation

The raw collection intentionally does not contain generated masks. Run the MP1
preprocessing step after extraction to create training masks.

## Reconstruct and verify the archive

From this directory:

```bash
cat mp1_headless_2200.tar.gz.part-* > mp1_headless_2200.tar.gz
sha256sum -c SHA256SUMS
```

All entries should report `OK`.

## Extract into the MP1 project

```bash
tar -xzf mp1_headless_2200.tar.gz -C ../../code/src/mp1
```

The extracted dataset will be located at:

```text
code/src/mp1/data/collection_20260924_headless/
├── images/  # 2200 PNG files, IDs 0 through 2199
└── poses/   # 2200 JSON files, IDs 0 through 2199
```

The collector verified that all 2,200 PNG files are readable 800x600 images
and that no files are exact duplicates.

