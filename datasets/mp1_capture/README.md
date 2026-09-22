# ECE 484 MP1 Lane Dataset

This dataset contains 506 synchronized samples collected from the
Silverstone Gazebo environment.

Each sample includes:

- `images/N.png`: 800x600 BGR camera image
- `masks/N.png`: 800x600 binary lane mask, values 0 or 255
- `poses/N.json`: vehicle position and orientation

## Reconstruct the archive

From this directory:

```bash
cat mp1_capture_dataset.tar.gz.part-* \
  > mp1_capture_dataset.tar.gz
```

Verify all files:

```bash
sha256sum -c SHA256SUMS
```

All entries should report `OK`.

## Extract into the MP1 project

Run this from `datasets/mp1_capture` in the repository:

```bash
tar -xzf mp1_capture_dataset.tar.gz \
  -C ../../code/src/mp1
```

The extracted dataset will be located at:

```text
code/src/mp1/data/capture/
├── images/
├── masks/
└── poses/
```

Warning: extracting may overwrite files with matching numeric names in an
existing dataset. Use a fresh clone or back up existing capture data first.

## Expected integrity

```text
images: 506
masks:  506
poses:  506
IDs:    0 through 505
image size: 800x600
mask values: 0 or 255
```

The `data` directory is ignored by Git, so extracted files should not appear
in normal `git status`.
