# UNSW-NB15 — Download Checklist

**Target folder (already created):** `D:\DATA\UNSW-NB15\`
**PCAP subfolder (already created):** `D:\DATA\UNSW-NB15\pcap\`

## Source (official — academic use, must cite the 5 papers on the page)
1. Open: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Click the **"download from HERE"** link → opens a Microsoft SharePoint folder.

## CSVs to download first (a few GB — unblocks modeling)
- [ ] `UNSW-NB15_1.csv`
- [ ] `UNSW-NB15_2.csv`
- [ ] `UNSW-NB15_3.csv`
- [ ] `UNSW-NB15_4.csv`
- [ ] `NUSW-NB15_features.csv`  (feature names/types — needed to read the 4 files)
- [ ] `UNSW-NB15_GT.csv`  (ground truth)
- [ ] `UNSW-NB15_LIST_EVENTS.csv`  (event list)

**Or** the ready-made partition (smaller, quicker to start):
- [ ] `UNSW_NB15_training-set.csv`  (175,341 rows)
- [ ] `UNSW_NB15_testing-set.csv`  (82,332 rows)

## Optional: PCAPs (~100 GB — only if you want packet-level features)
- [ ] pcap files → save into `D:\DATA\UNSW-NB15\pcap\`
- Disk check: 176 GB free; ~100 GB PCAPs leaves ~76 GB. OK but tight.

## Rules to avoid the previous broken download
- **Download files INDIVIDUALLY.** Do NOT use SharePoint **"Download all as zip"** — that is what produced the `*_Error.txt` stubs last time.
- Keep the **original filenames** (don't rename).
- If a file fails, re-download just that one file.

## When finished
Tell the agent: **"UNSW-NB15 files are in D:\DATA\UNSW-NB15"**. It will then:
1. Verify integrity — confirm files are real data (not HTML/stubs), check sizes + row counts.
2. Inspect the 49-feature schema + `attack_cat` / `Label`.
3. Update `datasets.yaml`, `feature_mapping.yaml`, `dataset_compatibility.csv`.
4. Slot UNSW-NB15 into role **B (generalization)** → CICIoT2023 returns to pure OOD.
