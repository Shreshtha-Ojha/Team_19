# LOL Dataset — manual download required

This project uses the **LOL dataset** from Chen Wei, Wenjing Wang, Wenhan Yang,
Jiaying Liu, *"Deep Retinex Decomposition for Low-Light Enhancement"*, BMVC 2018.

I could not script this download: it is hosted on Google Drive / Baidu Pan,
both of which require manual interaction (Google Drive shows a virus-scan /
large-file confirmation page for programmatic requests; Baidu Pan requires a
login and an extraction code). Source verified against the authors' project
page (https://daooshee.github.io/BMVC2018website/) on 2026-10-16.

## What to download

Pick ONE of:

- **Google Drive**: https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB
- **Baidu Pan**: https://pan.baidu.com/s/1ABMrDjBTeHIJGlOFIeP1IQ (extraction code: `acp3`)

## Where to put it

Unzip the archive and place its contents under:

```
data/raw/LOL/
```

so that it looks like this (the layout verified on the real archive):

```
data/raw/LOL/lol_dataset/
├── our485/{low,high}/   # 485 pairs — used for tuning pipeline parameters
└── eval15/{low,high}/   # 15 pairs  — test set for reported metrics
```

## Verification

Once downloaded, run:

```
python src/data/verify_lol.py
```

This will report, without assuming anything about internal naming:
- number of images found per split/subfolder
- whether low-light and normal-light images pair up 1:1 by filename
- image dimensions and any mismatches

Evaluation will not proceed until this passes.
