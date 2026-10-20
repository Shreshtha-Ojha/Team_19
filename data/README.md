# LOL dataset (optional)

**You don't need this to use the app or the command-line tool.** It's only
needed to reproduce the PSNR/SSIM measurements in the main README.

The LOL dataset (Wei, Wang, Yang, Liu, BMVC 2018) contains pairs of photos of
the same scene, one shot in low light and one in normal light. The
normal-light photo serves as the reference that enhanced images are scored
against. Project page: https://daooshee.github.io/BMVC2018website/

## Download

It has to be downloaded by hand from one of:

- **Google Drive:** https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB
- **Baidu Pan:** https://pan.baidu.com/s/1ABMrDjBTeHIJGlOFIeP1IQ (extraction code `acp3`)

## Where to put it

Unzip it into `data/raw/LOL/` so the folders look like this:

```
data/raw/LOL/lol_dataset/
├── our485/{low,high}/   # 485 pairs, used to pick the pipeline's settings
└── eval15/{low,high}/   # 15 pairs, used only for the reported scores
```

Then check the layout (from the repository root):

```bash
python src/data/verify_lol.py
```

It lists the image folders it finds and confirms that every low-light photo
has a normal-light partner with the same filename.
