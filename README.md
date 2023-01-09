# Crab.Toolkit.SExtractorPlus

A personal toolkit to run SExtractor Classic and Plus Plus

## First step

Make sure you have SExtractor Classic command in your terminal. 

## Examples 

### Run SExtractor Classic for an input image

```
/path/to/Crab.Toolkit.SExtractorPlus/bin/sextractor_classic_go_find_sources.py \
    input_image.fits
```

The default output directory will be `{input_image}_run_sextractor_classic_dir`. There will be a `SExtractor_OutputCatalogXY.csv` which contains the (x, y) pixel coordinates of each detected sources. 

See `sextractor_classic_go_find_sources.py --help` for more options, e.g., detection thresholds, output directory name, etc. 


