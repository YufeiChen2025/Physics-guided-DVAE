# Data sources

Download and extract the author-posted ZIP for each required release.

| Version | Archive | Bytes | Publisher's archive MD5 |
|---|---|---:|---|
| [3W 2.0.0](https://doi.org/10.6084/m9.figshare.29205836.v1) | 3w_dataset_2.0.0.zip | 1,794,819,830 | 2c23b87b60c5d19ed9cf9559efa6ffa7 |
| [3W 1.1.1](https://doi.org/10.6084/m9.figshare.29205947.v1) | 3w_dataset_1.1.1.zip | 722,500,556 | ae105a8abff4ac33058fea3ebb991fc2 |

Verify a downloaded archive with Get-FileHash <archive.zip> -Algorithm MD5 in PowerShell or md5sum <archive.zip> on Linux. The --data-root value must point directly at the level containing class folders 0 through 9 for 2.0.0 or 0 through 8 for 1.1.1. Each class folder contains original Parquet or CSV instances. The publisher's 1.1.1 ZIP has DATASET = 1.1.0 in dataset.ini. Use the versioned Figshare release and archive checksum to identify it.

The instance split lists are in protocols/v2/fixed_split.json and protocols/v111/fixed_split.json. The split is applied before generating windows. During 1.1.1 preparation the code rebuilds the original-observation mask and checks the frozen normalization, file membership and window counts. Train/validation/test window counts are 392,484/87,405/85,717 for 2.0.0 and 276,108/61,744/57,401 for 1.1.1. Keep several GB available for the CSV-to-NumPy cache.

The underlying data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The distributor's license files are included under licenses/. Cite the versioned dataset DOI, [Petrobras 3W](https://github.com/petrobras/3W) and the relevant paper shown in [Petrobras' citation instructions](https://github.com/petrobras/3W/blob/main/CITATION.md). The 2.0.0 data article is [Vaz et al., Scientific Data](https://doi.org/10.1038/s41597-026-07225-z).
