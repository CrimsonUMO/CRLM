# Universal Pathological Features Extractor

A package for universal pathological feature extraction.

## Environment

This script is an extension based on the TIAToolbox framework. You can download the repository and create the environment using the provided .yml file.

```bash
git clone https://github.com/CrimsonUMO/CRLM.git
conda env create -f CRLM.yml
```

## Quickstart

Here's a simple example of how to perform feature extract:

```python
tissues = ["BACK","DEB","LIN","LYM","STR","TUM"]
NUM_CLASSES = len(tissues)
label_dict = {i:label for label,i in zip(tissues,range(0,NUM_CLASSES))}
print(label_dict)

### initiate
csv_path = "./patch_predict.csv"

extractor = feature_extractor(
    output_df=csv_path,
    label_dict=label_dict)

### get original matrix
extractor.get_matrix()

### dilate for pericancerous area
extractor.dilate()

### feature extractor
extractor.calculate_first_order()
extractor.feature["First_order"].head()

extractor.calculate_interaction()
extractor.feature["Interaction"].head()

extractor.calculate_spatial(method = "DBSCAN",verbose = True)
extractor.feature["Spatial"].head()
df_points = extractor.points_cluster

### output
re = extractor.output()
re.head()
```

A detailed workflow is available and executable in the `Tutorial.ipynb` notebook.

## License

This project is released under Apache License 2.0. You can view the complete license text through the following link:

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

This project relies on multiple open-source libraries, which may have different licenses. The following is an overview of the main dependencies and their licenses:

- **pyvips**: Using [LGPL 2.1](https://libvips.github.io/libvips/Licensing.html) License. Due to the dynamic linking of this project to `pyvips`, users need to ensure compliance with the terms of LGPL 2.1, especially when distributing or modifying the software.

- Other dependencies (such as NumPy, Pandas, OpenCV, etc.) are licensed under a permissive license (such as BSD or MIT) that allows for free use and distribution.

## 

