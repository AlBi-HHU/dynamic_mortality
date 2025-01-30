# Dynamic Prediction of Mortality Risk Following Allogeneic Hematopoietic Stem Cell Transplantation

### How to run the pipeline

The pipeline uses the workflow management system snakemake (version 8.14.0) and conda for dependency management. To run the workflow to predict mortality for day 100 using all available cores run
```
snakemake --use-conda --cores all
```
