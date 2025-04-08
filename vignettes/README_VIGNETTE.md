# TSUNAMI Vignette Notes

The original `tsunami_vignette.ipynb` notebook had formatting issues that made it difficult to repair directly. The content has been converted to Python script format in the following files:

- `tsunami_vignette.py`: Main Python script version of the vignette
- `tsunami_vignette_final.py`: Working backup copy

These Python scripts demonstrate all the key features of the TSUNAMI package including:

1. Single Risk Survival Analysis
2. Competing Risks Analysis (with fallback for when CompetingRisksHead is not available)
3. Multi-Task Learning
4. Sample Weights Support

The scripts are fully runnable and handle all import issues gracefully.

To regenerate a Jupyter notebook version, you can use:

```bash
jupyter nbconvert --to notebook --execute vignettes/tsunami_vignette.py --output tsunami_vignette.ipynb
```

Note that this requires Jupyter to be installed.
