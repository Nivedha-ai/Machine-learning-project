# Inventory Demand Prediction for Jupyter

1. Extract the ZIP into a folder.
2. Open Inventory_Demand_Prediction.ipynb in Jupyter Notebook or JupyterLab.
3. Select a Python 3 kernel (Python 3.10+ recommended).
4. Run the installation cell once if needed, then Run All.
5. Keep sample_monthly_sales.csv in the notebook working folder.

If Jupyter is not installed: run `python -m pip install notebook` then `python -m notebook` in a terminal.

Includes 960 synthetic records for 10 items, 2 warehouses and 48 months, Jan 2022–Dec 2025. Generated for education with random seed 42, seasonal demand, gradual growth and Gaussian noise; not real customer data. The notebook forecasts Jan 2026 from this sample. No API key or SAP access needed.

Files: notebook, equivalent Python source, CSV dataset, requirements, and example evaluation/forecast outputs. Notebook cells are intentionally unexecuted so you can run them yourself. The equivalent code was executed to verify the workflow. Error measures are MAE, RMSE and WAPE, compared with a 3-month-average baseline. Held-out evaluation is one-step-ahead over the final six months, using only information known before each predicted month.
