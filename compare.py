
# run_pipeline_now.py
from New_model import run_pipeline

res = run_pipeline(
    data_dir=r"C:\Users\karanvsi\Desktop\Pycharm\Cache\cache_daily_new",
    out_dir=r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full",
    symbols_like=None,
    limit_files=None,
    ev_target="oc",
    embargo_days=5,
    export_xlsx=False
)

print("Watchlist:", res["watchlist_path"])
print("Panel:    ", res["panel_path"])
print("OOS:      ", res["oos_report_path"])
print("Calib:    ", res["calibration_table_path"])
print("Models dir:", r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full\models")
