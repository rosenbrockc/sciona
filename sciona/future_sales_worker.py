"""Fresh-process execution keeps native LightGBM state outside the graph runner."""
import json
import sys
from sciona.future_sales_contract import prepare
from sciona.future_sales_features import build_panel
from sciona.future_sales_training import train_forecast


def run(payload):
    p=prepare(payload).payload
    panel=build_panel(p['transactions'],p['entities'],p['boundaries'],**p['feature_controls'])
    result=train_forecast(panel,p['training_controls']);result['predictions']=result['predictions'].tolist()
    return result


def main():
    try:
        result=run(json.load(sys.stdin))
        sys.stdout.write(json.dumps(result,allow_nan=False))
    except Exception:
        # Do not propagate private payload details or native-library diagnostics.
        sys.stderr.write('Future Sales worker failed\n')
        raise SystemExit(1) from None


if __name__=='__main__':main()
