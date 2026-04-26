import sys
from app.product.cli import run_ml_comparison

if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "momentum"
    run_ml_comparison(strategy)
