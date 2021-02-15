import argparse
import json

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dict", type=json.loads, default='{}')
args = parser.parse_args()

print(args.dict)
