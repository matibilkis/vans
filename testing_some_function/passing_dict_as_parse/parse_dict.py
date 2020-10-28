import json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--noise_model', '--my-dict', type=json.loads)
    args = parser.parse_args()

    print(type(args.noise_model))
    print(args.noise_model)
    for k in args.noise_model.values():
        print(type(k))
