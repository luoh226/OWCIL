import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    # print(args)
    if args["cfg_options"] is not None:
        for arg in args["cfg_options"]:
            k, v = arg.split('=')

            if k == 'seed':
                v = v.split(",")
                v = [int(seed) for seed in v]
            elif v.count(".") == 1 and not v.startswith(".") and not v.endswith("."):
                v = float(v)
            elif v.isdigit():
                v = int(v)
            elif v == 'true':
                v = True
            elif v == 'false':
                v = False
            else:
                pass
            args[k] = v

    # print('=')
    # print(args)
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--cfg-options', nargs='+')

    return parser


if __name__ == '__main__':
    main()
