import argparse

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--input_dropout", type=float, default=0.3)
    parser.add_argument("--rule_1", type=str, default="True")
    parser.add_argument("--rule_2", type=str, default="True")
    parser.add_argument("--postfix", type=str, default="True")

    return parser.parse_args()
