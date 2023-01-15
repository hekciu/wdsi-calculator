import json
import parser


def is_config_valid(config):
    keys = ['MODEL_PATH', 'MODEL_TRAIN_DATA_PATH', 'train_from_scratch']
    for key in keys:
        if key not in config:
            return False
    return True


def load_config():
    with open('config.json') as config_file:
        config = json.load(config_file)
        return config if type(config) is dict else {}
    return {}


def parse_and_compute_equation(eq_arr, print_result=False):
    eq_str = ''
    for el in eq_arr:
        eq_str += el[0][0]
    result = eval(parser.expr(eq_str).compile())
    if print_result:
        print("equation")
        print(eq_str)
        print("result")
        print(result)
