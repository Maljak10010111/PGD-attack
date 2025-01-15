from functools import partial
from torch import nn
from robustbench import load_model

carmon_2019 = {
    'name': 'Carmon2019Unlabeled',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}
augustin_2020 = {
    'name': 'Augustin2020Adversarial',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'L2'
}

standard = {
    'name': 'Standard',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}
gowal_2021 = {
    'name': 'Gowal2021Improving_70_16_ddpm_100m',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}
chen_2020 = {
    'name': 'Chen2020Adversarial',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}

xu_2023 = {
    'name': 'Xu2023Exploring_WRN-28-10',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}

addepalli2022 = {
    'name': 'Addepalli2022Efficient_RN18',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}

ding2020 = {
    'name': 'Ding2020MMA',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'L2'
}

engstrom2019 = {
    'name': 'Engstrom2019Robustness',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'L2'
}


def load_robustbench_model(name: str, dataset: str, threat_model: str) -> nn.Module:
    model = load_model(model_name=name, dataset=dataset, threat_model=threat_model, model_dir="./checkpoints")
    return model


# Available CIFAR-10 models for use! Sourced from RobustBench library.
_local_cifar_models = {
    'carmon2019': partial(load_robustbench_model, name=carmon_2019['name'], dataset=carmon_2019['dataset'],
                          threat_model=carmon_2019['threat_model']),
    'augustin2020': partial(load_robustbench_model, name=augustin_2020['name'], dataset=augustin_2020['dataset'],
                            threat_model=augustin_2020['threat_model']),
    'standard': partial(load_robustbench_model, name=standard['name'], dataset=standard['dataset'],
                        threat_model=standard['threat_model']),
    'gowal2021': partial(load_robustbench_model, name=gowal_2021['name'], dataset=gowal_2021['dataset'],
                         threat_model=gowal_2021['threat_model']),
    'chen2020': partial(load_robustbench_model, name=chen_2020['name'], dataset=chen_2020['dataset'],
                        threat_model=chen_2020['threat_model']),
    'xu2023': partial(load_robustbench_model, name=xu_2023['name'], dataset=xu_2023['dataset'],
                      threat_model=xu_2023['threat_model']),
    'addepalli2022': partial(load_robustbench_model, name=addepalli2022['name'], dataset=addepalli2022['dataset'],
                             threat_model=addepalli2022['threat_model']),
    'ding2020': partial(load_robustbench_model, name=ding2020['name'], dataset=ding2020['dataset'],
                                 threat_model=ding2020['threat_model']),
    'engstrom2019': partial(load_robustbench_model, name=engstrom2019['name'], dataset=engstrom2019['dataset'],
                                     threat_model=engstrom2019['threat_model'])
}


def get_local_cifar10_model(name: str) -> nn.Module:
    return _local_cifar_models[name]()