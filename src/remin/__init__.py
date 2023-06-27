import importlib
import argparse
import json
from remin.solver import Solver, make_trainer
from remin.residual import make_loader
from remin import callbacks
import torch
import numpy as np

__version__ = '0.2.0'


def check_module(path: str) -> bool:
    try:
        importlib.util.find_spec(path)
        return True
    except:
        return False


def check_attribute(path: str, attr_name: str) -> bool:
    try:
        getattr(importlib.import_module(path), attr_name)
        return True
    except:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)

    import sys
    sys.path.append('.')

    args = parser.parse_args()
    config_file = args.config

    with open(config_file) as file:
        config = json.load(file)
    # Configuration
    confs = config.get('conf')
    if confs:
        seed = confs.get('seed')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f'Random seed value is set to: {seed}.')
        device = confs.get('device')
        if device is not None:
            if device == 'prefer_cuda':
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    torch.set_default_device(device)
                    print('Default device is set to cuda.')
                else:
                    print('Default device could not set to cuda, falling back to cpu.')
            try:
                dvc = torch.device(device)
                torch.set_default_device(dvc)
                print(f'Default device is set to {device}.')
            except RuntimeError as error:
                print(f'Default device could not set to {device}.')
                exit(-1)
        matml_prec = confs.get('matmul_precision')
        if matml_prec is not None:
            torch.set_float32_matmul_precision(matml_prec)
            print(f'Float32 Matmul precision is set to {matml_prec}.')

    # Training Configuration
    models = config.get('models')
    invalid_model = False
    for i, model in enumerate(models):
        if model.get('name') is None:
            print(f'No model name specified for the {i}th model in the list.')
            invalid_model = True
        if model.get('loss') is None:
            print(f'No loss is specified for {model.get("name")}.')
            invalid_model = True
        model_path = model.get('model_path')
        if model_path is None:
            print(f'No model path is specified for {model.get("name")}.')
            invalid_model = True
        else:
            if isinstance(model_path, list) and len(model_path) == 2:
                if check_module(model_path[0]) == False:
                    print(f'Could not find the model loader for {model.get("name")}.')
                    invalid_model = True
                else:
                    if check_attribute(model_path[0], model_path[1]) == False:
                        print(
                            f'No model with the name {model_path[1]} for {model["name"]}.'
                        )
                        invalid_model = True
                    for item in model['loss'].items():
                        if check_attribute(model_path[0], item[1]) == False:
                            print(f'Invalid loss {item[1]} for {model.get("name")}.')
                            invalid_model = True
            else:
                print(f'Invalid model path specification for {model["name"]}.')
                invalid_model = True
        data_path = model.get('data_path')
        if data_path is None:
            print(f'No data path is specified for {model["name"]}.')
            invalid_model = True
        else:
            if isinstance(data_path, list) and len(data_path) == 2:
                if check_module(data_path[0]) == False:
                    print(f'Could not find the data loader for {model.get("name")}.')
                    invalid_model = True
                else:
                    if check_attribute(data_path[0], data_path[1]) == False:
                        print(
                            f'No data with the name {data_path[1]} for {model["name"]}.'
                        )
                        invalid_model = True
            else:
                print(f'Invalid data path specification for {model["name"]}.')
                invalid_model = True
        if model.get('epochs') is None:
            print(f'No epochs is specified for {model.get("name")}.')
            invalid_model = True
        else:
            if model.get('epochs') < 0 or not isinstance(model.get('epochs'), int):
                print(f'Number of epochs must be a positive integer.')
                invalid_model = True
        if model.get('lr') is None:
            print(f'No learning rate is specified for {model.get("name")}.')
            invalid_model = True
        else:
            if model.get('lr') < 0:
                print(f'Learning rate must be positive.')
                invalid_model = True

    if invalid_model:
        print('Exitting due to above errors.')
        exit(-1)

    for model in models:
        torch.cuda.empty_cache()
        print(f'Begin training {model["name"]}:')

        ModelClass = getattr(importlib.import_module(model['model_path'][0]),
                             model['model_path'][1])
        data = getattr(importlib.import_module(model['data_path'][0]),
                       model['data_path'][1])

        instance = ModelClass()

        loader = make_loader(data, **model['loader'])

        epochs = model['epochs']
        lr = model['lr']

        optimizer = torch.optim.Adam(instance.parameters(), lr=lr)

        losses = model['loss']
        residual_loss, metric_loss = None, None

        residual_loss = getattr(importlib.import_module(model['model_path'][0]),
                        losses['resloss'])
        if losses.get('metloss'):
            metric_loss = getattr(importlib.import_module(model['model_path'][0]),
                            losses['metloss'])
        
        trainer = make_trainer(loader,
                               optimizer=optimizer,
                               residual_loss=residual_loss,
                               metric_loss=metric_loss)

        outfolder = 'outputs'
        if model.get('outputs'):
            outfolder = '/'.join(model['outputs'].split('.'))

        solver = Solver(instance, model['name'], outfolder, trainer=trainer)

        if model.get('callbacks'):
            calls = model['callbacks']
            callist = []
            for key in calls.keys():
                if key == 'total_time':
                    callist.append(callbacks.TotalTimeCallback())
                if key == 'save':
                    callist.append(callbacks.SaveCallback())
                if key == 'log':
                    callist.append(callbacks.LogCallback(calls[key][0], calls[key][1]))
                if key == 'plot':
                    for pkey in calls[key].keys():
                        callist.append(
                            callbacks.PlotCallback(state=pkey, name=calls[key][pkey]))
            solver.reset_callbacks(*callist)

        solver.fit(epochs)
