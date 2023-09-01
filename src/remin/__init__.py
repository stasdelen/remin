import sys
import os
import importlib
import argparse
import json
import csv
from remin.solver import Solver, make_trainer
from remin.residual import make_loader
from remin import callbacks
import matplotlib.pyplot as plt
import torch
import numpy as np

__version__ = '0.2.7'


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


def parse_config(config: dict):
    parse_error = False

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
                parse_error = True
        matml_prec = confs.get('matmul_precision')
        if matml_prec is not None:
            torch.set_float32_matmul_precision(matml_prec)
            print(f'Float32 Matmul precision is set to {matml_prec}.')
        postroc = confs.get('postprocess')
        if postroc is not None:
            if check_module(postroc[0]) == False:
                print('Could not find postproces script.')
                parse_error = True
            else:
                if check_attribute(postroc[0], postroc[1]) == False:
                    print(f'No postprocess function with the name {postroc[1]}.')
                    parse_error = True

    # Training Configuration
    models = config.get('models')

    for i, model in enumerate(models):
        if model.get('name') is None:
            print(f'No model name specified for the {i}th model in the list.')
            parse_error = True
        if model.get('loss') is None:
            print(f'No loss is specified for {model.get("name")}.')
            parse_error = True
        model_path = model.get('model_path')
        if model_path is None:
            print(f'No model path is specified for {model.get("name")}.')
            parse_error = True
        else:
            if isinstance(model_path, list) and len(model_path) == 2:
                if check_module(model_path[0]) == False:
                    print(f'Could not find the model loader for {model.get("name")}.')
                    parse_error = True
                else:
                    if check_attribute(model_path[0], model_path[1]) == False:
                        print(
                            f'No model with the name {model_path[1]} for {model["name"]}.'
                        )
                        parse_error = True
                    for item in model['loss'].items():
                        if check_attribute(model_path[0], item[1]) == False:
                            print(f'Invalid loss {item[1]} for {model.get("name")}.')
                            parse_error = True
            else:
                print(f'Invalid model path specification for {model["name"]}.')
                parse_error = True
        data_path = model.get('data_path')
        if data_path is None:
            print(f'No data path is specified for {model["name"]}.')
            parse_error = True
        else:
            if isinstance(data_path, list) and len(data_path) == 2:
                if check_module(data_path[0]) == False:
                    print(f'Could not find the data loader for {model.get("name")}.')
                    parse_error = True
                else:
                    if check_attribute(data_path[0], data_path[1]) == False:
                        print(
                            f'No data with the name {data_path[1]} for {model["name"]}.'
                        )
                        parse_error = True
            else:
                print(f'Invalid data path specification for {model["name"]}.')
                parse_error = True
        if model.get('epochs') is None:
            print(f'No epochs is specified for {model.get("name")}.')
            parse_error = True
        else:
            if model.get('epochs') < 0 or not isinstance(model.get('epochs'), int):
                print(f'Number of epochs must be a positive integer.')
                parse_error = True
        if model.get('lr') is None:
            print(f'No learning rate is specified for {model.get("name")}.')
            parse_error = True
        else:
            if model.get('lr') < 0:
                print(f'Learning rate must be positive.')
                parse_error = True
    return parse_error


def find_model(model_name: str, models: dict):
    model = None
    for m in models:
        if m['name'] == model_name:
            model = m
    return model


def postprocess(model_name: str, config_file: dict):

    conf = config_file['conf']
    models = config_file['models']
    model = find_model(model_name, models)

    if model is None:
        print(f'No model found with the name {model_name}.')
        exit(-1)

    print(f'Started postprocessing {model_name}.')

    ModelClass = getattr(importlib.import_module(model['model_path'][0]),
                         model['model_path'][1])
    postfunc = getattr(importlib.import_module(conf['postprocess'][0]),
                       conf['postprocess'][1])

    file_name = '/'.join(model['outputs'].split('.'))
    post_model = ModelClass()
    mdata = torch.load(file_name + '/' + model_name + '_best.pt')
    post_model.load_state_dict(mdata['model_state_dict'])
    post_model.eval()

    postfunc(post_model, file_name + '/' + model_name)


def train_model(model: dict):
    print(f'\nBegin training {model["name"]}:')

    ModelClass = getattr(importlib.import_module(model['model_path'][0]),
                         model['model_path'][1])
    if model.get('parameters'):
        datalib = importlib.import_module(model['data_path'][0])
        params = model['parameters']
        for key in params.keys():
            old_val = getattr(datalib, key)
            setattr(datalib, key, params[key])
            print(f'Changed {key}:\n\told: {old_val}\n\tnew: {getattr(datalib, key)}')
        data = getattr(datalib, model['data_path'][1])
    else:
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
                            losses['residual'])
    if losses.get('metric'):
        metric_loss = getattr(importlib.import_module(model['model_path'][0]),
                              losses['metric'])

    trainer = make_trainer(loader,
                           optimizer=optimizer,
                           residual_loss=residual_loss,
                           metric_loss=metric_loss)

    outfolder = 'outputs'
    if model.get('outputs'):
        outfolder = '/'.join(model['outputs'].split('.'))
    if outfolder == 'outputs':
        outfolder += '/' + model['name']

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
                if isinstance(calls[key], dict):
                    for pkey in calls[key].keys():
                        callist.append(
                            callbacks.PlotCallback(state=pkey, name=calls[key][pkey]))
                elif isinstance(calls[key], list):
                    for state in calls[key]:
                        callist.append(
                            callbacks.PlotCallback(state=state,
                                                   name=model['name'] + '_' + state))
            if key == 'csv':
                callist.append(callbacks.CSVCallback(model['name'] + '_data.csv'))
            if key == 'earlystopping':
                callist.append(callbacks.EarlyStoppingCallback(**calls[key]))

        solver.reset_callbacks(*callist)
    solver.fit(epochs)


def plot_losses(models: list, ylog=['log', 'metric']):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 5)
    for model in models:
        outpath = '/'.join(model['outputs'].split('.'))
        with open(outpath + '/' + model['name'] + '_data.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            nrow = next(csvreader).index(ylog[1])
            y = [float(row[nrow]) for row in csvreader]
        ax.plot(y, linewidth=2, label=model['name'])
    ax.set_yscale(ylog[0])
    ax.set(title='Loss vs. Epoch', ylabel='Loss', xlabel='Epoch')
    ax.grid(True)
    ax.legend(loc='best')
    plt.savefig('loss_vs_epoch.png', dpi=300)
    plt.show()


def main():
    sys.path.append('.')

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-p', '--postprocess', type=str)
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-t', '--tag', type=str)
    parser.add_argument('-r', '--read', action='store_true')
    parser.add_argument('-l', '--list', type=str, const='*', nargs='?')
    parser.add_argument('--plt', '--plot', type=str, nargs=2)

    args = parser.parse_args()
    config_file = args.config
    post_model = args.postprocess
    model_name = args.model
    train_tag = args.tag
    read_tag = args.read
    list_flag = args.list
    plt_flag = args.plt

    if config_file is None:
        config_file = 'training.json'

    if not os.path.isfile(config_file):
        print(f'Configuration file {config_file} is not found.')
        exit(-1)

    with open(config_file) as file:
        config = json.load(file)

    if read_tag:
        if parse_config(config):
            print('Exitting due to above errors.')
            exit(-1)
        print('Config file has no errors.')
        exit(0)

    if parse_config(config):
        print('Exitting due to above errors.')
        exit(-1)

    if post_model:
        postprocess(post_model, config)
        exit(0)
    elif list_flag:
        if list_flag == '*':
            print('Listing all models:')
            for i, model in enumerate(config['models']):
                print(f'\t{i}: {model["name"]}')
        else:
            print(f'Listing models with the tag {list_flag}:')
            for i, model in enumerate(config['models']):
                if model['tag'] == list_flag:
                    print(f'\t{i}: {model["name"]}')
        exit(0)
    elif plt_flag:
        if model_name:
            plot_losses([find_model(model_name, config['models'])], plt_flag)
        elif train_tag:
            models = []
            for model in config['models']:
                if model['tag'] == train_tag:
                    models.append(model)
            plot_losses(models, plt_flag)
        exit(0)

    models = config['models']

    if model_name:
        model = find_model(model_name, models)
        if model is None:
            print(f'No model found with the name {model_name}.')
            exit(-1)
        train_model(model)
    elif train_tag:
        for model in models:
            if model.get('tag') == train_tag:
                torch.cuda.empty_cache()
                train_model(model)
    else:
        for model in models:
            torch.cuda.empty_cache()
            train_model(model)
