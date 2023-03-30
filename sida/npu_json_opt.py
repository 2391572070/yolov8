#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   NPU Json Optimize
"""

import sys

sys.path.insert(0, '.')

import argparse
import os
import shutil
import json


def npu_json_optimize(opt):
    with open(opt.json_path, 'r', encoding='utf-8') as file:
        npu_json = json.load(file)
        Layers = npu_json.get('Layers', None)
        if Layers is None:
            return
        layer_infos = {}
        for layer_name, layer in Layers.items():
            op = layer.get('op', None)
            inputs = layer.get('inputs', [])
            input_infos = []
            for input in inputs:
                input_layer_name = input.lstrip('@').rsplit(':')[0]
                input_infos.append([input_layer_name, input])
                layer_info = layer_infos.get(input_layer_name, None)
                if layer_info is None:
                    layer_infos[input_layer_name] = {'output_infos': [[layer_name, input]]}
                else:
                    _output_infos = layer_info.get('output_infos', None)
                    if _output_infos is None:
                        layer_info['output_infos'] = [[layer_name, input]]
                    else:
                        _output_infos.append([layer_name, input])
            layer_info = layer_infos.get(layer_name, None)
            if layer_info is None:
                layer_infos[layer_name] = {'input_infos': input_infos, 'op': op}
            else:
                _input_infos = layer_info.get('input_infos', None)
                if _input_infos is None:
                    layer_info['input_infos'] = input_infos
                else:
                    _input_infos.extend(input_infos)
                layer_info['op'] = op

            parameters = layer.get('parameters', None)
            if parameters is None:
                continue

            if op == 'image_resize':
                type = parameters.get('type', None)
                if type == 'bilinear':
                    half_pixel = parameters.get('half_pixel', None)
                    if not half_pixel:
                        parameters['half_pixel'] = True
            shape = parameters.get('shape', None)
            if shape is None:
                continue
            shape[0] = 0

        yolov8_opt = args.yolov8_opt
        if yolov8_opt == 1:
            for layer_name, layer_info in layer_infos.items():
                op = layer_info.get('op', None)
                if op is None:
                    continue
                if op == 'reshape':
                    input_infos = layer_info.get('input_infos', {})
                    output_infos = layer_info.get('output_infos', {})
                    if (len(input_infos) == 1) and (len(output_infos) == 1):
                        input_layer_name = input_infos[0][0]
                        output_layer_name = output_infos[0][0]
                        input_layer = Layers[input_layer_name]
                        output_layer = Layers[output_layer_name]
                        layer = Layers[layer_name]
                        if (input_layer.get('op', None) != 'permute') or (output_layer.get('op', None) != 'permute'):
                            continue
                        input_param = input_layer.get('parameters', None)
                        output_param = output_layer.get('parameters', None)
                        layer_param = layer.get('parameters', None)
                        if (input_param is None) or (output_param is None) or (layer_param is None):
                            continue
                        input_perm = input_param.get('perm', None)
                        output_perm = output_param.get('perm', None)
                        layer_shape = layer_param.get('shape', [])
                        if (input_perm == "0 3 1 2") and (output_perm == "0 1 3 2") and (len(layer_shape) == 4):
                            _layer_info0 = layer_infos[output_layer_name]
                            _output_infos0 = _layer_info0['output_infos']
                            more_opt = False
                            if len(_output_infos0) == 1:
                                _output_layer_name1 = _output_infos0[0][0]
                                _layer_info1 = layer_infos[_output_layer_name1]
                                if _layer_info1.get('op', None) == 'softmax':
                                    _output_infos1 = _layer_info1['output_infos']
                                    if len(_output_infos1) == 1:
                                        _output_layer_name2 = _output_infos1[0][0]
                                        _layer_info2 = layer_infos[_output_layer_name2]
                                        if _layer_info2.get('op', None) == 'convolution':
                                            _layer2 = Layers[_output_layer_name2]
                                            _layer2_param = _layer2['parameters']
                                            is_ok = ((_layer2_param["ksize_h"] == 1) and (_layer2_param["ksize_w"] == 1) and
                                                  (_layer2_param["stride_h"] == 1) and (_layer2_param["stride_h"] == 1) and
                                                  (_layer2_param["pad_h"] == 0) and (_layer2_param["pad_h"] == 0))
                                            if is_ok:
                                                dilations = _layer2_param.get("dilation", [])
                                                for dilation in dilations:
                                                    if dilation != 1:
                                                        is_ok = False
                                                        break
                                            if is_ok:
                                                pads = _layer2_param.get("pad", [])
                                                for pad in pads:
                                                    if pad != 0:
                                                        is_ok = False
                                                        break
                                            if is_ok:
                                                _output_infos2 = _layer_info2['output_infos']
                                                if len(_output_infos2) == 1:
                                                    _output_layer_name3 = _output_infos2[0][0]
                                                    _layer_info3 = layer_infos[_output_layer_name3]
                                                    if _layer_info3.get('op', None) == 'permute':
                                                        _layer3 = Layers[_output_layer_name3]
                                                        _layer3_param = _layer3['parameters']
                                                        if _layer3_param.get('perm', None) == '0 2 1 3':
                                                            more_opt = True

                            if more_opt:
                                layer['outputs'] = output_layer['outputs']
                                _output_infos = layer_infos[output_layer_name]['output_infos']
                                for _output_info in _output_infos:
                                    _output_layer_name = _output_info[0]
                                    _output_layer = Layers[_output_layer_name]
                                    _inputs = _output_layer['inputs']
                                    for i, _input in enumerate(_inputs):
                                        _inputs[i] = _input.replace(output_layer_name, layer_name, 1)
                                Layers.pop(output_layer_name)

                                _layer2['outputs'] = _layer3['outputs']
                                _output_infos = layer_infos[_output_layer_name3].get('output_infos', None)
                                if _output_infos is not None:
                                    for _output_info in _output_infos:
                                        _output_layer_name = _output_info[0]
                                        _output_layer = Layers[_output_layer_name]
                                        _inputs = _output_layer['inputs']
                                        for i, _input in enumerate(_inputs):
                                            _inputs[i] = _input.replace(_output_layer_name3, _output_layer_name2, 1)
                                Layers.pop(_output_layer_name3)
                            else:
                                output_param['perm'] = "0 2 1 3"
                            _input_infos = layer_infos[input_layer_name]['input_infos']
                            _inputs = [_input_info[1] for _input_info in _input_infos]
                            layer['inputs'] = _inputs

                            new_shape = [layer_shape[0], layer_shape[3], layer_shape[1], layer_shape[2]]
                            layer_param['shape'] = new_shape

                            Layers.pop(input_layer_name)

    if args.replace:
        org_file_path = os.path.join(os.path.dirname(opt.json_path), 'org.' + os.path.basename(opt.json_path))
        shutil.move(opt.json_path, org_file_path)
        os.system('chmod a+wr {}'.format(org_file_path))
        out_file_path = args.json_path
    else:
        out_file_path = args.out_file_path
        if out_file_path is None:
            out_file_path = os.path.join(os.path.dirname(opt.json_path), 'opt.' + os.path.basename(opt.json_path))
        else:
            dir = os.path.dirname(out_file_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
                os.system('chmod a+wr \"{}\"'.format(dir))
    with open(out_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(npu_json, json_file, indent=4)
    os.system('chmod a+wr \"{}\"'.format(out_file_path))

    return None


# 1x64x20x11
# reshape 0, 4, 16 -1 1x4x16x220 1x4x220x16
# transpose 1x220x4x16
# softmax
# transpose 1x16x4x220
# conv 1x1x4x220
#
# 1x20x11x64
# permute 0 3 1 2  1x64x20x11
# reshape 0, 4, 16 -1 1x4x16x220
# permute 0 3 1 2  1x220x4x16
# softmax
# permute 0 2 1 3 1x4x220x16
# conv 1x1x220x4
#
# 0 -1 4 16
# 1x20x11x64
# permute 0 3 1 2  1x64x20x11
# reshape 0, 4, 16 -1 1x4x16x220
# permute 0 1 3 2 1x4x220x16
# softmax
# conv 1x1x220x4
#
# 1x20x11x64
# reshape 0, -1, 4, 16 1x220x4x16
# softmax
# conv 1x1x220x4


if __name__ == '__main__':
    parser = argparse.ArgumentParser("NPU Json Optimize")
    parser.add_argument('--json_path', type=str, help='model npu json path')
    parser.add_argument('--out_file_path', type=str, default=None, help='output npu json path')
    parser.add_argument("--replace", action='store_true', help="replace org npu json file")
    parser.add_argument('--yolov8_opt', type=int, default=None, help='yolov8 opt mode')
    args = parser.parse_args()

    npu_json_optimize(args)
