--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--
-- Updated to freeze convolutional layers for fine-tunning.
-- Requires my version of nnlr
-- Author: Eren Golge -- erengolge@gmail.com
--

require 'nn'
require 'cunn'
require 'cudnn'
require 'nnlr'
require "nnx"
require "models/dropresnet"
require "utils/NormalizedLinearNoBias"
require "inn"
inn.utils = require 'inn.utils'
local utils = require "utils"

local M = {}

function M.disableFeatureBackprop(features, maxLayer)
  noBackpropModules = nn.Sequential()
  for i = 1,maxLayer do
    noBackpropModules:add(features.modules[1])
    features:remove(1)
  end
  features:insert(nn.NoBackprop(noBackpropModules):cuda(), 1)
  return features
end


function M.setup(opt, checkpoint, classWeights)
    local model
    if checkpoint then
        local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
        assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
        print('=> Resuming model from ' .. modelPath)
        model = torch.load(modelPath):cuda()
    elseif opt.retrain ~= 'none' then
        assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
        print('Loading model from file: ' .. opt.retrain)
        model = torch.load(opt.retrain):cuda()
    else
        print('=> Creating model from file: models/' .. opt.netType .. '.lua')
        model = require('models/' .. opt.netType)(opt)
    end

    -- First remove any DataParallelTable
    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    -- optnet is an general library for reducing memory usage in neural networks
    if opt.optnet then
        print(' => optnet optimization...')
        local optnet = require 'optnet'
        local imsize = 224
        local sampleInput = torch.zeros(4,3,imsize,imsize):cuda()
        optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
    end

    -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
    -- containers override backwards to call backwards recursively on submodules
    if opt.shareGradInput then
        M.shareGradInput(model)
    end

    -- This freezes convolutional and batch normalization layers, prevents them
    -- backprop and update parameters.
    if opt.freeze then

        local checklist = {
            'cudnn.SpatialConvolution',
            'nn.SpatialConvolution',
            'cunn.SpatialConvolution',
            'nn.SpatialBatchNormalization',
            'nn.VolumetricBatchNormalization',
            'nn.BatchNormalization',
            'cudnn.SpatialBatchNormalization',
            'cudnn.VolumetricBatchNormalization',
            'cudnn.BatchNormalization',
        }
        -- reaplace all Linear layers regarding number of classes
        local layersToFreeze = {}
        for i=1, #checklist do
            layers, dummy = model:findModules(checklist[i])
            print(" => Num. of ".. checklist[i] .." layers to freeze: ".. #layers)
            for j=1, #layers do
                layersToFreeze[#layersToFreeze+1] = layers[j]
            end
        end
        print('=> Number of layers to freeze :'.. #layersToFreeze)

        for i=1, #layersToFreeze do
            layersToFreeze[i].accGradParameters = function(i, o ,e) end
            layersToFreeze[i].updateParameters = function(i, o, e) end
            layersToFreeze[i]:learningRate('weight', 0):weightDecay('weight',0):learningRate('bias', 0):weightDecay('bias',0)
        end
    end

    -- For resetting the classifier when fine-tuning on a different Dataset
    if opt.resetClassifier and not checkpoint then
        print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

          local orig = model:get(#model.modules)
          print(torch.type(orig))
          assert(torch.type(orig) == 'nn.Linear',
             'expected last layer to be fully connected')

        local linear

        -- linear = nn.Linear(orig.weight:size(2), opt.nClasses)
        -- linear.bias:zero()
        model:remove(#model.modules)
        -- model:add(linear:cuda())

        print(" => Changes for the new criterion!!!")
        -- for new criterion
        model:add(nn.NormalizedLinearNoBias(orig.weight:size(2), opt.nClasses):cuda())
        model:add(nn.Normalize(2):cuda())
    else
        local orig = model:get(#model.modules)
        assert(orig.weight:size(1) == opt.nClasses)
    end

    -- Set the CUDNN flags
    if opt.cudnn == 'fastest' then
        cudnn.fastest = true
        cudnn.benchmark = true
    elseif opt.cudnn == 'deterministic' then
        -- Use a deterministic convolution implementation
        model:apply(function(m)
            if m.setMode then m:setMode(1, 1, 1) end
        end)
    end

    -- Wrap the model with DataParallelTable, if using more than one GPU
    if opt.nGPU > 1 then
        local gpus = torch.range(1, opt.nGPU):totable()
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark

        local dpt = nn.DataParallelTable(1, true, true)
        :add(model, gpus)
        :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
        end)
        dpt.gradInput = nil

        model = dpt:cuda()
    end

    -- set class weights
    local criterion
    if opt.classWeighting then
        local cachePath = paths.concat(opt.gen, opt.dataset .. '.t7')
        local imageInfo = torch.load(cachePath)
        print(" => Class weighting enabled !!")
        class_weights = torch.Tensor(imageInfo.classWeights)
        -- criterion = nn.CrossEntropyCriterion(class_weights):cuda()
    else
        -- criterion = nn.CrossEntropyCriterion():cuda()
        print(" => ClassSimplexCriterion ")
        criterion = nn.ClassSimplexCriterion(opt.nClasses):cuda()
    end
    return model, criterion
end

function M.shareGradInput(model)
    local function sharingKey(m)
        local key = torch.type(m)
        if m.__shareGradInputKey then
            key = key .. ':' .. m.__shareGradInputKey
        end
        return key
    end

    -- Share gradInput for memory efficient backprop
    local cache = {}
    model:apply(function(m)
        local moduleType = torch.type(m)
        if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
            local key = sharingKey(m)
            if cache[key] == nil then
                cache[key] = torch.CudaStorage(1)
            end
            m.gradInput = torch.CudaTensor(cache[key], 1, 0)
        end
    end)
    for i, m in ipairs(model:findModules('nn.ConcatTable')) do
        if cache[i % 2] == nil then
            cache[i % 2] = torch.CudaStorage(1)
        end
        m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
    end
end

return M
