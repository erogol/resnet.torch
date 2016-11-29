-- Init a siammese network with given models

require 'nn'
require 'cunn'
require 'cudnn'
require 'nnlr'
require "nnx"
require "inn"
inn.utils = require 'inn.utils'
local utils = require "utils"

local M = {}

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
        optnet.optimizeMemory(model, sampleInput, {inplace = true, mode = 'training'})
    end

    -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
    -- containers override backwards to call backwards recursively on submodules
    if opt.shareGradInput then
        print(' => shareGradInput optimization...')
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
        print(' => Replacing classifier with ' .. opt.nFeatures .. '-way classifier')

        local orig = model:get(#model.modules)
        assert(torch.type(orig) == 'nn.Linear',
        'expected last layer to be fully connected')

        local linear
        linear = nn.Linear(orig.weight:size(2), opt.nFeatures)
        linear.bias:zero()
        model:remove(#model.modules)
        model:add(linear:cuda())
    else
        local orig = model:get(#model.modules)
        assert(orig.weight:size(1) == opt.nFeatures)
    end

    -- Set siammese layer
    --The siamese model
    siamese_encoder = nn.ParallelTable()
    siamese_encoder:add(model:cuda())
    siamese_encoder:add(model:clone('weight','bias', 'gradWeight','gradBias'):cuda()) --clone the encoder and share the weight, bias. Must also share the gradWeight and gradBias

    model_siamese = nn.Sequential()
    -- model_siamese:add(nn.SplitTable(1)) -- split input tensor along the rows (1st dimension) to table for input to ParallelTable
    model_siamese:add(siamese_encoder)
    model_siamese:add(nn.PairwiseDistance(2):cuda()) --L2 pairwise distance

    -- Set the CUDNN flags
    if opt.cudnn == 'fastest' then
        cudnn.fastest = true
        cudnn.benchmark = true
    elseif opt.cudnn == 'deterministic' then
        -- Use a deterministic convolution implementation
        model_siamese:apply(function(m)
            if m.setMode then m:setMode(1, 1, 1) end
        end)
    end

    -- set class weights
    local criterion
    margin = 1
    criterion = nn.HingeEmbeddingCriterion(margin):cuda()
    return model_siamese, criterion
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
