--
-- GoogleNet without aux classifiers
-- Author: Eren Golge -  erengolge@gmail.com
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = cudnn.SpatialMaxPooling
local SBatchNorm = cudnn.SpatialBatchNormalization

local function inception(input_size, config)
    local concat = nn.Concat(2)
    if config[1][1] ~= 0 then
        local conv1 = nn.Sequential()
        conv1:add(Convolution(input_size, config[1][1],1,1,1,1))
        conv1:add(SBatchNorm(config[1][1]))
        conv1:add(ReLU(true))
        concat:add(conv1)
    end

    local conv3 = nn.Sequential()
    conv3:add(Convolution(  input_size, config[2][1],1,1,1,1))
    conv3:add(SBatchNorm(config[2][1]))
    conv3:add(ReLU(true))
    conv3:add(Convolution(config[2][1], config[2][2],3,3,1,1,1,1))
    conv3:add(SBatchNorm(config[2][2]))
    conv3:add(ReLU(true))
    concat:add(conv3)

    local conv3xx = nn.Sequential()
    conv3xx:add(Convolution(  input_size, config[3][1],1,1,1,1))
    conv3xx:add(SBatchNorm(config[3][1]))
    conv3xx:add(ReLU(true))
    conv3xx:add(Convolution(config[3][1], config[3][2],3,3,1,1,1,1))
    conv3xx:add(SBatchNorm(config[3][2]))
    conv3xx:add(ReLU(true))
    conv3xx:add(Convolution(config[3][2], config[3][2],3,3,1,1,1,1))
    conv3xx:add(SBatchNorm(config[3][2]))
    conv3xx:add(ReLU(true))
    concat:add(conv3xx)

    local pool = nn.Sequential()
    pool:add(nn.SpatialZeroPadding(1,1,1,1)) -- remove after getting nn R2 into fbcode
    if config[4][1] == 'max' then
        pool:add(Max(3,3,1,1):ceil())
    elseif config[4][1] == 'avg' then
        pool:add(Avg(3,3,1,1):ceil())
    else
        error('Unknown pooling')
    end
    if config[4][2] ~= 0 then
        pool:add(Convolution(input_size, config[4][2],1,1,1,1))
        pool:add(SBatchNorm(config[4][2]))
        pool:add(ReLU(true))
    end
    concat:add(pool)

    return concat
end

function createModel(opt)
    local RATIO = 1
    local model = nn.Sequential()

    -- Reduction Layers
    model:add(Convolution(3,64/RATIO,7,7,2,2,3,3))
    model:add(SBatchNorm(64/RATIO))
    model:add(ReLU(true)) -->

    model:add(Max(3,3,2,2):ceil())
    model:add(Convolution(64/RATIO,64/RATIO,1,1))
    model:add(SBatchNorm(64/RATIO))
    model:add(ReLU(true))

    model:add(Convolution(64/RATIO,192/RATIO,3,3,1,1,1,1))
    model:add(SBatchNorm(192/RATIO))
    model:add(ReLU(true))

    -- model:add(Max(3,3,2,2):ceil())

    -- Inception Layers
    model:add(inception( 192/RATIO, {{ 64/RATIO},{ 64/RATIO, 64/RATIO},{ 64/RATIO, 96/RATIO},{'avg', 32/RATIO}})) -- 3(a)
    model:add(inception( 256/RATIO, {{ 64/RATIO},{ 64/RATIO, 96/RATIO},{ 64/RATIO, 96/RATIO},{'avg', 64/RATIO}})) -- 3(b)
    model:add(inception( 320/RATIO, {{  0},{128/RATIO,160/RATIO},{ 64/RATIO, 96/RATIO},{'max',  0}})) -- 3(c)
    model:add(Convolution(576/RATIO,576/RATIO,2,2,2,2))
    model:add(inception( 576/RATIO, {{224/RATIO},{ 64/RATIO, 96/RATIO},{ 96/RATIO,128/RATIO},{'avg',128/RATIO}})) -- 4(a)
    model:add(inception( 576/RATIO, {{192/RATIO},{ 96/RATIO,128/RATIO},{ 96/RATIO,128/RATIO},{'avg',128/RATIO}})) -- 4(b)
    model:add(inception( 576/RATIO, {{160/RATIO},{128/RATIO,160/RATIO},{128/RATIO,160/RATIO},{'avg', 96/RATIO}})) -- 4(c)
    model:add(inception( 576/RATIO, {{ 96/RATIO},{128/RATIO,192/RATIO},{160/RATIO,192/RATIO},{'avg', 96/RATIO}})) -- 4(d)

    model:add(inception( 576/RATIO, {{  0},{128/RATIO,192/RATIO},{192/RATIO,256/RATIO},{'max',  0}})) -- 4(e)
    model:add(Convolution(1024/RATIO,1024/RATIO,2,2,2,2))
    model:add(inception(1024/RATIO, {{352/RATIO},{192/RATIO,320/RATIO},{160/RATIO,224/RATIO},{'avg',128/RATIO}})) -- 5(a)
    model:add(inception(1024/RATIO, {{352/RATIO},{192/RATIO,320/RATIO},{192/RATIO,224/RATIO},{'max',128/RATIO}})) -- 5(b)
    model:add(Avg(8,8,1,1)) --  set for 128x128 input
    model:add(nn.View(1024/RATIO):setNumInputDims(3))
    model:add(nn.Dropout(0.3))
    model:add(nn.Linear(1024/RATIO,1000))

    local function ConvInit(name)
        for k,v in pairs(model:findModules(name)) do
            local n = v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/n))
            if cudnn.version >= 4000 then
                v.bias = nil
                v.gradBias = nil
            else
                v.bias:zero()
            end
        end
    end
    local function BNInit(name)
        for k,v in pairs(model:findModules(name)) do
            v.weight:fill(1)
            v.bias:zero()
        end
    end

    ConvInit('cudnn.SpatialConvolution')
    ConvInit('Convolution')
    BNInit('fbnn.SpatialBatchNormalization')
    BNInit('cudnn.SpatialBatchNormalization')
    BNInit('SBatchNorm')

    for k,v in pairs(model:findModules('nn.Linear')) do
        v.bias:zero()
    end
    model:cuda()

    if opt.cudnn == 'deterministic' then
        model:apply(function(m)
            if m.setMode then m:setMode(1,1,1) end
        end)
    end

    model:get(1).gradInput = nil

    return model
end

return createModel
