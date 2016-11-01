--
-- Stochastic Depth : https://arxiv.org/abs/1603.09382
-- Uses RReLU instead of ReLU for better reqularization effect
-- Author: Eren Golge -  erengolge@gmail.com
--

require 'nn'
require 'cudnn'
require 'cunn'
require 'torch'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = nn.RReLU -- cudnn.ReLU
local Max = cudnn.SpatialMaxPooling
local SBatchNorm = cudnn.SpatialBatchNormalization

--
-- New Class for Creating Stochastic Depth Network ---
--
local ResidualDrop, parent = torch.class('nn.ResidualDrop', 'nn.Container')

function ResidualDrop:__init(deathRate, nChannels, nOutChannels, stride)
    parent.__init(self)
    self.gradInput = torch.Tensor()
    self.gate = true
    self.train = true
    self.deathRate = deathRate
    nOutChannels = nOutChannels or nChannels
    stride = stride or 1

    self.net = nn.Sequential()
    self.net:add(Convolution(nChannels, nOutChannels, 3,3, stride,stride, 1,1))
    self.net:add(SBatchNorm(nOutChannels))
    self.net:add(ReLU(3,8))
    self.net:add(nn.Dropout(0.05))
    self.net:add(Convolution(nOutChannels, nOutChannels,3,3, 1,1, 1,1))
    self.net:add(SBatchNorm(nOutChannels))
    self.skip = nn.Sequential()
    self.skip:add(nn.Identity())
    if stride > 1 then
       -- optional downsampling
       self.skip:add(Avg(1, 1, stride,stride))
    end
    if nOutChannels > nChannels then
       -- optional padding, this is option A in their paper
       self.skip:add(nn.Padding(1, (nOutChannels - nChannels), 3))
    elseif nOutChannels < nChannels then
       print('Do not do this! nOutChannels < nChannels!')
    end

    self.modules = {self.net, self.skip}
end

function ResidualDrop:updateOutput(input)
    -- if torch.rand(1)[1] < self.deathRate then self.gate = false end
    local skip_forward = self.skip:forward(input)
    self.output:resizeAs(skip_forward):copy(skip_forward)
    if self.train then
      if self.gate then -- only compute convolutional output when gate is open
        self.output:add(self.net:forward(input))
      end
    else
      self.output:add(self.net:forward(input):mul(1-self.deathRate))
    end
    return self.output
end

function ResidualDrop:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):copy(self.skip:updateGradInput(input, gradOutput))
   if self.gate then
      self.gradInput:add(self.net:updateGradInput(input, gradOutput))
   end
   return self.gradInput
end

function ResidualDrop:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if self.gate then
      self.net:accGradParameters(input, gradOutput, scale)
   end
end

-- END of class

--
-- Adds a residual block to the passed in model
--
function addResidualDrop(model, deathRate, nChannels, nOutChannels, stride)
   model:add(nn.ResidualDrop(deathRate, nChannels, nOutChannels, stride))
   model:add(cudnn.ReLU(3,8))
   return model
end

local function layer(model, deathRate, nChannels, nOutChannels, count, stride)
   if stride > 1 then
     addResidualDrop(model, deathRate, nChannels, nOutChannels, 2)
     for i=1,(count-1) do
       addResidualDrop(model, deathRate, nOutChannels, nOutChannels, 1)
     end
   else
     for i=1, count do
       addResidualDrop(model, deathRate, nOutChannels, nOutChannels, 1)
     end
   end
end

local function createModel(opt)
  -- Saves 40% time according to http://torch.ch/blog/2016/02/04/resnets.html
  local cfg = {
     [18]  = {{2, 2, 2, 2}, 512},
     [34]  = {{3, 4, 6, 3}, 512},
     [50]  = {{3, 4, 6, 3}, 2048},
     [101] = {{3, 4, 23, 3}, 2048},
     [152] = {{3, 8, 36, 3}, 2048},
  }
  local depth = opt.depth
  assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
  local def, nFeatures = table.unpack(cfg[depth])

  iChannels = 64
  print(' | ResNet-' .. depth .. ' ImageNet')

  ---- Buidling the residual network model ----
  -- Input: 3x32x32
  print('Building model...')
  model = nn.Sequential()
  ------> 3, 32,32
  model:add(Convolution(3,64,7,7,2,2,3,3)) -- 64
  model:add(SBatchNorm(64))
  model:add(ReLU(3,8))
  model:add(Max(3,3,2,2,1,1)) -- 32
  layer(model, nil, 64, 64, def[1], 1) -- 32
  layer(model, nil, 64, 128, def[2], 2) -- 16
  layer(model, nil, 128, 256, def[3], 2) -- 8
  layer(model, nil, 256, 512, def[4], 2) -- 4
  model:add(Avg(7,7,1,1)):add(nn.Reshape(512))
  model:add(nn.Linear(nFeatures, 1000))

  --> Init layers
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
  ConvInit('nn.SpatialConvolution')
  BNInit('fbnn.SpatialBatchNormalization')
  BNInit('SBatchNorm')
  BNInit('nn.SpatialBatchNormalization')
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
