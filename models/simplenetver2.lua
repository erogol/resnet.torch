-- Testing class weigthing effect
-- 1. More layers better results
-- 2. Max pooling is better for fine-details
-- 3. BN reduces performance for small datasets
-- 4. ELU activation (faster convergence)
-- ?. YuV color space
-- ?. DarknetTiny
-- X. More layers to 4th and 5th modules
-- X. ClassSimplexCriterion -- https://arxiv.org/pdf/1506.08230v4.pdf
-- X. Adam optimizer
-- ?. Dilated convolution
-- X. AVG pooling instead of max (slower convergence and bad loss)
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = nn.ELU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
	local model = nn.Sequential()
	print(' | SimpleNet is created')

	-- The SimpleNet model
	model:add(Convolution(3,32,3,3,1,1,1,1))
	model:add(ReLU())
	model:add(Max(3,3,2,2,1,1))

    model:add(Convolution(32,64,3,3,1,1,1,1))
	model:add(ReLU())
	model:add(Max(3,3,2,2,1,1))

    model:add(Convolution(64,128,3,3,1,1,1,1))
	model:add(ReLU())
	model:add(Max(3,3,2,2,1,1))

    model:add(Convolution(128,256,3,3,1,1,1,1))
	model:add(ReLU())
	model:add(Max(3,3,2,2,1,1))

    model:add(Convolution(256,512,3,3,1,1,1,1))
    model:add(ReLU())
    model:add(Max(3,3,2,2,1,1))

    model:add(Convolution(512,1024,3,3,1,1,1,1))
	model:add(ReLU())
	model:add(Max(7,7,1,1))

	model:add(nn.View(1024):setNumInputDims(3))
	model:add(nn.Linear(1024, 1024))
    model:add(ReLU())
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(1024,1000))

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
	BNInit('cudnn.SpatialBatchNormalization')
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
