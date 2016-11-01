require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
local t = require 'datasets/transforms'

local file = io.open("models/synset.txt", "r");
local labels = {}
for line in file:lines() do
   table.insert(labels, line);
end

local model = torch.load('pretrained/resnet-101.t7')

local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

local img = image.load('img.jpg', 3, 'float')
 -- Scale, normalize, and crop the image
img = transform(img)
-- -- View as mini-batch of size 1
img = img:view(1, table.unpack(img:size():totable()))
-- -- Get the output of the layer before the (removed) fully connected layer

local output = model:forward(img:cuda())
local _ , predictions = output:float():sort(2, true) -- descending

top5 = predictions:narrow(2, 1, 5):squeeze(1)
print(labels[top5[1]])
print(labels[top5[2]])
print(labels[top5[3]])
print(labels[top5[4]])
print(labels[top5[5]])
