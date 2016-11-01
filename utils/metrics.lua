local M = {}

function M.accuracy_score(output, target, nCrops)
  if nCrops > 1 then
     -- Sum over crops
     output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
        --:exp()
        :sum(2):squeeze(2)
  end

  -- Coputes the top1 and top5 error rate
  local batchSize = output:size(1)

  local _ , predictions = output:float():sort(2, true) -- descending

  -- Find which predictions match the target
  local correct = predictions:eq(
     target:long():view(batchSize, 1):expandAs(output))

  -- Top-1 score
  local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

  -- Top-5 score, if there are at least 5 classes
  local len = math.min(5, correct:size(2))
  local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

  return top1 * 100, top5 * 100
end

return M
