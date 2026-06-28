-- Compact-only figure sizing.
--
-- Reduces image widths to fit the compact page budget WITHOUT altering any
-- content: it rewrites only the `width` attribute of the figures included from
-- the long document. Sizing is differentiated by importance:
--   - recall_per_frame (key quantitative bar chart): kept largest;
--   - side_by_side_examples (qualitative comparison grid): mid;
--   - the two training-convergence curves (least critical): smallest.
--
-- This filter is referenced only by technical_analysis_compact.qmd, so the long
-- document is unaffected.

local WIDTHS = {
  ["recall_per_frame.png"] = "78%",
  ["side_by_side_examples.png"] = "68%",
  ["loss_curves.png"] = "50%",
  ["metrics_curves.png"] = "50%",
}

function Image(img)
  local base = img.src:match("([^/\\]+)$") or img.src
  local w = WIDTHS[base]
  if w ~= nil then
    img.attributes["width"] = w
  end
  return img
end
