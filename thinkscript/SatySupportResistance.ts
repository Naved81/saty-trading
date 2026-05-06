# ═══════════════════════════════════════════════════════════════════════════════
# Saty Support & Resistance — Touch-Count Indicator  (ThinkScript / thinkorswim)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Draws two horizontal lines on the price chart:
#   RED   dashed line → the resistance level above current price that has been
#                        touched the most times within the threshold
#   GREEN dashed line → the support level below current price that has been
#                        touched the most times within the threshold
#
# A "touch" is any bar whose HIGH or LOW came within touchThresholdPct % of the
# candidate level.  Candidate levels are pivot highs (resistance) and pivot lows
# (support) detected within the lookback window.
#
# INPUTS
#   lookbackBars      — how many historical bars to search (default 200)
#   touchThresholdPct — proximity band as % of price; try 0.1, 0.2, or 0.3
#   pivotStrength     — bars on each side needed to confirm a pivot (default 3)
#
# HOW TO INSTALL
#   1. Open thinkorswim  →  Studies  →  Edit Studies  →  Create
#   2. Paste this entire script and click OK
#   3. Add to any price chart; the two lines overlay on the main price panel
#
# NOTE: Each fold uses unique iterator and accumulator names — ThinkScript
# treats fold variables as global within the script, not block-scoped.
# ═══════════════════════════════════════════════════════════════════════════════

declare upper;

# ── Inputs ─────────────────────────────────────────────────────────────────────
input lookbackBars      = 200;   # bars of history to scan for pivots and touches
input touchThresholdPct = 0.20;  # proximity band — % of current price (0.1–0.3 typical)
input pivotStrength     = 3;     # bars on each side that must be lower/higher to confirm pivot

# ── Pivot detection ────────────────────────────────────────────────────────────
#
# isPivotHigh: current high exceeds the highest of the next pivotStrength bars
#              AND is >= the highest of the previous pivotStrength bars.
# isPivotLow:  mirror logic for lows.
#
# GetValue(expr, -n) looks n bars INTO THE FUTURE — valid in non-scanning studies.

def isPivotHigh = high >  GetValue(Highest(high, pivotStrength), -1) and
                  high >= Highest(high[1], pivotStrength);

def isPivotLow  = low  <  GetValue(Lowest(low, pivotStrength), -1) and
                  low  <= Lowest(low[1], pivotStrength);

def pivHi = if isPivotHigh then high else Double.NaN;
def pivLo = if isPivotLow  then low  else Double.NaN;

# Threshold in absolute price units (recalculated each bar)
def thresh = close * touchThresholdPct / 100;

# ── Touch-count helper (inline via nested fold) ────────────────────────────────
#
# For a candidate level L (a pivot at offset i bars ago), the touch count is:
#   number of bars j in [0, lookbackBars) where
#     |high[j] - L| <= thresh  OR  |low[j] - L| <= thresh
#
# The two folds below (maxResTouches / resLine) both compute this inline.
# Identical structure is repeated for support to keep iterator names unique.

# ── Resistance ─────────────────────────────────────────────────────────────────

# Pass 1: find the maximum touch count across all pivot highs at or above close
def maxResTouches = fold riA = 0 to lookbackBars with rMx = 0 do
    if !IsNaN(GetValue(pivHi, riA)) and GetValue(pivHi, riA) >= close then
        Max(rMx,
            fold rjA = 0 to lookbackBars with rCntA = 0 do
                rCntA + (if AbsValue(GetValue(high, rjA) - GetValue(pivHi, riA)) <= thresh or
                            AbsValue(GetValue(low,  rjA) - GetValue(pivHi, riA)) <= thresh
                         then 1 else 0))
    else rMx;

# Pass 2: find the MOST RECENT pivot high whose touch count equals maxResTouches
#         (IsNaN guard on accumulator stops at the first — i.e. most recent — match)
def resLine = fold riB = 0 to lookbackBars with rLv = Double.NaN do
    if IsNaN(rLv) and
       !IsNaN(GetValue(pivHi, riB)) and
       GetValue(pivHi, riB) >= close and
       (fold rjB = 0 to lookbackBars with rCntB = 0 do
            rCntB + (if AbsValue(GetValue(high, rjB) - GetValue(pivHi, riB)) <= thresh or
                        AbsValue(GetValue(low,  rjB) - GetValue(pivHi, riB)) <= thresh
                     then 1 else 0)) >= maxResTouches
    then GetValue(pivHi, riB)
    else rLv;

# ── Support ─────────────────────────────────────────────────────────────────────

# Pass 1: find the maximum touch count across all pivot lows at or below close
def maxSupTouches = fold siA = 0 to lookbackBars with sMx = 0 do
    if !IsNaN(GetValue(pivLo, siA)) and GetValue(pivLo, siA) <= close then
        Max(sMx,
            fold sjA = 0 to lookbackBars with sCntA = 0 do
                sCntA + (if AbsValue(GetValue(high, sjA) - GetValue(pivLo, siA)) <= thresh or
                            AbsValue(GetValue(low,  sjA) - GetValue(pivLo, siA)) <= thresh
                         then 1 else 0))
    else sMx;

# Pass 2: find the MOST RECENT pivot low whose touch count equals maxSupTouches
def supLine = fold siB = 0 to lookbackBars with sLv = Double.NaN do
    if IsNaN(sLv) and
       !IsNaN(GetValue(pivLo, siB)) and
       GetValue(pivLo, siB) <= close and
       (fold sjB = 0 to lookbackBars with sCntB = 0 do
            sCntB + (if AbsValue(GetValue(high, sjB) - GetValue(pivLo, siB)) <= thresh or
                        AbsValue(GetValue(low,  sjB) - GetValue(pivLo, siB)) <= thresh
                     then 1 else 0)) >= maxSupTouches
    then GetValue(pivLo, siB)
    else sLv;

# ── Plots ───────────────────────────────────────────────────────────────────────
plot Resistance = resLine;
Resistance.SetDefaultColor(Color.RED);
Resistance.SetLineWeight(2);
Resistance.SetStyle(Curve.SHORT_DASH);

plot Support = supLine;
Support.SetDefaultColor(Color.GREEN);
Support.SetLineWeight(2);
Support.SetStyle(Curve.SHORT_DASH);

# ── Labels ──────────────────────────────────────────────────────────────────────
AddLabel(yes,
    "R: " + AsText(resLine, NumberFormat.TWO_DECIMAL_PLACES) +
    "  [" + maxResTouches + " touches]",
    Color.RED);

AddLabel(yes,
    "S: " + AsText(supLine, NumberFormat.TWO_DECIMAL_PLACES) +
    "  [" + maxSupTouches + " touches]",
    Color.GREEN);
