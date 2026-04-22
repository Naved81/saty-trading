# ═══════════════════════════════════════════════════════════════════════════════
# Saty Phase Oscillator — Divergence Indicator  (ThinkScript / thinkorswim)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Plots the Saty Phase Oscillator and marks every Regular and Hidden
# divergence directly on the oscillator panel.
#
# Matches the Python backtester (saty-trading/backtest/run_spo_divergence.py):
#   SPO   : 21-EMA pivot  |  14-period Wilder ATR  |  3-EMA smoothing
#   Pivots: lbL = 3  |  lbR = 1 (1-bar confirmation lag)
#   Range : 5 – 60 bars between consecutive pivot pairs
#
# Label legend
#   B    Regular Bull divergence   — price lower low,  osc higher low   (reversal up)
#   hB   Hidden Bull divergence    — price higher low,  osc lower low   (continuation up)
#   R    Regular Bear divergence   — price higher high, osc lower high  (reversal down)
#   hR   Hidden Bear divergence    — price lower high,  osc higher high (continuation down)
#   *    suffix → oscillator was in the EXTREME zone (|SPO| > 100) at the pivot
#
# Extreme divergences use bright, solid colours.
# Non-extreme divergences use muted colours (hidden by default via showExtOnly).
#
# HOW TO INSTALL
#   1. Open thinkorswim  →  Studies  →  Edit Studies  →  Create
#   2. Paste this entire script and click OK
#   3. Add to any intraday chart (3-min, 5-min, 10-min match the Python backtester)
# ═══════════════════════════════════════════════════════════════════════════════

declare lower;

# ── Inputs ─────────────────────────────────────────────────────────────────────
input emaSpan      = 21;    # pivot EMA length
input atrLen       = 14;    # ATR length (Wilder smoothing)
input smoothSpan   = 3;     # SPO final smoothing EMA
input pivotLeft    = 3;     # bars to the LEFT of an oscillator pivot
input pivotRight   = 1;     # confirmation lag — bars to the RIGHT (default 1)
input rangeMin     = 5;     # minimum bars between two consecutive pivot pairs
input rangeMax     = 60;    # maximum bars between two consecutive pivot pairs
input extremeLevel = 100;   # |SPO| threshold that marks the extreme zone
input showHidden   = yes;   # include hidden (trend-continuation) divergences
input showExtOnly  = yes;   # when yes: hide non-extreme divergences (less noise)

# ── Saty Phase Oscillator ──────────────────────────────────────────────────────
#  pivot   = 21-EMA of close
#  iatr    = 14-period Wilder ATR  (continuous — never resets per day)
#  raw_spo = ((close - pivot) / (3 × iatr)) × 100
#  spo     = 3-EMA of raw_spo

def pivEMA = ExpAverage(close, emaSpan);
def tr     = TrueRange(high, close, low);
def iatr   = WildersAverage(tr, atrLen);
def rawSPO = if iatr != 0 then ((close - pivEMA) / (3 * iatr)) * 100 else 0;
def spo    = ExpAverage(rawSPO, smoothSpan);

# ── Zone reference lines ───────────────────────────────────────────────────────
plot Zero     = 0;
plot XUp      = extremeLevel;
plot XDown    = -extremeLevel;
plot DistUp   = 61.8;
plot DistDown = -61.8;
plot AccUp    = 23.6;
plot AccDown  = -23.6;

Zero.SetDefaultColor(Color.DARK_GRAY);      Zero.SetLineWeight(1);
XUp.SetDefaultColor(Color.RED);             XUp.SetStyle(Curve.SHORT_DASH);    XUp.SetLineWeight(1);
XDown.SetDefaultColor(Color.GREEN);         XDown.SetStyle(Curve.SHORT_DASH);  XDown.SetLineWeight(1);
DistUp.SetDefaultColor(CreateColor(200, 100, 0));   DistUp.SetStyle(Curve.SHORT_DASH);
DistDown.SetDefaultColor(CreateColor(0, 180, 70));  DistDown.SetStyle(Curve.SHORT_DASH);
AccUp.SetDefaultColor(Color.DARK_GRAY);    AccUp.SetStyle(Curve.SHORT_DASH);
AccDown.SetDefaultColor(Color.DARK_GRAY);  AccDown.SetStyle(Curve.SHORT_DASH);

# ── SPO line — coloured by zone ───────────────────────────────────────────────
plot SPOLine = spo;
SPOLine.SetLineWeight(2);
SPOLine.AssignValueColor(
    if      spo >=  extremeLevel then Color.RED
    else if spo >=  61.8         then CreateColor(255, 140,   0)
    else if spo >=  23.6         then Color.YELLOW
    else if spo >  -23.6         then Color.LIGHT_GRAY
    else if spo >  -61.8         then Color.CYAN
    else if spo >  -extremeLevel then Color.GREEN
    else                              CreateColor(50,  230,  50)
);

# ── Oscillator pivot detection ────────────────────────────────────────────────
#
# At the CONFIRMATION bar (current bar), the actual pivot sits pivotRight bars
# ago.  The left-side bars are at offsets (pivotRight+1) … (pivotRight+pivotLeft)
# from the current bar.  The right confirmation bar is at offset (pivotRight-1).
#
# With defaults pivotLeft=3, pivotRight=1:
#   Pivot bar  : spo[1]
#   Left bars  : spo[2], spo[3], spo[4]   (all must be above/below the pivot)
#   Right bar  : spo[0]                   (must also be above/below the pivot)

def leftLowCount = fold i = 1 to pivotLeft + 1 with n = 0 do
    n + (if GetValue(spo, pivotRight + i) > spo[pivotRight] then 1 else 0);

def leftHighCount = fold i = 1 to pivotLeft + 1 with n = 0 do
    n + (if GetValue(spo, pivotRight + i) < spo[pivotRight] then 1 else 0);

# All pivotLeft bars on the left must satisfy the condition, plus the right bar.
def pivLow  = leftLowCount  == pivotLeft and GetValue(spo, pivotRight - 1) > spo[pivotRight];
def pivHigh = leftHighCount == pivotLeft and GetValue(spo, pivotRight - 1) < spo[pivotRight];

# ── Locate previous qualifying pivot within [rangeMin, rangeMax] bars ─────────
#
# Scan outward from rangeMin.  The fold stops accumulating once the first
# previous pivot is found (g > 0 short-circuits further assignment).
# Result = number of bars back to that previous CONFIRMATION bar.
# Result = 0 means no qualifying previous pivot exists.

def prevLowGap = fold j = rangeMin to rangeMax + 1 with g = 0 do
    if g == 0 and GetValue(pivLow, j) then j else g;

def prevHighGap = fold j = rangeMin to rangeMax + 1 with g = 0 do
    if g == 0 and GetValue(pivHigh, j) then j else g;

# ── Oscillator and price values at the two pivot bars ─────────────────────────
#
# The PIVOT bar for the current confirmation = current bar offset by pivotRight.
# The PIVOT bar for the previous confirmation = (prevGap + pivotRight) bars ago.

def curOsc  = spo[pivotRight];
def curLow  = low[pivotRight];
def curHigh = high[pivotRight];

def prevOscL  = GetValue(spo,  prevLowGap  + pivotRight);
def prevLow_  = GetValue(low,  prevLowGap  + pivotRight);
def prevOscH  = GetValue(spo,  prevHighGap + pivotRight);
def prevHigh_ = GetValue(high, prevHighGap + pivotRight);

# ── Four divergence types ─────────────────────────────────────────────────────

# Regular Bull  — price lower low  + osc higher low  → mean-reversion up
def bullDiv = pivLow
    and prevLowGap > 0
    and curLow  < prevLow_
    and curOsc  > prevOscL;

# Hidden Bull   — price higher low + osc lower low   → trend continuation up
def hidBullDiv = pivLow
    and prevLowGap > 0
    and showHidden
    and curLow  > prevLow_
    and curOsc  < prevOscL;

# Regular Bear  — price higher high + osc lower high → mean-reversion down
def bearDiv = pivHigh
    and prevHighGap > 0
    and curHigh > prevHigh_
    and curOsc  < prevOscH;

# Hidden Bear   — price lower high + osc higher high → trend continuation down
def hidBearDiv = pivHigh
    and prevHighGap > 0
    and showHidden
    and curHigh < prevHigh_
    and curOsc  > prevOscH;

# ── Extreme-zone flag (|SPO at pivot bar| > extremeLevel) ─────────────────────
def extFlagL = AbsValue(curOsc) > extremeLevel;
def extFlagH = AbsValue(curOsc) > extremeLevel;

def xBull    = bullDiv    and extFlagL;
def xHidBull = hidBullDiv and extFlagL;
def xBear    = bearDiv    and extFlagH;
def xHidBear = hidBearDiv and extFlagH;

# ── Signal arrows ─────────────────────────────────────────────────────────────
# Each divergence type gets two plots: one for non-extreme (muted), one for
# extreme (bright).  showExtOnly suppresses the muted plots when active.

# Regular Bull
plot pBullNorm = if !showExtOnly and bullDiv and !extFlagL then spo else Double.NaN;
pBullNorm.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
pBullNorm.SetDefaultColor(CreateColor(0, 140, 60));
pBullNorm.SetLineWeight(2);

plot pBullExt = if xBull then spo else Double.NaN;
pBullExt.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
pBullExt.SetDefaultColor(Color.GREEN);
pBullExt.SetLineWeight(4);

# Hidden Bull
plot pHidBullNorm = if !showExtOnly and hidBullDiv and !extFlagL then spo else Double.NaN;
pHidBullNorm.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
pHidBullNorm.SetDefaultColor(CreateColor(0, 100, 120));
pHidBullNorm.SetLineWeight(2);

plot pHidBullExt = if xHidBull then spo else Double.NaN;
pHidBullExt.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
pHidBullExt.SetDefaultColor(Color.CYAN);
pHidBullExt.SetLineWeight(4);

# Regular Bear
plot pBearNorm = if !showExtOnly and bearDiv and !extFlagH then spo else Double.NaN;
pBearNorm.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
pBearNorm.SetDefaultColor(CreateColor(160, 40, 40));
pBearNorm.SetLineWeight(2);

plot pBearExt = if xBear then spo else Double.NaN;
pBearExt.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
pBearExt.SetDefaultColor(Color.RED);
pBearExt.SetLineWeight(4);

# Hidden Bear
plot pHidBearNorm = if !showExtOnly and hidBearDiv and !extFlagH then spo else Double.NaN;
pHidBearNorm.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
pHidBearNorm.SetDefaultColor(CreateColor(200, 90, 0));
pHidBearNorm.SetLineWeight(2);

plot pHidBearExt = if xHidBear then spo else Double.NaN;
pHidBearExt.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
pHidBearExt.SetDefaultColor(CreateColor(255, 165, 0));
pHidBearExt.SetLineWeight(4);

# ── Text bubbles for extreme divergences only ─────────────────────────────────
# Bubbles appear at the SPO value on the confirmation bar.
# up = no  → bubble drawn BELOW the value (arrow points up)   = bullish
# up = yes → bubble drawn ABOVE the value (arrow points down) = bearish

AddChartBubble(xBull,    spo, "B*",  Color.GREEN,             no);
AddChartBubble(xHidBull, spo, "hB*", Color.CYAN,              no);
AddChartBubble(xBear,    spo, "R*",  Color.RED,               yes);
AddChartBubble(xHidBear, spo, "hR*", CreateColor(255, 165, 0), yes);

# ── Alerts ─────────────────────────────────────────────────────────────────────
Alert(xBull,    "SPO Extreme Regular Bull Div",  Alert.BAR, Sound.Ding);
Alert(xBear,    "SPO Extreme Regular Bear Div",  Alert.BAR, Sound.Ding);
Alert(xHidBull, "SPO Extreme Hidden Bull Div",   Alert.BAR, Sound.Ding);
Alert(xHidBear, "SPO Extreme Hidden Bear Div",   Alert.BAR, Sound.Ding);
Alert(bullDiv    and !extFlagL, "SPO Regular Bull Div",  Alert.BAR, Sound.Bell);
Alert(bearDiv    and !extFlagH, "SPO Regular Bear Div",  Alert.BAR, Sound.Bell);
Alert(hidBullDiv and !extFlagL, "SPO Hidden Bull Div",   Alert.BAR, Sound.Bell);
Alert(hidBearDiv and !extFlagH, "SPO Hidden Bear Div",   Alert.BAR, Sound.Bell);
