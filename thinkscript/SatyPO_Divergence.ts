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
input pivLeft      = 3;     # bars to the LEFT of an oscillator pivot
input pivRight     = 1;     # confirmation lag — bars to the RIGHT (default 1)
input rngMin       = 5;     # minimum bars between two consecutive pivot pairs
input rngMax       = 60;    # maximum bars between two consecutive pivot pairs
input xLevel       = 100;   # |SPO| threshold that marks the extreme zone
input showHidden   = yes;   # include hidden (trend-continuation) divergences
input showExtOnly  = yes;   # when yes: hide non-extreme divergences (less noise)

# ── Saty Phase Oscillator ──────────────────────────────────────────────────────
#  pivot   = 21-EMA of close
#  iatr    = 14-period Wilder ATR  (continuous — never resets per day)
#  rawSPO  = ((close - pivot) / (3 × iatr)) × 100
#  spo     = 3-EMA of rawSPO

def pivEMA  = ExpAverage(close, emaSpan);
def trueRng = TrueRange(high, close, low);
def iatr    = WildersAverage(trueRng, atrLen);
def rawSPO  = if iatr != 0 then ((close - pivEMA) / (3 * iatr)) * 100 else 0;
def spo     = ExpAverage(rawSPO, smoothSpan);

# ── Zone reference lines ───────────────────────────────────────────────────────
plot ZeroLine  = 0;
plot XUp       = xLevel;
plot XDown     = -xLevel;
plot DistUp    = 61.8;
plot DistDown  = -61.8;
plot AccUp     = 23.6;
plot AccDown   = -23.6;

ZeroLine.SetDefaultColor(Color.DARK_GRAY);     ZeroLine.SetLineWeight(1);
XUp.SetDefaultColor(Color.RED);                XUp.SetStyle(Curve.SHORT_DASH);   XUp.SetLineWeight(1);
XDown.SetDefaultColor(Color.GREEN);            XDown.SetStyle(Curve.SHORT_DASH); XDown.SetLineWeight(1);
DistUp.SetDefaultColor(CreateColor(200,100,0));   DistUp.SetStyle(Curve.SHORT_DASH);
DistDown.SetDefaultColor(CreateColor(0,180,70));  DistDown.SetStyle(Curve.SHORT_DASH);
AccUp.SetDefaultColor(Color.DARK_GRAY);        AccUp.SetStyle(Curve.SHORT_DASH);
AccDown.SetDefaultColor(Color.DARK_GRAY);      AccDown.SetStyle(Curve.SHORT_DASH);

# ── SPO line — coloured by zone ───────────────────────────────────────────────
plot SPOLine = spo;
SPOLine.SetLineWeight(2);
SPOLine.AssignValueColor(
    if      spo >=  xLevel then Color.RED
    else if spo >=  61.8   then CreateColor(255,140,0)
    else if spo >=  23.6   then Color.YELLOW
    else if spo >  -23.6   then Color.LIGHT_GRAY
    else if spo >  -61.8   then Color.CYAN
    else if spo >  -xLevel then Color.GREEN
    else                        CreateColor(50,230,50)
);

# ── Oscillator pivot detection ────────────────────────────────────────────────
#
# Confirmation bar = current bar.  Pivot bar = pivRight bars ago.
# Left bars run from (pivRight+1) to (pivRight+pivLeft) bars ago.
# Right bar = (pivRight-1) bars ago  (= current bar when pivRight=1).
#
# NOTE: each fold must use unique iterator AND accumulator names —
# ThinkScript treats fold variables as global, not block-scoped.

def llCount = fold llI = 1 to pivLeft + 1 with llN = 0 do
    llN + (if GetValue(spo, pivRight + llI) > spo[pivRight] then 1 else 0);

def lhCount = fold lhI = 1 to pivLeft + 1 with lhN = 0 do
    lhN + (if GetValue(spo, pivRight + lhI) < spo[pivRight] then 1 else 0);

# oscLow / oscHigh avoid collision with ThinkScript built-ins PivotLow / PivotHigh
def oscLow  = llCount == pivLeft and GetValue(spo, pivRight - 1) > spo[pivRight];
def oscHigh = lhCount == pivLeft and GetValue(spo, pivRight - 1) < spo[pivRight];

# ── Previous pivot search — unique iterator/accumulator per fold ──────────────
# Scan from rngMin outward; stop at first previous pivot found.
# Result = bars back to the previous confirmation bar  (0 = none found).

def prevLowGap = fold plJ = rngMin to rngMax + 1 with plG = 0 do
    if plG == 0 and GetValue(oscLow, plJ) then plJ else plG;

def prevHiGap  = fold phJ = rngMin to rngMax + 1 with phG = 0 do
    if phG == 0 and GetValue(oscHigh, phJ) then phJ else phG;

# ── Values at current and previous pivot bars ─────────────────────────────────
# Pivot bar offset from current bar = pivRight (for current) or prevGap+pivRight (for previous).

def curOsc    = spo[pivRight];
def curPLow   = low[pivRight];
def curPHigh  = high[pivRight];

def prevOscL  = GetValue(spo,  prevLowGap + pivRight);
def prevPLow  = GetValue(low,  prevLowGap + pivRight);
def prevOscH  = GetValue(spo,  prevHiGap  + pivRight);
def prevPHigh = GetValue(high, prevHiGap  + pivRight);

# ── Four divergence types ─────────────────────────────────────────────────────

# Regular Bull  — price lower low  + osc higher low  → reversal up
def bullDiv = oscLow
    and prevLowGap > 0
    and curPLow  < prevPLow
    and curOsc   > prevOscL;

# Hidden Bull   — price higher low + osc lower low   → continuation up
def hidBullDiv = oscLow
    and prevLowGap > 0
    and showHidden
    and curPLow  > prevPLow
    and curOsc   < prevOscL;

# Regular Bear  — price higher high + osc lower high → reversal down
def bearDiv = oscHigh
    and prevHiGap > 0
    and curPHigh > prevPHigh
    and curOsc   < prevOscH;

# Hidden Bear   — price lower high + osc higher high → continuation down
def hidBearDiv = oscHigh
    and prevHiGap > 0
    and showHidden
    and curPHigh < prevPHigh
    and curOsc   > prevOscH;

# ── Extreme-zone flag (|SPO at pivot| > xLevel) ───────────────────────────────
def extFlag = AbsValue(curOsc) > xLevel;

def xBull    = bullDiv    and extFlag;
def xHidBull = hidBullDiv and extFlag;
def xBear    = bearDiv    and extFlag;
def xHidBear = hidBearDiv and extFlag;

# ── Signal arrows ─────────────────────────────────────────────────────────────

# Regular Bull (green arrow up)
plot pBullNorm = if !showExtOnly and bullDiv and !extFlag then spo else Double.NaN;
pBullNorm.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
pBullNorm.SetDefaultColor(CreateColor(0,140,60));
pBullNorm.SetLineWeight(2);

plot pBullExt = if xBull then spo else Double.NaN;
pBullExt.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
pBullExt.SetDefaultColor(Color.GREEN);
pBullExt.SetLineWeight(4);

# Hidden Bull (cyan arrow up)
plot pHidBullNorm = if !showExtOnly and hidBullDiv and !extFlag then spo else Double.NaN;
pHidBullNorm.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
pHidBullNorm.SetDefaultColor(CreateColor(0,100,120));
pHidBullNorm.SetLineWeight(2);

plot pHidBullExt = if xHidBull then spo else Double.NaN;
pHidBullExt.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
pHidBullExt.SetDefaultColor(Color.CYAN);
pHidBullExt.SetLineWeight(4);

# Regular Bear (red arrow down)
plot pBearNorm = if !showExtOnly and bearDiv and !extFlag then spo else Double.NaN;
pBearNorm.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
pBearNorm.SetDefaultColor(CreateColor(160,40,40));
pBearNorm.SetLineWeight(2);

plot pBearExt = if xBear then spo else Double.NaN;
pBearExt.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
pBearExt.SetDefaultColor(Color.RED);
pBearExt.SetLineWeight(4);

# Hidden Bear (orange arrow down)
plot pHidBearNorm = if !showExtOnly and hidBearDiv and !extFlag then spo else Double.NaN;
pHidBearNorm.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
pHidBearNorm.SetDefaultColor(CreateColor(200,90,0));
pHidBearNorm.SetLineWeight(2);

plot pHidBearExt = if xHidBear then spo else Double.NaN;
pHidBearExt.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
pHidBearExt.SetDefaultColor(CreateColor(255,165,0));
pHidBearExt.SetLineWeight(4);

# ── Text bubbles — extreme signals only ───────────────────────────────────────
# up = no  → bubble below the value (bullish)
# up = yes → bubble above the value (bearish)

AddChartBubble(xBull,    spo, "Extreme", Color.GREEN,             no);
AddChartBubble(xHidBull, spo, "Extreme", Color.CYAN,              no);
AddChartBubble(xBear,    spo, "Extreme", Color.RED,               yes);
AddChartBubble(xHidBear, spo, "Extreme", CreateColor(255,165,0),  yes);

# ── Alerts ─────────────────────────────────────────────────────────────────────
Alert(xBull,    "SPO Extreme Regular Bull Div",  Alert.BAR, Sound.Ding);
Alert(xBear,    "SPO Extreme Regular Bear Div",  Alert.BAR, Sound.Ding);
Alert(xHidBull, "SPO Extreme Hidden Bull Div",   Alert.BAR, Sound.Ding);
Alert(xHidBear, "SPO Extreme Hidden Bear Div",   Alert.BAR, Sound.Ding);
Alert(bullDiv    and !extFlag, "SPO Regular Bull Div",  Alert.BAR, Sound.Bell);
Alert(bearDiv    and !extFlag, "SPO Regular Bear Div",  Alert.BAR, Sound.Bell);
Alert(hidBullDiv and !extFlag, "SPO Hidden Bull Div",   Alert.BAR, Sound.Bell);
Alert(hidBearDiv and !extFlag, "SPO Hidden Bear Div",   Alert.BAR, Sound.Bell);
