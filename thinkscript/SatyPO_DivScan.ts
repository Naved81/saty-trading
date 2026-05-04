# ═══════════════════════════════════════════════════════════════════════════════
# Saty PO Divergence — Stock Hacker Scanner  (ThinkScript)
# ═══════════════════════════════════════════════════════════════════════════════
#
# HOW TO USE
#   1. Save this as a new study (e.g. "SatyPO_DivScan")
#      Studies → Edit Studies → Create → paste → OK
#   2. Open the Scan tab → Stock Hacker
#   3. Click "Add Study Filter" → select SatyPO_DivScan → set scan is true (= 1)
#   4. Set your aggregation period (3 min / 5 min / 10 min)
#   5. Run scan — returns any symbol with a fresh extreme divergence signal
#
# The script fires on the CURRENT (most recent) bar.
# To catch signals within the last N bars, increase recentBars from 1.
#
# Parameters match the indicator (SatyPO_Divergence) exactly.
# ═══════════════════════════════════════════════════════════════════════════════

input emaSpan    = 21;
input atrLen     = 14;
input smoothSpan = 3;
input pivLeft    = 3;
input pivRight   = 1;
input rngMin     = 5;
input rngMax     = 60;
input xLevel     = 100;    # |SPO| threshold for extreme zone
input showHidden = yes;    # include hidden (continuation) divergences
input scanBull   = yes;    # include bullish signals in scan output
input scanBear   = yes;    # include bearish signals in scan output
input recentBars = 1;      # flag if signal fired within this many bars (1 = current bar only)

# ── Saty Phase Oscillator ──────────────────────────────────────────────────────
def pivEMA  = ExpAverage(close, emaSpan);
def trueRng = TrueRange(high, close, low);
def iatr    = WildersAverage(trueRng, atrLen);
def rawSPO  = if iatr != 0 then ((close - pivEMA) / (3 * iatr)) * 100 else 0;
def spo     = ExpAverage(rawSPO, smoothSpan);

# ── Oscillator pivot detection ─────────────────────────────────────────────────
def llCount = fold llI = 1 to pivLeft + 1 with llN = 0 do
    llN + (if GetValue(spo, pivRight + llI) > spo[pivRight] then 1 else 0);

def lhCount = fold lhI = 1 to pivLeft + 1 with lhN = 0 do
    lhN + (if GetValue(spo, pivRight + lhI) < spo[pivRight] then 1 else 0);

def oscLow  = llCount == pivLeft and GetValue(spo, pivRight - 1) > spo[pivRight];
def oscHigh = lhCount == pivLeft and GetValue(spo, pivRight - 1) < spo[pivRight];

# ── Previous pivot search ──────────────────────────────────────────────────────
def prevLowGap = fold plJ = rngMin to rngMax + 1 with plG = 0 do
    if plG == 0 and GetValue(oscLow, plJ) then plJ else plG;

def prevHiGap  = fold phJ = rngMin to rngMax + 1 with phG = 0 do
    if phG == 0 and GetValue(oscHigh, phJ) then phJ else phG;

# ── Pivot bar values ───────────────────────────────────────────────────────────
def curOsc   = spo[pivRight];
def curPLow  = low[pivRight];
def curPHigh = high[pivRight];

def prevOscL  = GetValue(spo,  prevLowGap + pivRight);
def prevPLow  = GetValue(low,  prevLowGap + pivRight);
def prevOscH  = GetValue(spo,  prevHiGap  + pivRight);
def prevPHigh = GetValue(high, prevHiGap  + pivRight);

# ── Divergence types ───────────────────────────────────────────────────────────
def bullDiv = oscLow
    and prevLowGap > 0
    and curPLow < prevPLow
    and curOsc  > prevOscL;

def hidBullDiv = oscLow
    and prevLowGap > 0
    and showHidden
    and curPLow > prevPLow
    and curOsc  < prevOscL;

def bearDiv = oscHigh
    and prevHiGap > 0
    and curPHigh > prevPHigh
    and curOsc   < prevOscH;

def hidBearDiv = oscHigh
    and prevHiGap > 0
    and showHidden
    and curPHigh < prevPHigh
    and curOsc   > prevOscH;

# ── Extreme filter ─────────────────────────────────────────────────────────────
def extFlag = AbsValue(curOsc) > xLevel;

def xBull    = bullDiv    and extFlag;
def xHidBull = hidBullDiv and extFlag;
def xBear    = bearDiv    and extFlag;
def xHidBear = hidBearDiv and extFlag;

# ── Signal — fires if ANY qualifying divergence occurred within recentBars ─────
def bullSig = if scanBull then xBull or xHidBull else 0;
def bearSig = if scanBear then xBear or xHidBear else 0;
def rawSig  = bullSig or bearSig;

plot scan = Sum(rawSig, recentBars) > 0;
