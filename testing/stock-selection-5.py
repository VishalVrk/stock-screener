import pandas as pd
import yfinance as yf
import numpy as np
import os
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "stock-screener.csv"
OUTPUT_FILE = "Final_Candlestick_Screener.csv"

# ZigZag Settings
MONTHLY_DEV = 10.0  # 10% for Monthly Swings
DAILY_DEV = 5.0     # 5% for Daily Swings

# Fibonacci (Must retrace at least 0.382)
MIN_FIB = 0.382 

# ==========================================
# 1. CANDLESTICK PATTERNS
# ==========================================
def check_candlestick_patterns(df):
    """
    Checks the LAST row of the dataframe for specific bullish patterns.
    Returns: (bool, pattern_name)
    """
    if len(df) < 2: return False, ""
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Pre-calc candle properties
    body = abs(curr['Close'] - curr['Open'])
    range_len = curr['High'] - curr['Low']
    upper_shadow = curr['High'] - max(curr['Close'], curr['Open'])
    lower_shadow = min(curr['Close'], curr['Open']) - curr['Low']
    
    # Avoid division by zero
    if range_len == 0: return False, ""
    if curr['Close'] == 0: return False, ""

    # 1. BULLISH ENGULFING
    # Prev Red, Curr Green. Curr body engulfs Prev body.
    is_engulfing = (prev['Close'] < prev['Open']) and \
                   (curr['Close'] > curr['Open']) and \
                   (curr['Open'] <= prev['Close']) and \
                   (curr['Close'] >= prev['Open'])
                   
    if is_engulfing: return True, "Bullish Engulfing"

    # 2. HAMMER
    # Small body at top, long lower shadow (>2x body), small/no upper shadow
    # Body usually in upper 30% of range (simplification)
    is_hammer = (lower_shadow >= 2 * body) and \
                (upper_shadow <= body) and \
                (body < 0.3 * range_len)
    
    if is_hammer: return True, "Hammer"

    # 3. DRAGONFLY DOJI
    # Open/Close/High are very close. Long lower shadow.
    is_doji_body = body <= (0.01 * curr['Close'])
    is_dragonfly = is_doji_body and \
                   (lower_shadow >= 0.6 * range_len) and \
                   (upper_shadow <= 0.1 * range_len)
                   
    if is_dragonfly: return True, "Dragonfly Doji"

    # 4. LONG LEGGED DOJI + GREEN CANDLE
    # Note: User said "Long Legged Doji followed by Green". 
    # This implies the Doji was YESTERDAY (prev) and TODAY (curr) is Green.
    
    prev_body = abs(prev['Close'] - prev['Open'])
    prev_range = prev['High'] - prev['Low']
    
    is_prev_doji = prev_body <= (0.01 * prev['Close'])
    # Long legged means shadows are significant on both sides, or just total range is decent.
    # We will just check if it's a Doji and Today is Green.
    is_curr_green = curr['Close'] > curr['Open']
    
    if is_prev_doji and is_curr_green:
        return True, "Long Legged Doji + Green"

    return False, ""

# ==========================================
# 2. ZIGZAG & FIBONACCI
# ==========================================
def calculate_zigzag(df, deviation_pct=5):
    """ Returns DataFrame: [Date, Value, Type] """
    tmp_df = df.copy()
    tmp_df['Date'] = tmp_df.index
    tmp_df = tmp_df.reset_index(drop=True)
    deviation = deviation_pct / 100.0
    pivots = []
    
    if len(tmp_df) < 5: return pd.DataFrame()

    trend = 1 if tmp_df.at[1, 'Close'] > tmp_df.at[0, 'Close'] else -1
    last_pivot_val = tmp_df.at[0, 'Low'] if trend == 1 else tmp_df.at[0, 'High']
    last_pivot_idx = 0
    
    for i in range(1, len(tmp_df)):
        curr_high = tmp_df.at[i, 'High']
        curr_low = tmp_df.at[i, 'Low']
        
        if trend == 1: # Uptrend
            if curr_high > last_pivot_val:
                last_pivot_val = curr_high
                last_pivot_idx = i
            elif curr_low < last_pivot_val * (1 - deviation):
                pivots.append({'Date': tmp_df.at[last_pivot_idx, 'Date'], 'Value': last_pivot_val, 'Type': 'High'})
                trend = -1
                last_pivot_val = curr_low
                last_pivot_idx = i
        else: # Downtrend
            if curr_low < last_pivot_val:
                last_pivot_val = curr_low
                last_pivot_idx = i
            elif curr_high > last_pivot_val * (1 + deviation):
                pivots.append({'Date': tmp_df.at[last_pivot_idx, 'Date'], 'Value': last_pivot_val, 'Type': 'Low'})
                trend = 1
                last_pivot_val = curr_high
                last_pivot_idx = i
    
    # Add the last forming pivot (current state) for logic checks
    # This helps identify "Moving towards HL" or "Moving towards HH"
    if trend == 1:
        # Currently rising, forming a High
        pivots.append({'Date': tmp_df.iloc[-1]['Date'], 'Value': tmp_df.iloc[-1]['High'], 'Type': 'Forming High'})
    else:
        # Currently falling, forming a Low
        pivots.append({'Date': tmp_df.iloc[-1]['Date'], 'Value': tmp_df.iloc[-1]['Low'], 'Type': 'Forming Low'})

    return pd.DataFrame(pivots)

def validate_fib(low_val, prev_low, prev_high):
    impulse = prev_high - prev_low
    if impulse <= 0: return False
    retracement = prev_high - low_val
    ratio = retracement / impulse
    # Must retrace at least 0.382 and not exceed 100% (Higher Low)
    return ratio >= MIN_FIB and ratio < 1.0

# ==========================================
# 3. ANALYSIS ENGINE
# ==========================================
def analyze_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5y", interval="1d")
        if len(hist) < 250: return None
        curr_price = hist['Close'].iloc[-1]

        # ---------------------------------------------------------
        # CONDITION 6: CANDLESTICK PATTERN (Do this first to save time)
        # ---------------------------------------------------------
        has_candle, pattern_name = check_candlestick_patterns(hist)
        if not has_candle: return None

        # ---------------------------------------------------------
        # CONDITION 1 & 2: MONTHLY STRUCTURE
        # ---------------------------------------------------------
        hist_monthly = hist.resample('ME').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
        m_pivots = calculate_zigzag(hist_monthly, deviation_pct=MONTHLY_DEV)
        
        # We need completed pivots + 1 forming pivot
        if len(m_pivots) < 3: return None
        
        monthly_ok = False
        
        # Get confirmed pivots (exclude 'Forming')
        confirmed_m = m_pivots[m_pivots['Type'].isin(['High', 'Low'])]
        if len(confirmed_m) < 2: return None

        last_conf_pivot = confirmed_m.iloc[-1]
        
        # Check Condition 1: Formed at least 1 HL (post HH) or 1 HH.
        # Check Condition 2: Phase Check
        
        if last_conf_pivot['Type'] == 'High':
            # Phase: Completed HH, Moving towards HL
            # Requirement: The completed HH must be a Higher High
            prev_highs = confirmed_m[confirmed_m['Type'] == 'High']
            if len(prev_highs) < 2: 
                # Only 1 High exists. Does it qualify "at least 1 HH"? Yes.
                # Is it moving towards HL? Yes.
                monthly_ok = True
            else:
                # Compare with previous High
                if prev_highs.iloc[-1]['Value'] > prev_highs.iloc[-2]['Value']:
                    monthly_ok = True
        
        elif last_conf_pivot['Type'] == 'Low':
            # Phase: Completed HL, Moving towards HH
            # Requirement: The completed HL must be a Higher Low (and Fib validated)
            # AND the high before it must be a Higher High
            
            prev_lows = confirmed_m[confirmed_m['Type'] == 'Low']
            prev_highs = confirmed_m[confirmed_m['Type'] == 'High']
            
            if len(prev_lows) >= 2 and len(prev_highs) >= 1:
                curr_low = prev_lows.iloc[-1]['Value']
                prev_low = prev_lows.iloc[-2]['Value']
                high_between = prev_highs.iloc[-1]['Value'] # Approximate check
                
                # Check HL
                if curr_low > prev_low:
                    # Check Fib
                    if validate_fib(curr_low, prev_low, high_between):
                        monthly_ok = True

        if not monthly_ok: return None

        # ---------------------------------------------------------
        # CONDITION 3, 4, 5: DAILY STRUCTURE
        # ---------------------------------------------------------
        d_pivots = calculate_zigzag(hist, deviation_pct=DAILY_DEV)
        d_conf = d_pivots[d_pivots['Type'].isin(['High', 'Low'])]
        
        # Cond 3: Series of HL (at least 1) and HH (at least 1)
        # Sequence must be: ... HL -> HH ... or ... HH -> HL ...
        d_highs = d_conf[d_conf['Type'] == 'High']
        d_lows = d_conf[d_conf['Type'] == 'Low']
        
        if len(d_highs) < 2 or len(d_lows) < 2: return None
        
        # Verify specific "Higher" sequence
        # Last HH > Prev HH?
        last_hh = d_highs.iloc[-1]
        prev_hh = d_highs.iloc[-2]
        if last_hh['Value'] <= prev_hh['Value']: return None # Failed HH check
        
        # Last HL > Prev HL?
        last_hl = d_lows.iloc[-1]
        prev_hl = d_lows.iloc[-2]
        if last_hl['Value'] <= prev_hl['Value']: return None # Failed HL check
        
        # Cond 5: Fib Check on recent HL
        # We need the High that preceded 'last_hl'
        relevant_high = d_highs[d_highs['Date'] < last_hl['Date']]
        if relevant_high.empty: return None
        h_val = relevant_high.iloc[-1]['Value']
        
        # We need the Low that preceded that High
        relevant_low = d_lows[d_lows['Date'] < relevant_high.iloc[-1]['Date']]
        if relevant_low.empty: return None
        l_val = relevant_low.iloc[-1]['Value']
        
        if not validate_fib(last_hl['Value'], l_val, h_val):
            return None

        # Cond 4: Moving Phase & Price Location
        # "Going from HH towards HL OR Forming/Formed HL"
        last_pivot_type = d_conf.iloc[-1]['Type']
        
        daily_ok = False
        
        if last_pivot_type == 'High':
            # We are dropping from HH.
            # Price must be < HH and > HL
            if curr_price < last_hh['Value'] and curr_price > last_hl['Value']:
                daily_ok = True
                
        elif last_pivot_type == 'Low':
            # We just formed a Low. Is it a Higher Low?
            # We already checked 'last_hl' > 'prev_hl' above.
            # So if we are here, we formed a HL and are rising.
            # Price must be < HH (to stay in range per Cond 3 interpretation)
            if curr_price < last_hh['Value']:
                daily_ok = True
        
        if daily_ok:
            return {
                "Symbol": symbol,
                "Price": round(curr_price, 2),
                "Pattern": pattern_name,
                "1M Structure": "Valid Bullish",
                "1D Structure": "Valid HH/HL Series"
            }

    except Exception as e:
        return None
    return None

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
    else:
        print(f"Reading {INPUT_FILE}...")
        input_df = pd.read_csv(INPUT_FILE)
        raw_symbols = input_df['Symbol'].tolist()
        
        # Symbol Cleanup
        symbols = [f"{s}.NS" if ".NS" not in s and ".BO" not in s else s for s in raw_symbols]
        
        print(f"Scanning {len(symbols)} stocks...")
        results = []
        
        for ticker in tqdm(symbols):
            res = analyze_stock(ticker)
            if res:
                results.append(res)
        
        if results:
            df_final = pd.DataFrame(results)
            print("\n" + "="*50)
            print("  STOCKS MATCHING ALL CONDITIONS  ")
            print("="*50)
            print(df_final.to_string(index=False))
            df_final.to_csv(OUTPUT_FILE, index=False)
            print(f"\nSaved to {OUTPUT_FILE}")
        else:
            print("\nNo stocks matched.")