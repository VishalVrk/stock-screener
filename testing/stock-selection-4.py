import pandas as pd
import yfinance as yf
import numpy as np
import os
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "stock-screener.csv"
OUTPUT_FILE = "Advanced_MultiFrame_Screener.csv"

# ZigZag Settings
MONTHLY_DEV = 10.0  # 10% for Monthly Swings
DAILY_DEV = 5.0     # 5% for Daily Swings

# Valid Fibonacci Levels for Retracement
# We check if the retracement depth is at least 0.382
MIN_FIB = 0.382 
MAX_FIB = 1.0   # If it goes beyond 100%, it's not a Higher Low anymore

# ==========================================
# 1. ZIGZAG ALGORITHM
# ==========================================
def calculate_zigzag(df, deviation_pct=5):
    """
    Identifies Swing Highs and Lows.
    Returns DataFrame: [Date, Value, Type]
    """
    tmp_df = df.copy()
    tmp_df['Date'] = tmp_df.index
    tmp_df = tmp_df.reset_index(drop=True)
    deviation = deviation_pct / 100.0
    pivots = []
    
    if len(tmp_df) < 5: return pd.DataFrame()

    # Initial Trend
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
                pivots.append({
                    'Date': tmp_df.at[last_pivot_idx, 'Date'], 
                    'Value': last_pivot_val, 
                    'Type': 'High'
                })
                trend = -1
                last_pivot_val = curr_low
                last_pivot_idx = i
        else: # Downtrend
            if curr_low < last_pivot_val:
                last_pivot_val = curr_low
                last_pivot_idx = i
            elif curr_high > last_pivot_val * (1 + deviation):
                pivots.append({
                    'Date': tmp_df.at[last_pivot_idx, 'Date'], 
                    'Value': last_pivot_val, 
                    'Type': 'Low'
                })
                trend = 1
                last_pivot_val = curr_high
                last_pivot_idx = i

    return pd.DataFrame(pivots)

# ==========================================
# 2. FIBONACCI VALIDATOR
# ==========================================
def validate_fib(low_val, prev_low, prev_high):
    """
    Checks if 'low_val' is a valid higher low relative to the 
    move 'prev_low' -> 'prev_high'.
    """
    impulse_leg = prev_high - prev_low
    if impulse_leg == 0: return False
    
    retracement = prev_high - low_val
    ratio = retracement / impulse_leg
    
    # Check if ratio is >= 0.382 and < 1.0 (Higher Low)
    return ratio >= MIN_FIB and ratio < MAX_FIB

# ==========================================
# 3. ANALYSIS ENGINE
# ==========================================
def analyze_stock(symbol):
    try:
        # Fetch Data
        hist = yf.download(symbol, period="5y", interval="1d", progress=False, threads=False)
        
        if len(hist) < 250: return None
        curr_price = hist['Close'].iloc[-1]

        # ---------------------------------------------------------
        # CONDITION 1 & 2: MONTHLY TIMEFRAME
        # ---------------------------------------------------------
        hist_monthly = hist.resample('ME').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        })
        m_pivots = calculate_zigzag(hist_monthly, deviation_pct=MONTHLY_DEV)
        
        if len(m_pivots) < 4: return None # Need enough history for structure
        
        last_m_pivot = m_pivots.iloc[-1]
        monthly_valid = False
        monthly_phase = ""

        # Separate Highs and Lows
        m_highs = m_pivots[m_pivots['Type'] == 'High']
        m_lows = m_pivots[m_pivots['Type'] == 'Low']
        
        if len(m_highs) < 2 or len(m_lows) < 2: return None

        # --- CHECK 1: Formed at least 1 Higher Low? ---
        # Get the last completed Low and the one before it
        last_completed_low = m_lows.iloc[-1]
        prev_completed_low = m_lows.iloc[-2]
        
        if last_completed_low['Value'] <= prev_completed_low['Value']:
            # The last formed low was NOT a higher low. 
            # However, we might be IN Phase A where the HL is forming but not confirmed.
            # So we check strictly based on the PHASE logic below.
            pass 

        # --- CHECK 2: Phase Logic ---
        
        # PHASE A: Completed HH -> Moving towards HL
        if last_m_pivot['Type'] == 'High':
            # 1. Verify this High is a Higher High
            prev_high = m_highs.iloc[-2]
            if last_m_pivot['Value'] > prev_high['Value']:
                # 2. Verify the Previous Low was a Higher Low (Structure Check)
                # The low *before* this High must be > the low *before that*
                # Sequence: L1 -> H1 -> L2(HL) -> H2(HH) -> Current
                relevant_low_2 = m_lows[m_lows['Date'] < last_m_pivot['Date']].iloc[-1]
                relevant_low_1 = m_lows[m_lows['Date'] < relevant_low_2['Date']].iloc[-1]
                
                # Check Structure & Fib
                relevant_high_1 = m_highs[m_highs['Date'] < relevant_low_2['Date']].iloc[-1]
                
                is_hl = relevant_low_2['Value'] > relevant_low_1['Value']
                is_fib = validate_fib(relevant_low_2['Value'], relevant_low_1['Value'], relevant_high_1['Value'])
                
                if is_hl and is_fib:
                    monthly_valid = True
                    monthly_phase = "Phase A: HH Done -> Moving to HL"

        # PHASE B: Completed HL -> Moving towards HH
        elif last_m_pivot['Type'] == 'Low':
            # 1. Verify this Low is a Higher Low
            prev_low = m_lows.iloc[-2]
            if last_m_pivot['Value'] > prev_low['Value']:
                # 2. Verify Fib for this completed Low
                # Find the high before this low
                prev_high = m_highs[m_highs['Date'] < last_m_pivot['Date']].iloc[-1]
                prev_prev_low = m_lows[m_lows['Date'] < prev_high['Date']].iloc[-1]
                
                if validate_fib(last_m_pivot['Value'], prev_prev_low['Value'], prev_high['Value']):
                    # 3. Verify Previous High was Higher High
                    prev_prev_high = m_highs[m_highs['Date'] < prev_high['Date']].iloc[-1]
                    if prev_high['Value'] > prev_prev_high['Value']:
                        monthly_valid = True
                        monthly_phase = "Phase B: HL Done -> Moving to HH"

        if not monthly_valid: return None

        # ---------------------------------------------------------
        # CONDITION 3, 4, 5: DAILY TIMEFRAME
        # ---------------------------------------------------------
        d_pivots = calculate_zigzag(hist, deviation_pct=DAILY_DEV)
        d_highs = d_pivots[d_pivots['Type'] == 'High']
        d_lows = d_pivots[d_pivots['Type'] == 'Low']
        
        if len(d_highs) < 2 or len(d_lows) < 3: return None
        
        # --- CHECK 3: Series of HL (at least 2) and HH (at least 1) ---
        # Let's look at the last 2 confirmed Lows and last 1 confirmed High
        
        l_last = d_lows.iloc[-1]
        l_prev = d_lows.iloc[-2]
        h_last = d_highs.iloc[-1]
        
        # Check Higher Lows Structure
        if l_last['Value'] <= l_prev['Value']:
            return None # Failed HL Series
            
        # Check Higher High Structure (This High must be > Prev High)
        h_prev = d_highs.iloc[-2]
        if h_last['Value'] <= h_prev['Value']:
            return None # Failed HH Series

        # --- CHECK 5: Fib Confirmation for Daily HLs ---
        # Validate l_last (The most recent confirmed low)
        # Leg: l_prev -> h_last? No, Leg is l_prev -> High_between_lprev_and_llast
        # Proper Sequence: Low_A -> High_B -> Low_C (We validate Low_C against Leg A->B)
        
        # Validate Last Low
        high_before_last_low = d_highs[d_highs['Date'] < l_last['Date']].iloc[-1]
        low_before_that = d_lows[d_lows['Date'] < high_before_last_low['Date']].iloc[-1]
        
        if not validate_fib(l_last['Value'], low_before_that['Value'], high_before_last_low['Value']):
            return None
            
        # Validate Prev Low (l_prev)
        high_before_prev_low = d_highs[d_highs['Date'] < l_prev['Date']].iloc[-1]
        low_start = d_lows[d_lows['Date'] < high_before_prev_low['Date']].iloc[-1]
        
        if not validate_fib(l_prev['Value'], low_start['Value'], high_before_prev_low['Value']):
            return None

        # --- CHECK 4 & 3 (Price Location & Direction) ---
        # "Current price must be between the last higher high and higher low"
        # Identify the RANGE: Last Confirmed HH vs Last Confirmed HL
        
        range_high = h_last['Value']
        range_low = l_last['Value']
        
        # NOTE: ZigZag calculates confirmed points.
        # If the last pivot was a High, we are dropping. Range is High to (forming) Low.
        # If the last pivot was a Low, we are rising. Range is Low to (forming) High.
        
        last_pivot = d_pivots.iloc[-1]
        
        in_range = False
        direction = ""
        
        if last_pivot['Type'] == 'High':
            # We are dropping from High. The "Last HL" is the one before this High.
            # Wait, Condition 3 says "Last Higher High and Higher Low".
            # The most recent COMPLETED HH is h_last.
            # The most recent COMPLETED HL is l_last.
            # Usually in an uptrend, HL happens, then HH.
            
            # Scenario: HL -> HH (Last Pivot) -> Price dropping.
            # Range: Price is < HH. Price should be > HL.
            if curr_price < h_last['Value'] and curr_price > l_last['Value']:
                in_range = True
                direction = "Pullback from HH"
                
        elif last_pivot['Type'] == 'Low':
            # Scenario: HH -> HL (Last Pivot) -> Price Rising.
            # Condition 4 says: "Going from HH towards HL OR Forming/Formed HL".
            # If we just formed HL and are rising, does that count? 
            # "Formed a Higher Low" is explicitly allowed in Cond 4.
            # Price Constraint: Between Last HH and Last HL.
            # Here Price > HL. Price must be < HH.
            if curr_price > l_last['Value'] and curr_price < h_last['Value']:
                in_range = True
                direction = "Formed HL, Rising inside range"

        if not in_range: return None

        return {
            "Symbol": symbol,
            "Price": round(curr_price, 2),
            "1M Structure": monthly_phase,
            "1D Structure": direction,
            "Last HH": round(h_last['Value'], 2),
            "Last HL": round(l_last['Value'], 2)
        }

    except Exception as e:
        # print(e)
        return None
    return None

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
    else:
        print("Loading CSV...")
        input_df = pd.read_csv(INPUT_FILE)
        raw_symbols = input_df['Symbol'].tolist()
        
        # Formatting for yfinance
        symbols = [f"{s}.NS" if ".NS" not in s and ".BO" not in s else s for s in raw_symbols]
        
        print(f"Scanning {len(symbols)} stocks with Multi-Timeframe Fib Logic...")
        results = []
        
        for ticker in tqdm(symbols):
            res = analyze_stock(ticker)
            if res:
                results.append(res)
        
        if results:
            df_final = pd.DataFrame(results)
            print("\n" + "="*50)
            print("  QUALIFIED STOCKS  ")
            print("="*50)
            print(df_final.to_string(index=False))
            df_final.to_csv(OUTPUT_FILE, index=False)
            print(f"\nSaved results to {OUTPUT_FILE}")
        else:
            print("\nNo stocks matched the strict conditions.")