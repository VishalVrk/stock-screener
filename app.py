from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import pandas as pd
import yfinance as yf
import numpy as np
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import threading
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['LOGS_FOLDER'] = 'logs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['LOGS_FOLDER'], exist_ok=True)

# Store logs in memory for the session
session_logs = []

MONTHLY_DEV = 10.0
DAILY_DEV = 5.0
MIN_FIB = 0.382
MAX_FIB = 1.0

def calculate_zigzag(df, deviation_pct=5):
    tmp_df = df.copy()
    tmp_df['Date'] = tmp_df.index
    tmp_df = tmp_df.reset_index(drop=True)
    deviation = deviation_pct / 100.0
    pivots = []
    
    if len(tmp_df) < 5:
        return pd.DataFrame()

    trend = 1 if tmp_df.at[1, 'Close'] > tmp_df.at[0, 'Close'] else -1
    last_pivot_val = tmp_df.at[0, 'Low'] if trend == 1 else tmp_df.at[0, 'High']
    last_pivot_idx = 0
    
    for i in range(1, len(tmp_df)):
        curr_high = tmp_df.at[i, 'High']
        curr_low = tmp_df.at[i, 'Low']
        
        if trend == 1:
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
        else:
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

def validate_fib(low_val, prev_low, prev_high):
    impulse_leg = prev_high - prev_low
    if impulse_leg == 0:
        return False
    
    retracement = prev_high - low_val
    ratio = retracement / impulse_leg
    
    return ratio >= MIN_FIB and ratio < MAX_FIB

def analyze_stock(symbol):
    try:
        log_message = f"📊 Analyzing {symbol}..."
        socketio.emit('log', {
            'message': log_message,
            'level': 'info',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
        socketio.sleep(0)
        
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5y", interval="1d")
        
        if len(hist) < 250:
            log_message = f"⚠️  {symbol}: Insufficient data (< 250 days)"
            socketio.emit('log', {
                'message': log_message,
                'level': 'warning',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
            return None
        
        curr_price = hist['Close'].iloc[-1]
        log_message = f"✓ {symbol}: Current price ₹{curr_price:.2f}"
        socketio.emit('log', {
            'message': log_message,
            'level': 'success',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
        socketio.sleep(0)

        # Monthly Timeframe
        log_message = f"   Analyzing monthly timeframe..."
        socketio.emit('log', {
            'message': log_message,
            'level': 'info',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
        socketio.sleep(0)
        
        hist_monthly = hist.resample('ME').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        })
        m_pivots = calculate_zigzag(hist_monthly, deviation_pct=MONTHLY_DEV)
        
        if len(m_pivots) < 4:
            log_message = f"⚠️  {symbol}: Insufficient monthly pivots"
            socketio.emit('log', {
                'message': log_message,
                'level': 'warning',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
            return None
        
        last_m_pivot = m_pivots.iloc[-1]
        monthly_valid = False
        monthly_phase = ""

        m_highs = m_pivots[m_pivots['Type'] == 'High']
        m_lows = m_pivots[m_pivots['Type'] == 'Low']
        
        if len(m_highs) < 2 or len(m_lows) < 2:
            log_message = f"⚠️  {symbol}: Invalid monthly structure"
            socketio.emit('log', {
                'message': log_message,
                'level': 'warning',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
            return None

        # Phase Logic
        if last_m_pivot['Type'] == 'High':
            prev_high = m_highs.iloc[-2]
            if last_m_pivot['Value'] > prev_high['Value']:
                relevant_low_2 = m_lows[m_lows['Date'] < last_m_pivot['Date']].iloc[-1]
                relevant_low_1 = m_lows[m_lows['Date'] < relevant_low_2['Date']].iloc[-1]
                relevant_high_1 = m_highs[m_highs['Date'] < relevant_low_2['Date']].iloc[-1]
                
                is_hl = relevant_low_2['Value'] > relevant_low_1['Value']
                is_fib = validate_fib(relevant_low_2['Value'], relevant_low_1['Value'], relevant_high_1['Value'])
                
                if is_hl and is_fib:
                    monthly_valid = True
                    monthly_phase = "Phase A: HH Done -> Moving to HL"
                    log_message = f"   ✓ Monthly: {monthly_phase}"
                    socketio.emit('log', {
                        'message': log_message,
                        'level': 'success',
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
                    session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
                    socketio.sleep(0)

        elif last_m_pivot['Type'] == 'Low':
            prev_low = m_lows.iloc[-2]
            if last_m_pivot['Value'] > prev_low['Value']:
                prev_high = m_highs[m_highs['Date'] < last_m_pivot['Date']].iloc[-1]
                prev_prev_low = m_lows[m_lows['Date'] < prev_high['Date']].iloc[-1]
                
                if validate_fib(last_m_pivot['Value'], prev_prev_low['Value'], prev_high['Value']):
                    prev_prev_high = m_highs[m_highs['Date'] < prev_high['Date']].iloc[-1]
                    if prev_high['Value'] > prev_prev_high['Value']:
                        monthly_valid = True
                        monthly_phase = "Phase B: HL Done -> Moving to HH"
                        log_message = f"   ✓ Monthly: {monthly_phase}"
                        socketio.emit('log', {
                            'message': log_message,
                            'level': 'success',
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        })
                        session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
                        socketio.sleep(0)

        if not monthly_valid:
            log_message = f"✗ {symbol}: Failed monthly criteria"
            socketio.emit('log', {
                'message': log_message,
                'level': 'error',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
            return None

        # Daily Timeframe
        log_message = f"   Analyzing daily timeframe..."
        socketio.emit('log', {
            'message': log_message,
            'level': 'info',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
        socketio.sleep(0)
        
        d_pivots = calculate_zigzag(hist, deviation_pct=DAILY_DEV)
        d_highs = d_pivots[d_pivots['Type'] == 'High']
        d_lows = d_pivots[d_pivots['Type'] == 'Low']
        
        if len(d_highs) < 2 or len(d_lows) < 3:
            log_message = f"✗ {symbol}: Failed daily structure"
            socketio.emit('log', {
                'message': log_message,
                'level': 'error',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
            return None
        
        l_last = d_lows.iloc[-1]
        l_prev = d_lows.iloc[-2]
        h_last = d_highs.iloc[-1]
        
        if l_last['Value'] <= l_prev['Value']:
            log_message = f"✗ {symbol}: No higher low series"
            socketio.emit('log', {
                'message': log_message,
                'level': 'error',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
            return None
            
        h_prev = d_highs.iloc[-2]
        if h_last['Value'] <= h_prev['Value']:
            log_message = f"✗ {symbol}: No higher high series"
            socketio.emit('log', {
                'message': log_message,
                'level': 'error',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
            return None

        # Fibonacci Validation
        high_before_last_low = d_highs[d_highs['Date'] < l_last['Date']].iloc[-1]
        low_before_that = d_lows[d_lows['Date'] < high_before_last_low['Date']].iloc[-1]
        
        if not validate_fib(l_last['Value'], low_before_that['Value'], high_before_last_low['Value']):
            log_message = f"✗ {symbol}: Failed Fibonacci validation"
            socketio.emit('log', {
                'message': log_message,
                'level': 'error',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
            return None
            
        high_before_prev_low = d_highs[d_highs['Date'] < l_prev['Date']].iloc[-1]
        low_start = d_lows[d_lows['Date'] < high_before_prev_low['Date']].iloc[-1]
        
        if not validate_fib(l_prev['Value'], low_start['Value'], high_before_prev_low['Value']):
            log_message = f"✗ {symbol}: Failed Fibonacci validation"
            socketio.emit('log', {
                'message': log_message,
                'level': 'error',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
            return None

        range_high = h_last['Value']
        range_low = l_last['Value']
        last_pivot = d_pivots.iloc[-1]
        
        in_range = False
        direction = ""
        
        if last_pivot['Type'] == 'High':
            if curr_price < h_last['Value'] and curr_price > l_last['Value']:
                in_range = True
                direction = "Pullback from HH"
                log_message = f"   ✓ Daily: {direction}"
                socketio.emit('log', {
                    'message': log_message,
                    'level': 'success',
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
                session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
                socketio.sleep(0)
                
        elif last_pivot['Type'] == 'Low':
            if curr_price > l_last['Value'] and curr_price < h_last['Value']:
                in_range = True
                direction = "Formed HL, Rising inside range"
                log_message = f"   ✓ Daily: {direction}"
                socketio.emit('log', {
                    'message': log_message,
                    'level': 'success',
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
                session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
                socketio.sleep(0)

        if not in_range:
            log_message = f"✗ {symbol}: Price outside valid range"
            socketio.emit('log', {
                'message': log_message,
                'level': 'error',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
            return None

        log_message = f"🎯 {symbol}: QUALIFIED!"
        socketio.emit('log', {
            'message': log_message,
            'level': 'qualified',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
        socketio.sleep(0)
        
        return {
            "Symbol": symbol,
            "Price": round(curr_price, 2),
            "1M Structure": monthly_phase,
            "1D Structure": direction,
            "Last HH": round(h_last['Value'], 2),
            "Last HL": round(l_last['Value'], 2)
        }

    except Exception as e:
        log_message = f"✗ {symbol}: Error - {str(e)}"
        socketio.emit('log', {
            'message': log_message,
            'level': 'error',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        input_df = pd.read_csv(filepath)
        
        if 'Symbol' not in input_df.columns:
            return jsonify({'error': 'CSV must contain a "Symbol" column'}), 400
        
        raw_symbols = input_df['Symbol'].tolist()
        symbols = [f"{s}.NS" if ".NS" not in s and ".BO" not in s else s for s in raw_symbols]
        
        return jsonify({
            'success': True,
            'total': len(symbols),
            'symbols': symbols
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('start_processing')
def handle_processing(data):
    global session_logs
    session_logs = []  # Reset logs for new session
    
    symbols = data.get('symbols', [])
    
    log_message = f"🚀 Starting analysis of {len(symbols)} stocks..."
    socketio.emit('log', {
        'message': log_message,
        'level': 'info',
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
    socketio.sleep(0)
    
    log_message = f"📋 Configuration: Monthly Dev={MONTHLY_DEV}%, Daily Dev={DAILY_DEV}%"
    socketio.emit('log', {
        'message': log_message,
        'level': 'info',
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
    socketio.sleep(0)
    
    log_message = "="*60
    socketio.emit('log', {
        'message': log_message,
        'level': 'info',
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
    socketio.sleep(0)
    
    results = []
    for idx, symbol in enumerate(symbols, 1):
        log_message = f"\n[{idx}/{len(symbols)}] Processing {symbol}"
        socketio.emit('log', {
            'message': log_message,
            'level': 'info',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
        socketio.sleep(0)
        
        res = analyze_stock(symbol)
        if res:
            results.append(res)
        
        # Emit progress
        socketio.emit('progress', {
            'current': idx,
            'total': len(symbols),
            'percent': int((idx / len(symbols)) * 100)
        })
        socketio.sleep(0)
    
    log_message = "="*60
    socketio.emit('log', {
        'message': log_message,
        'level': 'info',
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
    socketio.sleep(0)
    
    log_message = f"✅ Analysis complete! Found {len(results)} qualified stocks."
    socketio.emit('log', {
        'message': log_message,
        'level': 'success',
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    session_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
    socketio.sleep(0)
    
    if results:
        df_final = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"screener_results_{timestamp}.csv"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        df_final.to_csv(output_path, index=False)
        
        # Save logs to file
        log_filename = f"analysis_log_{timestamp}.txt"
        log_path = os.path.join(app.config['LOGS_FOLDER'], log_filename)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(session_logs))
        
        socketio.emit('complete', {
            'success': True,
            'count': len(results),
            'results': results,
            'filename': output_filename,
            'log_filename': log_filename,
            'timestamp': timestamp
        })
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"analysis_log_{timestamp}.txt"
        log_path = os.path.join(app.config['LOGS_FOLDER'], log_filename)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(session_logs))
            
        socketio.emit('complete', {
            'success': True,
            'count': 0,
            'results': [],
            'message': 'No stocks matched the criteria',
            'log_filename': log_filename,
            'timestamp': timestamp
        })

@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

@app.route('/download_log/<filename>')
def download_log(filename):
    filepath = os.path.join(app.config['LOGS_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    data = request.get_json()
    results = data.get('results', [])
    timestamp = data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Create PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#4338ca'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    # Title
    title = Paragraph("Multi-Timeframe Stock Screener Results", title_style)
    elements.append(title)
    
    # Info
    info_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Total Qualified Stocks: {len(results)}"
    info = Paragraph(info_text, styles['Normal'])
    elements.append(info)
    elements.append(Spacer(1, 20))
    
    if results:
        # Table data
        table_data = [['Symbol', 'Price', 'Monthly Structure', 'Daily Structure', 'Last HH', 'Last HL']]
        for stock in results:
            table_data.append([
                stock['Symbol'],
                f"₹{stock['Price']}",
                stock['1M Structure'],
                stock['1D Structure'],
                f"₹{stock['Last HH']}",
                f"₹{stock['Last HL']}"
            ])
        
        # Create table
        table = Table(table_data, colWidths=[1*inch, 0.8*inch, 1.8*inch, 1.8*inch, 0.8*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4338ca')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')])
        ]))
        
        elements.append(table)
    else:
        no_results = Paragraph("No stocks matched the criteria.", styles['Normal'])
        elements.append(no_results)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'screener_results_{timestamp}.pdf'
    )

@app.route('/export_image', methods=['POST'])
def export_image():
    data = request.get_json()
    results = data.get('results', [])
    timestamp = data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Calculate image dimensions
    row_height = 40
    header_height = 100
    table_header_height = 50
    padding = 40
    width = 1200
    height = header_height + table_header_height + (len(results) * row_height) + padding * 2
    
    # Create image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        header_font = ImageFont.truetype("arial.ttf", 16)
        data_font = ImageFont.truetype("arial.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        data_font = ImageFont.load_default()
    
    # Title
    title = "Multi-Timeframe Stock Screener Results"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((width - title_width) / 2, padding), title, fill='#4338ca', font=title_font)
    
    # Info
    info_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total: {len(results)}"
    draw.text((padding, padding + 50), info_text, fill='#6b7280', font=header_font)
    
    # Table header
    y_offset = header_height + padding
    columns = ['Symbol', 'Price', 'Monthly Structure', 'Daily Structure', 'HH', 'HL']
    col_widths = [150, 100, 300, 300, 100, 100]
    
    # Draw header background
    draw.rectangle([(padding, y_offset), (width - padding, y_offset + table_header_height)], fill='#4338ca')
    
    x_offset = padding + 10
    for i, col in enumerate(columns):
        draw.text((x_offset, y_offset + 15), col, fill='white', font=header_font)
        x_offset += col_widths[i]
    
    y_offset += table_header_height
    
    # Draw data rows
    for idx, stock in enumerate(results):
        bg_color = '#f3f4f6' if idx % 2 == 0 else 'white'
        draw.rectangle([(padding, y_offset), (width - padding, y_offset + row_height)], fill=bg_color)
        
        row_data = [
            stock['Symbol'],
            f"₹{stock['Price']}",
            stock['1M Structure'][:30] + '...' if len(stock['1M Structure']) > 30 else stock['1M Structure'],
            stock['1D Structure'][:30] + '...' if len(stock['1D Structure']) > 30 else stock['1D Structure'],
            f"₹{stock['Last HH']}",
            f"₹{stock['Last HL']}"
        ]
        
        x_offset = padding + 10
        for i, value in enumerate(row_data):
            draw.text((x_offset, y_offset + 12), str(value), fill='#1f2937', font=data_font)
            x_offset += col_widths[i]
        
        y_offset += row_height
    
    # Save to buffer
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='image/png',
        as_attachment=True,
        download_name=f'screener_results_{timestamp}.png'
    )

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0",debug=True, port=8888, allow_unsafe_werkzeug=True)