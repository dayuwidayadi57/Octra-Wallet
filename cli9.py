import json, base64, hashlib, time, sys, re, random, string, os, shutil, asyncio, aiohttp, threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import nacl.signing
import uuid
import logging

DARK_MODE_COLORS = {
    'r': '\033[0m',
    'b': '\033[38;5;39m',
    'c': '\033[38;5;45m',
    'g': '\033[38;5;82m',
    'y': '\033[38;5;220m',
    'R': '\033[38;5;203m',
    'w': '\033[38;5;255m',
    'B': '\033[1m',
    'bg_main': '\033[48;5;232m',
    'bg_box': '\033[48;5;234m',
    'bg_success': '\033[48;5;22m',
    'bg_error': '\033[48;5;52m',
    'P': '\033[38;5;165m',
    'G': '\033[38;5;46m',
}

LIGHT_MODE_COLORS = {
    'r': '\033[0m',
    'b': '\033[38;5;39m',
    'c': '\033[38;5;45m',
    'g': '\033[38;5;82m',
    'y': '\033[38;5;220m',
    'R': '\033[38;5;203m',
    'w': '\033[38;5;255m',
    'B': '\033[1m',
    'bg_main': '\033[48;5;17m',
    'bg_box': '\033[48;5;18m',
    'bg_success': '\033[48;5;22m',
    'bg_error': '\033[48;5;52m',
    'P': '\033[38;5;165m',
    'G': '\033[38;5;46m',
}

current_theme = "Dark"
c = DARK_MODE_COLORS

priv, addr, rpc = None, None, None
sk, pub = None, None

b58 = re.compile(r"^oct[1-9A-HJ-NP-Za-km-z]{44}$")
μ = 1_000_000

h = []
cb = 0.0
cn = 0
nonce_cache = {}
nonce_lock = asyncio.Lock()

lu, lh = 0, 0
session = None
executor = ThreadPoolExecutor(max_workers=1)
stop_flag = threading.Event()
spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
spinner_idx = 0
ui_needs_redraw = True

contacts = {}
WALLETS_FILE = 'wallet.json'
active_wallet_idx = 0
wallets = []

auto_send_task = None
auto_send_stop_event = threading.Event()
auto_refresh_task = None
AUTO_REFRESH_INTERVAL = 30

watchdog_task = None
watchdog_stop_event = threading.Event()
watchdog_target_address = None
watchdog_duration_seconds = 0
watchdog_start_time = None
WATCHDOG_CHECK_INTERVAL = 5

DEV_DONATION_WALLET = "octDeaDxuiBDDY63zz5ZcU9cf76UjiNSq6CeLCY2VXDa2DD"
DEV_DONATION_PERCENTAGE = 0.001
GAS_FEE_PERCENTAGE = 0.001
TOTAL_FEE_PERCENTAGE = DEV_DONATION_PERCENTAGE + GAS_FEE_PERCENTAGE

last_cursor_pos = None

LOG_FILE = "octra_error.log"

def setup_logging():
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
#            logging.StreamHandler(sys.stderr) # Opsional: tampilkan error di konsol juga
        ]
    )

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.error("Unhandled exception occurred:", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

def sz():
    return shutil.get_terminal_size((80, 25))

def at(x, y, t, cl=''):
    print(f"\033[{y};{x}H{c['bg_main']}{cl}{t}{c['bg_main']}", end='')

def inp(x, y):
    print(f"\033[{y};{x}H{c['bg_box']}{c['B']}{c['w']}", end='', flush=True)
    return input()

async def ainp(x, y):
    global last_cursor_pos
    last_cursor_pos = (x, y)
    cr = sz()
    print(f"\033[{y};{x}H{c['bg_box']}{' ' * (cr[0] - x)}{c['bg_main']}", end='', flush=True)
    print(f"\033[{y};{x}H{c['bg_box']}{c['B']}{c['w']}", end='', flush=True)
    try:
        user_input = await asyncio.get_event_loop().run_in_executor(executor, input)
        last_cursor_pos = None
        return user_input
    except:
        stop_flag.set()
        last_cursor_pos = None
        return ''

def wait():
    cr = sz()
    msg = "Press Enter to continue..."
    main_box_w = cr[0] - 4
    wait_box_x = (cr[0] - main_box_w) // 2
    wait_box_h = 3
    wait_box_y = cr[1] - wait_box_h - 1

    box(wait_box_x, wait_box_y, main_box_w, wait_box_h, "")
    
    msg_len = len(msg)
    x_pos_text = wait_box_x + (main_box_w - msg_len) // 2
    y_pos_text = wait_box_y + 1
    at(x_pos_text, y_pos_text, msg, c['y'] + c['B'])
    
    print(f"\033[{y_pos_text};{x_pos_text + msg_len}H", end='', flush=True)
    input()

    for i in range(wait_box_h):
        at(wait_box_x, wait_box_y + i, " " * main_box_w, c['bg_main'])
    sys.stdout.flush()

async def awaitkey():
    global last_cursor_pos
    cr = sz()
    msg = "Press Enter to continue..."
    main_box_w = cr[0] - 4
    wait_box_x = (cr[0] - main_box_w) // 2
    wait_box_h = 3
    wait_box_y = cr[1] - wait_box_h - 1

    box(wait_box_x, wait_box_y, main_box_w, wait_box_h, "")
    
    msg_len = len(msg)
    x_pos_text = wait_box_x + (main_box_w - msg_len) // 2
    y_pos_text = wait_box_y + 1
    at(x_pos_text, y_pos_text, msg, c['y'] + c['B'])
    
    print(f"\033[{y_pos_text};{x_pos_text + msg_len}H{c['bg_main']}", end='', flush=True)
    try:
        await asyncio.get_event_loop().run_in_executor(executor, input)
    except:
        stop_flag.set()
    
    for i in range(wait_box_h):
        at(wait_box_x, wait_box_y + i, " " * main_box_w, c['bg_main'])
    sys.stdout.flush()

    if last_cursor_pos:
        print(f"\033[{last_cursor_pos[1]};{last_cursor_pos[0]}H", end='', flush=True)
    else:
        status_y = cr[1] - 1
        x_pos_main_box = (cr[0] - (cr[0] - 4)) // 2
        prompt_text = ""
        if auto_send_task and not auto_send_task.done():
            prompt_text = "STOP AUTO-SEND (0): "
        elif watchdog_task and not watchdog_task.done():
            prompt_text = "STOP WATCHDOG (0): "
        else:
            prompt_text = "⚡ READY | Enter Command Number: "
        print(f"\033[{status_y};{x_pos_main_box + 2 + len(prompt_text)}H", end='', flush=True)
    sys.stdout.flush()

def display_notification(message, color_code=c['w'], bg_color=c['bg_box'], is_spinner=False):
    cr = sz()
    explorer_w = cr[0] - 4
    notification_h = 3 
    notification_y = cr[1] - notification_h - 1 
    notification_x = 2 

    for i in range(notification_h):
        at(notification_x, notification_y + i, " " * explorer_w, c['bg_main'])

    box(notification_x, notification_y, explorer_w, notification_h, "📢 Notification")
    at(notification_x + 2, notification_y + 1, " " * (explorer_w - 4), c['bg_box'])
    at(notification_x + 2, notification_y + 1, message, color_code)
    sys.stdout.flush()

display_notification.last_message = ""

def _display_notification_wrapper(message, color_code=c['w'], bg_color=c['bg_box'], is_spinner=False):
    global display_notification, last_cursor_pos
    display_notification.last_message = message

    cr = sz()
    explorer_w = cr[0] - 4
    notification_h = 3
    notification_y = cr[1] - notification_h - 1
    notification_x = 2

    for i in range(notification_h):
        at(notification_x, notification_y + i, " " * explorer_w, c['bg_main'])

    box(notification_x, notification_y, explorer_w, notification_h, "📢 Notification")
    at(notification_x + 2, notification_y + 1, " " * (explorer_w - 4), c['bg_box'])
    at(notification_x + 2, notification_y + 1, message, color_code)
    sys.stdout.flush()

    if last_cursor_pos:
        print(f"\033[{last_cursor_pos[1]};{last_cursor_pos[0]}H", end='', flush=True)
    else:
        status_y = cr[1] - 1
        x_pos_main_box = (cr[0] - (cr[0] - 4)) // 2
        prompt_text = ""
        if auto_send_task and not auto_send_task.done():
            prompt_text = "STOP AUTO-SEND (0): "
        elif watchdog_task and not watchdog_task.done():
            prompt_text = "STOP WATCHDOG (0): "
        else:
            prompt_text = "⚡ READY | Enter Command Number: "
        print(f"\033[{status_y};{x_pos_main_box + 2 + len(prompt_text)}H", end='', flush=True)
    sys.stdout.flush()

async def display_confirmation(message, default_yes=True):
    cr = sz()
    confirm_h = 5
    confirm_y = cr[1] - confirm_h - 1
    confirm_x = 2
    confirm_w = cr[0] - 4

    box(confirm_x, confirm_y, confirm_w, confirm_h, "❓ CONFIRMATION")

    max_msg_len = confirm_w - 4
    message_lines = []
    words = message.split(' ')
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > max_msg_len and current_line:
            message_lines.append(current_line)
            current_line = word
        else:
            current_line += (" " if current_line else "") + word
    if current_line:
        message_lines.append(current_line)

    for i, line in enumerate(message_lines):
        if confirm_y + 1 + i >= confirm_y + confirm_h - 2:
            break
        at(confirm_x + 2, confirm_y + 1 + i, " " * (confirm_w - 4), c['bg_box'])
        at(confirm_x + 2, confirm_y + 1 + i, line, c['y'])

    input_line_y = confirm_y + 1 + len(message_lines)

    if input_line_y >= confirm_y + confirm_h - 1:
        input_line_y = confirm_y + confirm_h - 2

    at(confirm_x + 2, input_line_y, " " * (confirm_w - 4), c['bg_box'])

    prompt = "[Y/N]: "
    if default_yes:
        prompt = "[Y/n]: "

    at(confirm_x + 2, input_line_y, prompt, c['B'] + c['w'])

    user_input = (await ainp(confirm_x + 2 + len(prompt), input_line_y)).strip().lower()

    for i in range(confirm_h):
        at(confirm_x, confirm_y + i, " " * confirm_w, c['bg_main'])
    sys.stdout.flush()
    
    if default_yes:
        return user_input == 'y' or user_input == ''
    else:
        return user_input == 'y'

_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def _base58_encode(v: bytes) -> str:
    if not isinstance(v, bytes):
        raise TypeError("v must be bytes")
    
    orig_len = len(v)
    
    i = 0
    while i < len(v) and v[i] == 0:
        i += 1
    new_len = len(v) - i
    
    num = int.from_bytes(v, 'big')

    encoded = ""
    while num > 0:
        num, remainder = divmod(num, 58)
        encoded = _BASE58_ALPHABET[remainder] + encoded
    
    encoded = ('1' * (orig_len - new_len)) + encoded
    return encoded

def get_octra_address_from_pubkey_bytes(pub_key_bytes: bytes) -> str:
    sha256_pubkey = hashlib.sha256(pub_key_bytes).digest()
    return "oct" + _base58_encode(sha256_pubkey)

def _save_all_wallets():
    try:
        with open(WALLETS_FILE, 'w') as f:
            json.dump({'wallets': wallets, 'active_wallet_idx': active_wallet_idx}, f, indent=2)
        return True
    except Exception as e:
        _display_notification_wrapper(f"ERROR saving wallets: {str(e)}", c['R'], c['bg_error'])
        logging.error(f"Error saving wallets: {e}", exc_info=True)
        return False

def _load_and_activate_wallet(wallet_data: dict):
    global priv, addr, rpc, sk, pub
    try:
        priv_key_b64 = wallet_data.get('priv')
        rpc_node = wallet_data.get('rpc', 'https://octra.network')

        temp_sk = nacl.signing.SigningKey(base64.b64decode(priv_key_b64))
        temp_pub_bytes = temp_sk.verify_key.encode()
        temp_pub_b64 = base64.b64encode(temp_pub_bytes).decode()
        temp_addr_standard = get_octra_address_from_pubkey_bytes(temp_pub_bytes)

        if wallet_data.get('addr') != temp_addr_standard:
            wallet_data['addr'] = temp_addr_standard
            
        priv = priv_key_b64
        addr = temp_addr_standard
        rpc = rpc_node
        sk = temp_sk
        pub = temp_pub_b64
        return True
    except Exception as e:
        _display_notification_wrapper(f"ERROR activating wallet: {str(e)}", c['R'], c['bg_error'])
        logging.error(f"Error activating wallet: {e}", exc_info=True)
        return False

def ld():
    global wallets, active_wallet_idx
    wallets.clear()
    active_wallet_idx = 0

    try:
        if os.path.exists(WALLETS_FILE):
            with open(WALLETS_FILE, 'r') as f:
                data = json.load(f)
                
                if 'priv' in data and 'addr' in data:
                    wallets.append({
                        'priv': data['priv'],
                        'addr': data['addr'], 
                        'rpc': data.get('rpc', 'https://octra.network')
                    })
                    active_wallet_idx = 0
                    _save_all_wallets()
                    _display_notification_wrapper("Old wallet format detected & converted!", c['y'])
                else:
                    wallets = data.get('wallets', [])
                    active_wallet_idx = data.get('active_wallet_idx', 0)
                
                if not wallets:
                    return False
                
                if not (0 <= active_wallet_idx < len(wallets)):
                    active_wallet_idx = 0
                
                if _load_and_activate_wallet(wallets[active_wallet_idx]):
                    _save_all_wallets()
                    return True
        return False
    except Exception as e:
        _display_notification_wrapper(f"ERROR loading wallets from file: {str(e)}", c['R'], c['bg_error'])
        logging.error(f"Error loading wallets from file: {e}", exc_info=True)
        priv, addr, rpc, sk, pub = None, None, None, None, None
        wallets.clear()
        active_wallet_idx = 0
        return False

def load_contacts():
    global contacts
    try:
        if os.path.exists('contacts.json'):
            with open('contacts.json', 'r') as f:
                contacts = json.load(f)
        else:
            contacts = {}
        return True
    except Exception as e:
        contacts = {}
        _display_notification_wrapper(f"ERROR loading contacts: {str(e)}", c['R'], c['bg_error'])
        logging.error(f"Error loading contacts: {e}", exc_info=True)
        return False

def save_contacts():
    try:
        with open('contacts.json', 'w') as f:
            json.dump(contacts, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error saving contacts: {e}", exc_info=True)
        return False

def fill():
    cr = sz()
    print(f"\033[{cr[1]};{1}H", end='') 
    print(f"{c['bg_main']}", end='')
    for _ in range(cr[1]):
        print(" " * cr[0])
    print("\033[H", end='')

def box(x, y, w, h, t=""):
    print(f"\033[{y};{x}H{c['bg_box']}{c['w']}┌{'─' * (w - 2)}┐{c['bg_main']}")
    if t:
        title_len = len(t)
        title_start = x + (w - title_len - 4) // 2
        print(f"\033[{y};{title_start}H{c['bg_box']}{c['w']}┤ {c['B']}{c['c']}{t}{c['w']} ├{c['bg_main']}")
    for i in range(1, h - 1):
        print(f"\033[{y + i};{x}H{c['bg_box']}{c['w']}│{' ' * (w - 2)}│{c['bg_main']}")
    print(f"\033[{y + h - 1};{x}H{c['bg_box']}{c['w']}└{'─' * (w - 2)}┘{c['bg_main']}")

async def spin_animation_in_notification_box(msg, color=c['c']):
    global spinner_idx, last_cursor_pos

    try:
        while True:
            _display_notification_wrapper(f"{spinner_frames[spinner_idx]} {msg}", color, c['bg_box'], is_spinner=True)
            spinner_idx = (spinner_idx + 1) % len(spinner_frames)
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        _display_notification_wrapper("", c['bg_box'])
        sys.stdout.flush()
        if last_cursor_pos:
            print(f"\033[{last_cursor_pos[1]};{last_cursor_pos[0]}H", end='', flush=True)
        else:
            cr = sz()
            status_y = cr[1] - 1
            x_pos_main_box = (cr[0] - (cr[0] - 4)) // 2
            prompt_text = ""
            if auto_send_task and not auto_send_task.done():
                prompt_text = "STOP AUTO-SEND (0): "
            elif watchdog_task and not watchdog_task.done():
                prompt_text = "STOP WATCHDOG (0): "
            else:
                prompt_text = "⚡ READY | Enter Command Number: "
            print(f"\033[{status_y};{x_pos_main_box + 2 + len(prompt_text)}H", end='', flush=True)
        sys.stdout.flush()
    except Exception as e:
        logging.error(f"Error in spin_animation_in_notification_box: {e}", exc_info=True)

async def req(m, p, d=None, t=60, retries=5, delay=5):
    global session
    url = f"{rpc}{p}"

    for attempt in range(1, retries + 1):
        try:
            async with getattr(session, m.lower())(
                url,
                json=d if m.upper() == 'POST' else None,
                timeout=aiohttp.ClientTimeout(total=t)
            ) as resp:

                try:
                    content_type = resp.headers.get('Content-Type', '')
                    if 'charset=' in content_type:
                        charset = content_type.split('charset=')[-1].strip()
                        text = await resp.text(encoding=charset, errors='replace')
                    else:
                        text = await resp.text(encoding='utf-8', errors='replace')
                except Exception as decode_e:
                    raw_bytes = await resp.read()
                    try:
                        text = raw_bytes.decode('utf-8', errors='replace')
                    except:
                        text = raw_bytes.decode('latin-1', errors='replace')
                    logging.error(f"[Decode Error] Failed to decode response from {url} (Attempt {attempt}): {decode_e}", exc_info=True)

                if resp.status >= 400:
                    logging.error(f"[HTTP Error] {resp.status} from {url}: {text[:200]}")
                    return resp.status, f"RPC Error {resp.status}: {text[:200]}", None

                try:
                    return resp.status, text, json.loads(text) if text else None
                except json.JSONDecodeError:
                    logging.warning(f"[JSON Decode] Failed to parse JSON from {url}")
                    return resp.status, text, None

        except asyncio.TimeoutError:
            logging.warning(f"[Timeout] Attempt {attempt}/{retries} timed out after {t}s: {url}")
            if attempt == retries:
                return 0, f"Timeout after {t}s (URL: {url})", None
        except aiohttp.ClientError as e:
            logging.warning(f"[ClientError] Attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                return 0, f"Network Error: {str(e)}", None
        except Exception as e:
            logging.exception(f"[Unexpected Error] in req() (Attempt {attempt}): {e}")
            if attempt == retries:
                return 0, f"Unexpected Error: {type(e).__name__}", None

        await asyncio.sleep(delay)

async def comprehensive_data_refresh():
    global cb, cn, lu, lh, ui_needs_redraw, h

    old_cb, old_cn = cb, cn

    current_transactions_map = {tx['hash']: tx for tx in h}

    spin_task_refresh = None
    if cb is None or cn is None: 
        spin_task_refresh = asyncio.create_task(spin_animation_in_notification_box("Loading account data...", c['y']))

    try:
        current_rpc_nonce, current_rpc_balance = await _fetch_balance_nonce()

        if current_rpc_nonce is not None:
            async with nonce_lock:
                if cn is None or current_rpc_nonce >= cn:
                    cn = current_rpc_nonce
                cb = current_rpc_balance
                nonce_cache[addr] = cn 

    finally:
        if spin_task_refresh and not spin_task_refresh.done():
            spin_task_refresh.cancel()
            try: await spin_task_refresh
            except asyncio.CancelledError: pass
            _display_notification_wrapper("", c['bg_box'])

    s, t, j = await req('GET', f'/address/{addr}?limit=50') 

    transactions_from_rpc = []
    if s == 200 and j and 'recent_transactions' in j:
        tx_hashes = [ref["hash"] for ref in j.get('recent_transactions', [])]

        fetch_tasks = [req('GET', f'/tx/{hash}', t=5) for hash in tx_hashes]

        tx_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for i, (ref, result) in enumerate(zip(j.get('recent_transactions', []), tx_results)):
            if isinstance(result, Exception):
                logging.error(f"Error fetching transaction hash {ref.get('hash')}: {result}", exc_info=True)
                continue
            s2, _, j2 = result
            if s2 == 200 and j2 and 'parsed_tx' in j2:
                try:
                    p = j2['parsed_tx']
                    tx_hash = ref['hash']
                    is_inbound = p.get('to') == addr
                    amount_raw = p.get('amount_raw', p.get('amount', '0'))
                    amount_float = float(amount_raw) if '.' in str(amount_raw) else int(amount_raw) / μ
                    
                    rpc_tx_data = {
                        'time': datetime.fromtimestamp(p.get('timestamp', 0)),
                        'hash': tx_hash,
                        'amt': amount_float,
                        'to': p.get('to') if not is_inbound else p.get('from'),
                        'type': 'in' if is_inbound else 'out',
                        'ok': True,
                        'nonce': p.get('nonce', 0),
                        'epoch': ref.get('epoch', 0),
                        'raw_tx': j2.get('raw_tx'),
                        'parsed_tx': j2.get('parsed_tx')
                    }
                    transactions_from_rpc.append(rpc_tx_data)
                except Exception as parse_e:
                    logging.error(f"Error parsing transaction {tx_hash}: {parse_e}", exc_info=True)
    
    new_h = []
    updated_something = False

    for rpc_tx in transactions_from_rpc:
        existing_tx = current_transactions_map.get(rpc_tx['hash'])
        if existing_tx:
            if existing_tx.get('epoch') == 0 and rpc_tx.get('epoch', 0) > 0:
                existing_tx['epoch'] = rpc_tx['epoch']
                existing_tx['time'] = rpc_tx['time']
                existing_tx['parsed_tx'] = rpc_tx['parsed_tx']
                existing_tx['ok'] = True
                updated_something = True
            new_h.append(existing_tx)
        else:
            new_h.append(rpc_tx)
            updated_something = True
            
    for local_tx in h:
        if local_tx['hash'] not in {tx['hash'] for tx in new_h}:
            new_h.append(local_tx)
            
    new_h[:] = sorted(new_h, key=lambda x: x['time'], reverse=True)[:50]
    
    if s == 404 or (s == 200 and t and 'no transactions' in t.lower()):
        if not new_h:
            new_h.clear() 
    
    h[:] = new_h
    lh = time.time()

    if (cb != old_cb or cn != old_cn or updated_something):
        ui_needs_redraw = True

    
async def _fetch_balance_nonce(retries=5, delay=0.5):
    global cb, cn, lu, nonce_cache

    for attempt in range(retries):
        try:
            results = await asyncio.gather(
                req('GET', f'/balance/{addr}', t=30),
                req('GET', '/staging', t=5),
                return_exceptions=True
            )
            
            s, t, j = results[0] if not isinstance(results[0], Exception) else (0, str(results[0]), None)
            s2, _, j2 = results[1] if not isinstance(results[1], Exception) else (0, None, None)

            if s == 200 and j:
                rpc_chain_nonce = int(j.get('nonce', 0))
                fetched_balance = float(j.get('balance', 0))
                lu = time.time()
                
                max_staged_nonce = 0
                if s2 == 200 and j2 and 'staged_transactions' in j2:
                    our_staged_txs = [tx for tx in j2.get('staged_transactions', []) if tx.get('from') == addr]
                    if our_staged_txs:
                        staged_nonces = [int(tx.get('nonce', 0)) for tx in our_staged_txs]
                        max_staged_nonce = max(staged_nonces)

                current_rpc_effective_nonce = max(rpc_chain_nonce, max_staged_nonce)

                async with nonce_lock:
                    if addr in nonce_cache:
                        current_rpc_effective_nonce = max(current_rpc_effective_nonce, nonce_cache[addr])
                    # Update cache dengan nonce terbaru
                    nonce_cache[addr] = current_rpc_effective_nonce

                return current_rpc_effective_nonce, fetched_balance

            elif s == 404:
                async with nonce_lock:
                    if addr not in nonce_cache:
                        nonce_cache[addr] = 0
                return 0, 0.0
            elif s == 200 and t and not j:
                try:
                    parts = t.strip().split()
                    if len(parts) >= 2:
                        fetched_balance = float(parts[0]) if parts[0].replace('.', '').isdigit() else 0.0
                        chain_nonce_from_text = int(parts[1]) if parts[1].isdigit() else 0
                        
                        lu = time.time()
                        
                        async with nonce_lock:
                            if addr in nonce_cache:
                                chain_nonce_from_text = max(chain_nonce_from_text, nonce_cache[addr])
                            nonce_cache[addr] = chain_nonce_from_text
                        
                        return chain_nonce_from_text, fetched_balance
                    else:
                        logging.error(f"Failed to parse balance/nonce from text format: {t}", exc_info=True)
                        return None, None
                except Exception as parse_e:
                    logging.error(f"Error parsing balance/nonce from text format: {parse_e}", exc_info=True)
                    return None, None
            
            await asyncio.sleep(delay)

        except Exception as e:
            logging.error(f"Network or RPC error during balance/nonce fetch (attempt {attempt+1}): {e}", exc_info=True)
            await asyncio.sleep(delay)

    _display_notification_wrapper(f"ERROR: RPC connection failed to load account data after {retries} attempts.", c['R'], c['bg_error'])
    logging.error(f"RPC connection failed to load account data after {retries} attempts for address {addr}", exc_info=True)
    return None, None

def mk(to, a, n):
    tx = {
        "from": addr,
        "to_": to,
        "amount": str(int(a * μ)),
        "nonce": int(n),
        "ou": "1000" if a < 1000 else "3000",
        "timestamp": time.time() + random.random() * 0.01
    }
    bl = json.dumps(tx, separators=(",", ":"))
    try:
        sig = base64.b64encode(sk.sign(bl.encode()).signature).decode()
    except Exception as e:
        logging.error(f"Error signing transaction: {e}", exc_info=True)
        raise
    tx.update(signature=sig, public_key=pub)
    return tx, hashlib.sha256(bl.encode()).hexdigest()

async def snd(tx):
    t0 = time.time()
    s, t, j = await req('POST', '/send-tx', tx, t=30)
    dt = time.time() - t0
    if s == 200:
        if j and j.get('status') == 'accepted':
            return True, j.get('tx_hash', ''), dt, j
        elif t.lower().startswith('ok'):
            return True, t.split()[-1], dt, None
        else:
            logging.error(f"Transaction rejected by node. Status: {s}, Response: {t}, JSON: {j}", exc_info=True)
            return False, j.get('error', 'Transaction rejected by node.') if j else t, dt, j
    else:
        logging.error(f"Failed to send transaction. Status: {s}, Response: {t}, JSON: {j}", exc_info=True)
        return False, t, dt, j

async def expl(x, y, w, hb):
    box(x, y, w, hb, "◎ WALLET DASHBOARD")

    at(x + 2, y + 2, "🔑 Address :", c['c'])
    at(x + 13, y + 2, addr, c['w'])
    at(x + 2, y + 3, "💰 Balance :", c['c'])
    at(x + 13, y + 3, f"{cb:,.6f} OCT" if cb is not None else "---", c['B'] + c['g'] if cb is not None else c['w'])
    at(x + 2, y + 4, "#️⃣ Nonce :", c['c'])
    at(x + 13, y + 4, str(cn) if cn is not None else "---", c['w'])
    at(x + 2, y + 5, "🌐 Public :", c['c'])
    at(x + 13, y + 5, pub, c['w'])

    _, _, j_staging = await req('GET', '/staging', t=2)
    sc = len([tx for tx in j_staging.get('staged_transactions', []) if tx.get('from') == addr]) if j_staging and 'staged_transactions' in j_staging else 0
    at(x + 2, y + 6, "📤 Staging :", c['c'])
    at(x + 13, y + 6, f"{sc} pending transactions" if sc else "None", c['y'] if sc else c['w'])
    at(x + 1, y + 7, "─" * (w - 2), c['w'])

    at(x + 2, y + 8, "📜 RECENT TRANSACTIONS:", c['B'] + c['c'])
    if not h:
        at(x + 2, y + 10, "No transactions yet.", c['y'])
    else:
        time_col_width = 9
        type_col_width = 6
        amount_col_width = 12
        status_col_width = 20

        dynamic_addr_col_width = max(13, w - 55) 
        
        header_line = f"{'🕒 Time':<{time_col_width}} {'♻️ Type':<{type_col_width}} {'💰 Amount':<{amount_col_width}} {'💸 Destination':<{dynamic_addr_col_width}} {'📊 Status':<{status_col_width}}"
        at(x + 2, y + 10, header_line, c['c'])
        at(x + 2, y + 11, "─" * (len(header_line)), c['w'])

        seen_hashes = set()
        display_count = 0
        sorted_h = sorted(h, key=lambda x: x['time'], reverse=True)
        for tx in sorted_h:
            if tx['hash'] in seen_hashes:
                continue
            seen_hashes.add(tx['hash'])
            if display_count >= min(len(h), hb - 15):
                break
            is_pending = not tx.get('epoch')
            
            status_symbol = "🔵 Pending" if is_pending else f"✅ Confirmed (E{tx.get('epoch', 0)})"
            status_color = c['y'] if is_pending else c['g']
            type_symbol = "OUT"
            type_color = c['R']
            if tx['type'] == 'in':
                type_symbol = "IN"
                type_color = c['g']

            full_address = str(tx.get('to', '---'))
            
            if len(full_address) > dynamic_addr_col_width:
                display_to_short = full_address[:5] + "..." + full_address[-5:]
                if len(display_to_short) > dynamic_addr_col_width:
                    display_to_short = display_to_short[:dynamic_addr_col_width-3] + "..."
            else:
                display_to_short = full_address
            
            display_to_padded = f"{display_to_short:<{dynamic_addr_col_width}}"

            display_status_text = status_symbol
            
            offset_geser_destination_kanan = 3
            destination_print_x = x + 2 + time_col_width + 1 + type_col_width + 1 + amount_col_width + 1 + offset_geser_destination_kanan
           
            status_print_x = destination_print_x + dynamic_addr_col_width + 1
            offset_geser_status_kiri = 6
            status_print_x -= offset_geser_status_kiri 

            at(x + 2, y + 12 + display_count, f"{tx['time'].strftime('%H:%M:%S'):<{time_col_width}}", c['w'])
            at(x + 2 + time_col_width + 1, y + 12 + display_count, f"{type_symbol:<{type_col_width}}", type_color)
            at(x + 2 + time_col_width + 1 + type_col_width + 1, y + 12 + display_count, f"{float(tx['amt']):>{amount_col_width}.6f}", c['w'])
            
            at(destination_print_x, y + 12 + display_count, display_to_padded, c['w'])
           
            at(status_print_x, y + 12 + display_count, f"{display_status_text:<{status_col_width}}", status_color)
            
            display_count += 1
        
        if len(h) > display_count:
            at(x + 2, y + 12 + display_count, f"({len(h) - display_count} more transactions...)", c['y'])

async def show_transaction_details():
    cr = sz()
    cls()
    fill()
    main_box_w = cr[0] - 4
    w, hb = main_box_w, cr[1] - 4
    x = (cr[0] - w) // 2
    y = 2
    box(x, y, w, hb, "📜 TRANSACTION HISTORY & DETAILS")

    if not h:
        _display_notification_wrapper("No transactions found yet.", c['y'])
        await awaitkey()
        ui_needs_redraw = True
        return

    at(x + 2, y + 3, "Recent Transactions (Enter # for details, 0 to go back):", c['B'] + c['c'])
    at(x + 2, y + 4, "─────────────────────────────────────────────────────────────────────────────────", c['w'])
    
    display_limit = min(len(h), hb - 10)
    for i in range(display_limit):
        tx = h[i]
        is_pending = not tx.get('epoch')
        status_symbol = "🔵 Pending" if is_pending else f"✅ Confirmed (E{tx.get('epoch', 0)})"
        status_color = c['y'] if is_pending else c['g']
        type_symbol = "OUT" if tx['type'] == 'out' else "IN"
        type_color = c['R'] if tx['type'] == 'out' else c['g']

        display_to = tx['to']
        if len(display_to) > 25:
            display_to = display_to[:22] + "..."
        
        line = f"[{i+1:>2}] {tx['time'].strftime('%m-%d %H:%M:%S')} | {type_symbol:<3} {tx['amt']:>10.6f} OCT | {display_to:<25} | {status_symbol}"
        at(x + 2, y + 5 + i, line, c['w'])
    
    if len(h) > display_limit:
        at(x + 2, y + 5 + display_limit, f"({len(h) - display_limit} more transactions...)", c['y'])

    at(x + 2, y + 6 + display_limit, "─────────────────────────────────────────────────────────────────────────────────", c['w'])
    at(x + 2, y + 7 + display_limit, "Enter transaction number: ", c['B'] + c['w'])
    
    choice_str = await ainp(x + 28, y + 7 + display_limit)

    if not choice_str.strip():
        _display_notification_wrapper("Empty input. Returning to menu.", c['y'])
        await awaitkey()
        ui_needs_redraw = True
        return

    try:
        choice = int(choice_str)
        if choice == 0:
            ui_needs_redraw = True
            return
        if not (1 <= choice <= len(h)):
            raise ValueError("Invalid transaction number.")
        
        selected_tx = h[choice - 1]
        
        cls()
        fill()
        box(x, y, w, hb, "🔍 TRANSACTION DETAILS")
        
        at(x + 2, y + 3, "Hash:", c['c'])
        at(x + 10, y + 3, selected_tx['hash'], c['w'])
        
        at(x + 2, y + 4, "Time:", c['c'])
        at(x + 10, y + 4, selected_tx['time'].strftime('%Y-%m-%d %H:%M:%S'), c['w'])

        at(x + 2, y + 5, "Type:", c['c'])
        at(x + 10, y + 5, selected_tx['type'].upper(), c['g'] if selected_tx['type'] == 'in' else c['R'])
        
        at(x + 2, y + 6, "Amount:", c['c'])
        at(x + 10, y + 6, f"{selected_tx['amt']:,.6f} OCT", c['B'] + c['g'])
        
        parsed_tx_data = selected_tx.get('parsed_tx')
        
        at(x + 2, y + 7, "From:", c['c'])
        at(x + 10, y + 7, parsed_tx_data.get('from', 'N/A') if parsed_tx_data else 'N/A', c['w'])

        at(x + 2, y + 8, "To:", c['c'])
        at(x + 10, y + 8, parsed_tx_data.get('to', 'N/A') if parsed_tx_data else 'N/A', c['w'])
        
        at(x + 2, y + 9, "Nonce:", c['c'])
        at(x + 10, y + 9, str(selected_tx['nonce']) if parsed_tx_data else 'N/A', c['w'])
        
        at(x + 2, y + 10, "Epoch:", c['c'])
        at(x + 10, y + 10, str(selected_tx['epoch']) if selected_tx['epoch'] else "Pending", c['y'] if not selected_tx['epoch'] else c['w'])

        at(x + 2, y + 12, "Raw Transaction (Partial):", c['c'])
        raw_tx_str = json.dumps(selected_tx.get('raw_tx', {}), indent=2) if selected_tx.get('raw_tx') else "{}"
        raw_lines = raw_tx_str.split('\n')
        for i, line in enumerate(raw_lines):
            if y + 13 + i >= y + hb - 2:
                at(x + 2, y + 13 + i, "(...)", c['y'])
                break
            at(x + 2, y + 13 + i, line, c['w'])

    except ValueError as e:
        _display_notification_wrapper(f"❌ Invalid input: {e}. Please enter a valid number.", c['R'], c['bg_error'])
        logging.error(f"ValueError in show_transaction_details: {e}", exc_info=True)
    except Exception as e:
        _display_notification_wrapper(f"❌ Error: {str(e)}", c['R'], c['bg_error'])
        logging.error(f"Error in show_transaction_details: {e}", exc_info=True)
    
    await awaitkey()
    ui_needs_redraw = True


async def show_info_box():
    cr = sz()
    cls()
    fill()
    main_box_w = cr[0] - 4
    w = main_box_w
    hb = 30
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    box(x, y, w, hb, "ℹ️ INFO & DISCLAIMER")

    at(x + 2, y + 2, "✨ What's New in v9.0 STABLE:", c['B'] + c['c'])
    at(x + 4, y + 3, "- Enhanced UI stability: Fixed missing box borders.", c['w'])
    at(x + 4, y + 4, "- Robust input handling: Prevents crashes from empty numeric inputs.", c['w'])
    at(x + 4, y + 5, "- Smarter Multi-Send: Improved nonce handling for reliable batch transactions.", c['w'])
    at(x + 4, y + 6, "- WatchDog Mode: Automated balance monitoring and transfer (NEW!).", c['w'])
    at(x + 4, y + 7, "- Multi-wallet support: Manage multiple accounts easily (NEW!).", c['w'])
    at(x + 4, y + 8, "- The client is now more stable and user-friendly.", c['w'])
    
    at(x + 2, y + 10, "⚠️ Disclaimer:", c['y'] + c['B'])
    at(x + 4, y + 11, "- This is a Testnet environment for Octra.", c['c'])
    at(x + 4, y + 12, "- Testnet tokens have NO commercial value.", c['c'])
    at(x + 4, y + 13, "- The client is actively updated. Monitor changes!", c['c'])

    at(x + 2, y + hb - 6, "🔗 Connect:", c['G'] + c['B'])
    at(x + 4, y + hb - 5, "Rebuild by : https://t.me/dayu_widayadi", c['y'] + c['B'])

    at(x + 2, y + hb - 3, "↩️ Navigation:", c['P'] + c['B'])
    at(x + 4, y + hb - 2, "Press Enter to go back...", c['y'])
    await awaitkey()
    ui_needs_redraw = True


async def manage_contacts():
    global contacts
    cr = sz()
    cls()
    fill()
    main_box_w = cr[0] - 4
    w, hb = main_box_w, cr[1] - 4
    x = (cr[0] - w) // 2
    y = 2
    box(x, y, w, hb, "📇 MANAGE CONTACTS")

    while True:
        cls()
        fill()
        box(x, y, w, hb, "📇 MANAGE CONTACTS")
        
        at(x + 2, y + 2, "Your Contacts:", c['B'] + c['c'])
        at(x + 2, y + 3, "──────────────────────────────────────────────────────────", c['w'])
        if not contacts:
            _display_notification_wrapper("No contacts found. Add one!", c['y'])
            at(x + 2, y + 5, " " * (w - 4), c['bg_box']) 
        else:
            display_count = 0
            sorted_contacts = list(sorted(contacts.items()))
            for i, (name, address) in enumerate(sorted_contacts):
                if display_count >= hb - 10:
                    at(x + 2, y + 5 + display_count, "(... more contacts)", c['y'])
                    break
                at(x + 2, y + 5 + display_count, f"[{i+1:>2}] {name:<20} : {address}", c['w'])
                display_count += 1
        
        at(x + 2, y + hb - 7, "──────────────────────────────────────────────────────────", c['w'])
        at(x + 2, y + hb - 6, "[1] Add New Contact", c['w'])
        at(x + 2, y + hb - 5, "[2] Edit Contact", c['w'])
        at(x + 2, y + hb - 4, "[3] Delete Contact", c['w'])
        at(x + 2, y + hb - 3, "[0] Back to Main Menu", c['y'])
        at(x + 2, y + hb - 2, "Choice: ", c['B'] + c['w'])

        choice_str = (await ainp(x + 10, y + hb - 2)).strip()

        if choice_str == '1':
            cls()
            fill()
            w_sub, hb_sub = main_box_w, 10
            x_sub = (cr[0] - w_sub) // 2
            y_sub = (cr[1] - hb_sub) // 2
            box(x_sub, y_sub, w_sub, hb_sub, "➕ ADD NEW CONTACT")
            at(x_sub + 2, y_sub + 3, "Enter contact name: ", c['y'])
            name = (await ainp(x_sub + 22, y_sub + 3)).strip()
            if not name:
                _display_notification_wrapper("Name cannot be empty!", c['R'], c['bg_error'])
                await awaitkey()
                continue
            if name in contacts:
                _display_notification_wrapper("Contact with this name already exists!", c['R'], c['bg_error'])
                await awaitkey()
                continue

            at(x_sub + 2, y_sub + 4, "Enter Octra address: ", c['y'])
            address = (await ainp(x_sub + 24, y_sub + 4)).strip()
            if not b58.match(address): 
                _display_notification_wrapper("❌ Invalid Octra address format or length!", c['R'], c['bg_error'])
                await awaitkey()
                continue
            
            contacts[name] = address
            save_contacts()
            _display_notification_wrapper(f"✅ Contact '{name}' added!", c['g'])
            await awaitkey()

        elif choice_str == '2':
            await edit_contact_flow()
            
            cls() 
            fill()

        elif choice_str == '3':
            await delete_contact_flow()
           
            cls() 
            fill()

        elif choice_str == '0':
            ui_needs_redraw = True
            return
        else:
            _display_notification_wrapper("Invalid choice.", c['R'], c['bg_error'])
            await asyncio.sleep(1)

async def edit_contact_flow():
    global contacts
    cr = sz()
    cls()
    fill()
    main_box_w = cr[0] - 4
    w, hb = main_box_w, cr[1] - 4
    x = (cr[0] - w) // 2
    y = 2
    box(x, y, w, hb, "📝 EDIT CONTACT")

    if not contacts:
        _display_notification_wrapper("No contacts to edit.", c['y'])
        await awaitkey()
        ui_needs_redraw = True
        return

    while True:
        cls()
        fill()
        box(x, y, w, hb, "📝 EDIT CONTACT")
        at(x + 2, y + 3, "Select a contact to edit (Enter # or name, 0 to cancel):", c['B'] + c['c'])
        at(x + 2, y + 4, "──────────────────────────────────────────────────────────", c['w'])
        
        sorted_contacts_list = list(sorted(contacts.items()))
        display_limit = min(len(sorted_contacts_list), hb - 10) 
        for i, (name, address) in enumerate(sorted_contacts_list[:display_limit]):
            at(x + 2, y + 5 + i, f"[{i+1:>2}] {name:<20} : {address}", c['w'])
        
        if len(sorted_contacts_list) > display_limit:
            at(x + 2, y + 5 + display_limit, f"(... {len(sorted_contacts_list) - display_limit} more contacts)", c['y'])

        at(x + 2, y + 6 + display_limit, "──────────────────────────────────────────────────────────", c['w'])
        at(x + 2, y + 7 + display_limit, "Choice (number or name): ", c['B'] + c['w'])
        
        edit_choice = (await ainp(x + 2 + len("Choice (number or name): "), y + 7 + display_limit)).strip()

        if edit_choice == '0':
            _display_notification_wrapper("Edit contact cancelled.", c['y'])
            await asyncio.sleep(1)
            ui_needs_redraw = True
            return
        
        contact_to_edit_name = None
        if edit_choice.isdigit():
            try:
                idx = int(edit_choice) - 1
                if 0 <= idx < len(sorted_contacts_list):
                    contact_to_edit_name = sorted_contacts_list[idx][0]
                else:
                    _display_notification_wrapper("❌ Invalid contact number.", c['R'], c['bg_error'])
                    await awaitkey()
                    continue
            except ValueError as e:
                _display_notification_wrapper("❌ Invalid input. Please enter a number or name.", c['R'], c['bg_error'])
                logging.error(f"ValueError in edit_contact_flow (choice): {e}", exc_info=True)
                await awaitkey()
                continue
        else:
            if edit_choice in contacts:
                contact_to_edit_name = edit_choice
            else:
                _display_notification_wrapper("❌ Contact name not found.", c['R'], c['bg_error'])
                await awaitkey()
                continue
        
        old_address = contacts[contact_to_edit_name]

        cls()
        fill()
        w_sub, hb_sub = main_box_w, 12
        x_sub = (cr[0] - w_sub) // 2
        y_sub = (cr[1] - hb_sub) // 2
        box(x_sub, y_sub, w_sub, hb_sub, f"EDITING: {contact_to_edit_name}")

        at(x_sub + 2, y_sub + 3, f"Current Name: {contact_to_edit_name}", c['c'])
        at(x_sub + 2, y_sub + 4, f"Current Address: {old_address}", c['c'])
        at(x_sub + 2, y_sub + 6, "Enter NEW name (leave empty to keep current): ", c['y'])
        new_name = (await ainp(x_sub + 2 + len("Enter NEW name (leave empty to keep current): "), y_sub + 6)).strip()
        
        at(x_sub + 2, y_sub + 8, "Enter NEW address (leave empty to keep current): ", c['y'])
        new_address = (await ainp(x_sub + 2 + len("Enter NEW address (leave empty to keep current): "), y_sub + 8)).strip()

        if not new_name and not new_address:
            _display_notification_wrapper("No changes made. Edit cancelled.", c['y'])
            await awaitkey()
            ui_needs_redraw = True
            return

        if new_address and not b58.match(new_address):
            _display_notification_wrapper("❌ Invalid Octra address format for new address!", c['R'], c['bg_error'])
            await awaitkey()
            continue

        if new_name and new_name != contact_to_edit_name and new_name in contacts:
            _display_notification_wrapper("❌ A contact with this new name already exists!", c['R'], c['bg_error'])
            await awaitkey()
            continue
        
        if new_name and new_name != contact_to_edit_name:
            contacts[new_name] = contacts.pop(contact_to_edit_name)
            contact_to_edit_name = new_name
            name_changed = True
        else:
            name_changed = False

        if new_address:
            contacts[contact_to_edit_name] = new_address
            address_changed = True
        else:
            address_changed = False

        if save_contacts():
            if name_changed and address_changed:
                _display_notification_wrapper(f"✅ Contact '{contact_to_edit_name}' updated (name and address)!", c['g'])
            elif name_changed:
                _display_notification_wrapper(f"✅ Contact '{contact_to_edit_name}' name updated!", c['g'])
            elif address_changed:
                _display_notification_wrapper(f"✅ Contact '{contact_to_edit_name}' address updated!", c['g'])
            else:
                _display_notification_wrapper("No changes applied.", c['y'])
        else:
            _display_notification_wrapper("❌ Failed to save changes to contacts.", c['R'], c['bg_error'])
        
        await awaitkey()
        ui_needs_redraw = True
        return

async def delete_contact_flow():
    global contacts
    cr = sz()
    cls()
    fill()
    main_box_w = cr[0] - 4
    w, hb = main_box_w, cr[1] - 4
    x = (cr[0] - w) // 2
    y = 2
    box(x, y, w, hb, "➖ DELETE CONTACT")

    if not contacts:
        _display_notification_wrapper("No contacts to delete.", c['y'])
        await awaitkey()
        ui_needs_redraw = True
        return

    while True:
        cls()
        fill()
        box(x, y, w, hb, "➖ DELETE CONTACT")
        at(x + 2, y + 3, "Select a contact to delete (Enter #, name, or 'all' to delete all; 0 to cancel):", c['B'] + c['c'])
        at(x + 2, y + 4, "──────────────────────────────────────────────────────────", c['w'])
        
        sorted_contacts_list = list(sorted(contacts.items()))
        display_limit = min(len(sorted_contacts_list), hb - 10) 
        for i, (name, address) in enumerate(sorted_contacts_list[:display_limit]):
            at(x + 2, y + 5 + i, f"[{i+1:>2}] {name:<20} : {address}", c['w'])
        
        if len(sorted_contacts_list) > display_limit:
            at(x + 2, y + 5 + display_limit, f"(... {len(sorted_contacts_list) - display_limit} more contacts)", c['y'])

        at(x + 2, y + 6 + display_limit, "──────────────────────────────────────────────────────────", c['w'])
        at(x + 2, y + 7 + display_limit, "Choice (number, name, or 'all'): ", c['B'] + c['w'])
        
        delete_choice = (await ainp(x + 2 + len("Choice (number, name, or 'all'): "), y + 7 + display_limit)).strip().lower()

        if delete_choice == '0':
            _display_notification_wrapper("Deletion cancelled.", c['y'])
            await asyncio.sleep(1)
            ui_needs_redraw = True
            return
        
        contacts_to_delete = []

        if delete_choice == 'all':
            if await display_confirmation("Are you sure you want to delete ALL contacts? This cannot be undone."):
                contacts_to_delete = list(contacts.keys())
            else:
                _display_notification_wrapper("Deletion of all contacts cancelled.", c['y'])
                await asyncio.sleep(1)
                continue
        elif delete_choice.isdigit():
            try:
                idx = int(delete_choice) - 1
                if 0 <= idx < len(sorted_contacts_list):
                    contacts_to_delete.append(sorted_contacts_list[idx][0])
                else:
                    _display_notification_wrapper("❌ Invalid contact number.", c['R'], c['bg_error'])
                    await awaitkey()
                    continue
            except ValueError as e:
                _display_notification_wrapper("❌ Invalid input. Please enter a number or name.", c['R'], c['bg_error'])
                logging.error(f"ValueError in delete_contact_flow (choice): {e}", exc_info=True)
                await awaitkey()
                continue
        else: 
            if delete_choice in contacts:
                contacts_to_delete.append(delete_choice)
            else:
                _display_notification_wrapper("❌ Contact name not found.", c['R'], c['bg_error'])
                await awaitkey()
                continue
        
        if not contacts_to_delete:
            continue

        successful_deletes = []
        for name_to_delete in contacts_to_delete:
            if await display_confirmation(f"Are you sure you want to delete contact '{name_to_delete}'?"): 
                try:
                    del contacts[name_to_delete]
                    successful_deletes.append(name_to_delete)
                except KeyError as e:
                    _display_notification_wrapper(f"Contact '{name_to_delete}' not found (already deleted?).", c['y'])
                    logging.warning(f"Attempted to delete non-existent contact: {name_to_delete}. Error: {e}", exc_info=True)
            else:
                _display_notification_wrapper(f"Deletion of '{name_to_delete}' cancelled.", c['y'])

        if successful_deletes:
            if save_contacts():
                _display_notification_wrapper(f"✅ Contacts deleted: {', '.join(successful_deletes)}", c['g'])
            else:
                _display_notification_wrapper("❌ Failed to save contacts after deletion.", c['R'], c['bg_error'])
        else:
            _display_notification_wrapper("No contacts were deleted.", c['y'])
        
        await awaitkey()
        ui_needs_redraw = True
        return

async def smart_multi_send():
    global lu, nonce_cache, ui_needs_redraw, h, cb
    cr = sz()
    cls()
    fill()

    main_box_w = cr[0] - 4
    notification_area_height = 5
    hb = cr[1] - notification_area_height - 2
    y = 1

    min_content_lines = 15
    if hb < min_content_lines + 2:
        hb = min_content_lines + 2

    x = (cr[0] - main_box_w) // 2
    
    box(x, y, main_box_w, hb, "🚀 SMART MULTI-SEND")

    if priv is None:
        _display_notification_wrapper("Wallet not loaded. Please generate or load one first.", c['R'], c['bg_error'])
        await awaitkey()
        ui_needs_redraw = True
        return

    recipients_to_send = []

    while True:
        cls() 
        fill()
        box(x, y, main_box_w, hb, "🚀 SMART MULTI-SEND: SELECT RECIPIENTS")
        at(x + 2, y + 3, "Choose Multi-Send Option:", c['B'] + c['c'])
        at(x + 2, y + 5, "[1] Send to All Contacts", c['w'])
        at(x + 2, y + 7, "[2] Select Specific Contacts", c['w'])
        at(x + 2, y + 9, "[3] Load Addresses from File (.txt)", c['w'])
        at(x + 2, y + 11, "[0] Back to Main Menu", c['y'])
        at(x + 2, y + 13, "Choice: ", c['B'] + c['w'])

        multi_choice = (await ainp(x + 10, y + 13)).strip()

        if multi_choice == '1':
            if not contacts:
                _display_notification_wrapper("No contacts found! Add contacts via menu option [6].", c['R'], c['bg_error'])
                await awaitkey()
                continue
            for name, address in contacts.items():
                recipients_to_send.append((address, name)) 
            _display_notification_wrapper(f"Selected: All {len(recipients_to_send)} contacts.", c['g'])
            await asyncio.sleep(1)
            break
        elif multi_choice == '2':
            cls()
            fill()
            w_sub = main_box_w
            hb_sub = cr[1] - notification_area_height - 2
            y_sub = 1
            if hb_sub < 15 + 2:
                hb_sub = 15 + 2
            x_sub = (cr[0] - w_sub) // 2

            box(x_sub, y_sub, w_sub, hb_sub, "🚀 SELECT CONTACTS FOR MULTI-SEND")
            at(x_sub + 2, y_sub + 3, "Available Contacts:", c['B'] + c['c'])
            at(x_sub + 2, y_sub + 4, "──────────────────────────────────────────────────────────", c['w'])
            
            contact_list_display = list(sorted(contacts.items()))
            # Adjust display limit based on hb_sub
            display_limit_contacts = min(len(contact_list_display), hb_sub - 10) 
            for i, (name, address) in enumerate(contact_list_display[:display_limit_contacts]):
                at(x_sub + 2, y_sub + 5 + i, f"[{i+1:>2}] {name:<20} : {address}", c['w'])
            
            if not contact_list_display:
                _display_notification_wrapper("No contacts to select.", c['R'], c['bg_error'])
                await awaitkey()
                continue
            
            at(x_sub + 2, y_sub + 5 + display_limit_contacts, "──────────────────────────────────────────────────────────", c['w'])
            at(x_sub + 2, y_sub + 6 + display_limit_contacts, "Enter numbers (e.g., 1,3,5) or 0 to cancel: ", c['B'] + c['w'])
            
            selected_indices_str = (await ainp(x_sub + 46, y_sub + 6 + display_limit_contacts)).strip()
            if not selected_indices_str or selected_indices_str == '0':
                ui_needs_redraw = True
                return
            
            try:
                selected_indices = [int(i.strip()) for i in selected_indices_str.split(',')]
                for idx in selected_indices:
                    if 1 <= idx <= len(contact_list_display):
                        name, address = contact_list_display[idx-1]
                        recipients_to_send.append((address, name))
                    else:
                        raise ValueError(f"Invalid contact number: {idx}")
            except ValueError as e:
                _display_notification_wrapper(f"❌ Invalid input: {e}", c['R'], c['bg_error'])
                logging.error(f"ValueError in smart_multi_send (select contacts): {e}", exc_info=True)
                await awaitkey()
                continue
            
            if not recipients_to_send:
                _display_notification_wrapper("No valid contacts selected. Try again.", c['y'])
                await awaitkey()
                continue
            _display_notification_wrapper(f"Selected {len(recipients_to_send)} contacts.", c['g'])
            await asyncio.sleep(1)
            break
        elif multi_choice == '3': 
            cls()
            fill()

            w_file_box = main_box_w
            hb_file_box = 8
            y_file_box = (cr[1] - hb_file_box) // 2
            x_file_box = (cr[0] - w_file_box) // 2
            
            box(x_file_box, y_file_box, w_file_box, hb_file_box, "📂 LOAD ADDRESSES FROM FILE")
            
            at(x_file_box + 2, y_file_box + 3, "Enter filename (e.g., addresses.txt): ", c['y'])
            filename = (await ainp(x_file_box + 40, y_file_box + 3)).strip()
            
            if not filename:
                _display_notification_wrapper("Filename cannot be empty.", c['R'], c['bg_error'])
                await awaitkey()
                continue

            try:
                with open(filename, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        address = line.strip()
                        if address and b58.match(address):
                            recipients_to_send.append((address, f"file_line_{line_num}"))
                        elif address:
                            _display_notification_wrapper(f"WARNING: Skipping invalid address format on line {line_num}: {address[:30]}...", c['R'], c['bg_error'])
                            logging.warning(f"Invalid address format in file {filename} on line {line_num}: {address[:30]}", exc_info=True)
                            await awaitkey()
                            recipients_to_send.clear()
                            break
                if not recipients_to_send:
                    if not recipients_to_send and "❌ Invalid address format" not in _display_notification_wrapper.last_message:
                        _display_notification_wrapper(f"File '{filename}' contains no valid addresses.", c['R'], c['bg_error'])
                    await awaitkey()
                    continue
                _display_notification_wrapper(f"✅ Loaded {len(recipients_to_send)} addresses from '{filename}'.", c['g'])
                await asyncio.sleep(1)
                break
            except FileNotFoundError as e:
                _display_notification_wrapper(f"❌ File not found: '{filename}'", c['R'], c['bg_error'])
                logging.error(f"File not found: {filename}. Error: {e}", exc_info=True)
                await awaitkey()
                continue
            except Exception as e:
                _display_notification_wrapper(f"❌ Error reading file: {str(e)}", c['R'], c['bg_error'])
                logging.error(f"Error reading file {filename}: {e}", exc_info=True)
                await awaitkey()
                continue

        elif multi_choice == '0':
            ui_needs_redraw = True
            return
        else:
            _display_notification_wrapper("Invalid choice.", c['R'], c['bg_error'])
            await asyncio.sleep(1)
            continue

    if not recipients_to_send:
        _display_notification_wrapper("No recipients selected. Exiting multi-send.", c['y'])
        await awaitkey()
        ui_needs_redraw = True
        return

    cls()
    fill()
    box(x, y, main_box_w, hb, "🚀 SMART MULTI-SEND")
    at(x + 2, y + 3, "Amount to send per recipient (OCT):", c['y'])
    amount_per_recipient_str = (await ainp(x + 39, y + 3)).strip()
    
    if not amount_per_recipient_str:
        _display_notification_wrapper("Amount cannot be empty. Multi-send cancelled.", c['R'], c['bg_error'])
        await awaitkey()
        ui_needs_redraw = True
        return

    try:
        amount_per_recipient = float(amount_per_recipient_str)
        if amount_per_recipient <= 0:
            raise ValueError("Amount must be positive.")
    except ValueError as e:
        _display_notification_wrapper(f"❌ Invalid amount! {e}", c['R'], c['bg_error'])
        logging.error(f"ValueError in smart_multi_send (amount): {e}", exc_info=True)
        await awaitkey()
        ui_needs_redraw = True
        return

    final_recipients_with_amount = [(addr_val, amount_per_recipient) for addr_val, name_val in recipients_to_send]
    
    tot = amount_per_recipient * len(final_recipients_with_amount)
    at(x + 2, y + 5, "──────────────────────────────────────────────────────────", c['w'])
    at(x + 2, y + 6, f"Summary: {len(final_recipients_with_amount)} recipients, Total: {tot:,.6f} OCT", c['B'] + c['y'])
    
    print(f"\033[{y + hb - 1};{x}H{c['bg_box']}{c['w']}└{'─' * (main_box_w - 2)}┘{c['bg_main']}", end='')
    sys.stdout.flush()

    lu = 0
    current_nonce_val, current_balance = await _fetch_balance_nonce()
    
    if current_balance is None:
        _display_notification_wrapper(f"❌ Insufficient balance! (Could not retrieve current balance).", c['R'], c['bg_error'])
        await awaitkey()
        ui_needs_redraw = True
        return

    if current_balance < tot:
        _display_notification_wrapper(f"❌ Insufficient balance! ({current_balance:,.6f} OCT < {tot:,.6f} OCT)", c['R'], c['bg_error'])
        await awaitkey()
        ui_needs_redraw = True
        return
    
    initial_nonce_for_batch = current_nonce_val + 1 
    confirm_y_pos = y + 8
    if not (await display_confirmation(f"Confirm send all? (Starting Nonce to use: {initial_nonce_for_batch}):")):
        ui_needs_redraw = True
        return
    
    spin_task_init = asyncio.create_task(spin_animation_in_notification_box("Initializing Multi-Send...", c['c']))
    
    s_total_multi, f_total_multi = 0, 0
    
    async def send_single_tx_in_multi(to_addr_multi, amount_multi, tx_index, total_tx_count, base_nonce):
        nonlocal s_total_multi, f_total_multi
        
        current_nonce_to_use = base_nonce + tx_index 

        current_tx_msg = f"⚡ Sending {amount_multi:,.6f} OCT to {to_addr_multi[:10]}... (Tx {tx_index+1}/{total_tx_count}, Nonce: {current_nonce_to_use})"
        spin_task_send = asyncio.create_task(spin_animation_in_notification_box(current_tx_msg, c['y']))
        
        ok, hs_msg, dt, _ = await snd(mk(to_addr_multi, amount_multi, current_nonce_to_use)[0])
        
        if spin_task_send and not spin_task_send.done():
            spin_task_send.cancel()
            try: await spin_task_send 
            except asyncio.CancelledError: pass

        if ok:
            s_total_multi += 1
            status_text = f"✅ SUCCESS: {hs_msg[:20]}..."
            h.append({
                'time': datetime.now(),
                'hash': hs_msg,
                'amt': amount_multi,
                'to': to_addr_multi,
                'type': 'out',
                'ok': True,
                'nonce': current_nonce_to_use,
                'epoch': 0 
            })

        else:
            f_total_multi += 1
            status_text = f"❌ FAILED: {str(hs_msg)[:20]}"
            
            if "duplicate" in hs_msg.lower():
                status_text += " (Duplicate TX? Nonce already used)" 
                _display_notification_wrapper(f"⚠️ Duplicate transaction detected for {to_addr_multi[:10]}... Nonce {current_nonce_to_use} might be used.", c['y'])
                logging.warning(f"Duplicate transaction detected for {to_addr_multi[:10]}... Nonce {current_nonce_to_use}", exc_info=True)


        _display_notification_wrapper(f"[{tx_index + 1}/{total_tx_count}] {amount_multi:,.6f} OCT to {to_addr_multi[:20]}... {status_text}", c['w'])
        await asyncio.sleep(0.05)
        return ok, hs_msg

    tasks = []
    for i, (to_addr_multi, amount_multi) in enumerate(final_recipients_with_amount):
        tasks.append(send_single_tx_in_multi(to_addr_multi, amount_multi, i, len(final_recipients_with_amount), initial_nonce_for_batch))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)

    if spin_task_init and not spin_task_init.done():
        spin_task_init.cancel()
        try: await spin_task_init 
        except asyncio.CancelledError: pass
    _display_notification_wrapper("", c['bg_box'])

    _display_notification_wrapper("Multi-send completed. Refreshing all data...", c['c'])
    lu = 0 
    await comprehensive_data_refresh() 
    
    final_msg_color = c['bg_success'] if f_total_multi == 0 else c['bg_error']
    _display_notification_wrapper(f"✨ Multi-send Complete: {s_total_multi} SUCCESS, {f_total_multi} FAILED", final_msg_color + c['w'] + c['B'])
    await awaitkey()
    ui_needs_redraw = True


async def create_new_account_flow():
    global priv, addr, rpc, sk, pub, ui_needs_redraw, active_wallet_idx, h, cb, cn, nonce_cache
    
    cr = sz()
    cls()
    fill()
    main_box_w = cr[0] - 4
    w, hb = main_box_w, 10
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    box(x, y, w, hb, "✨ GENERATE NEW ACCOUNT")

    spin_task = asyncio.create_task(spin_animation_in_notification_box("Generating new keypair...", c['c']))
    
    try:
        new_sk_obj = nacl.signing.SigningKey.generate()
        new_priv_b64 = base64.b64encode(new_sk_obj.encode()).decode()
        new_pub_bytes = new_sk_obj.verify_key.encode()
        new_addr_standard = get_octra_address_from_pubkey_bytes(new_pub_bytes)
        
        if spin_task and not spin_task.done():
            spin_task.cancel()
            try: await spin_task 
            except asyncio.CancelledError: pass 
            
        wallets.append({
            'priv': new_priv_b64,
            'addr': new_addr_standard,
            'rpc': 'https://octra.network'
        })
        active_wallet_idx = len(wallets) - 1
        
        if _save_all_wallets():
            priv = new_priv_b64
            addr = new_addr_standard
            rpc = 'https://octra.network'
            sk = new_sk_obj
            pub = base64.b64encode(new_pub_bytes).decode()
            
            h.clear()
            cb, cn = None, None 
            nonce_cache[addr] = 0 
            
            spin_task_refresh = asyncio.create_task(spin_animation_in_notification_box("New account created and refreshing data...", c['c']))
            await comprehensive_data_refresh() 
            if spin_task_refresh and not spin_task_refresh.done():
                spin_task_refresh.cancel()
                try: await spin_task_refresh
                except asyncio.CancelledError: pass

            _display_notification_wrapper(f"✅ New account {addr} generated and activated!", c['g'], c['bg_success'])
            at(x + 2, y + 3, f"Address: {addr}", c['w'])
            at(x + 2, y + 4, f"Private Key (SAVE THIS!): {priv}", c['y'])
        else:
            _display_notification_wrapper("❌ Failed to save new account.", c['R'], c['bg_error'])
            logging.error(f"Failed to save new account: Private key generated but save failed.", exc_info=True)
            
    except Exception as e:
        if spin_task and not spin_task.done():
            spin_task.cancel()
            try: await spin_task 
            except asyncio.CancelledError: pass
        _display_notification_wrapper(f"❌ Error generating new account: {str(e)}", c['R'], c['bg_error'])
        logging.error(f"Error generating new account: {e}", exc_info=True)
        
    await awaitkey()
    ui_needs_redraw = True

async def add_account():
    global ui_needs_redraw, nonce_cache
    cr = sz()
    cls()
    fill()
    main_box_w = cr[0] - 4
    w, hb = main_box_w, 10
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    box(x, y, w, hb, "➕ ADD NEW ACCOUNT")

    at(x + 2, y + 3, "Enter Private Key (base64): ", c['y'])
    new_priv_key_b64 = (await ainp(x + 30, y + 3)).strip()

    if not new_priv_key_b64:
        _display_notification_wrapper("Private Key cannot be empty!", c['R'], c['bg_error'])
        await awaitkey()
        ui_needs_redraw = True
        return
    
    try:
        temp_sk = nacl.signing.SigningKey(base64.b64decode(new_priv_key_b64))
        temp_pub_bytes = temp_sk.verify_key.encode()
        new_addr_standard = get_octra_address_from_pubkey_bytes(temp_pub_bytes)
        
        if not b58.match(new_addr_standard):
            raise ValueError("The provided private key generates an invalid Octra address format.")
        
    except Exception as e:
        _display_notification_wrapper(f"❌ Invalid Private Key format or generates an invalid address: {str(e)}", c['R'], c['bg_error'])
        logging.error(f"Invalid Private Key input: {e}", exc_info=True)
        await awaitkey()
        ui_needs_redraw = True
        return

    for wallet in wallets:
        if wallet.get('priv') == new_priv_key_b64:
            _display_notification_wrapper("Wallet already exists in your wallet.json!", c['y'])
            await awaitkey()
            ui_needs_redraw = True
            return

    wallets.append({
        'priv': new_priv_key_b64,
        'addr': new_addr_standard, 
        'rpc': 'https://octra.network' 
    })

    if _save_all_wallets():
        nonce_cache[new_addr_standard] = nonce_cache.get(new_addr_standard, 0) 
        _display_notification_wrapper(f"✅ Account {new_addr_standard} added successfully!", c['g'], c['bg_success'])
        await comprehensive_data_refresh() 
    else:
        _display_notification_wrapper("❌ Failed to save new account.", c['R'], c['bg_error'])
        logging.error(f"Failed to save new account via add_account: {new_addr_standard}", exc_info=True)
    
    await awaitkey()
    ui_needs_redraw = True

async def change_account():
    global active_wallet_idx, priv, addr, rpc, sk, pub, ui_needs_redraw, h, cb, cn, nonce_cache
    cr = sz()
    cls()
    fill()
    main_box_w = cr[0] - 4
    w, hb = main_box_w, cr[1] - 4
    x = (cr[0] - w) // 2
    y = 2
    box(x, y, w, hb, "🔄 CHANGE ACCOUNT")

    if not wallets:
        _display_notification_wrapper("No accounts found. Please add a new account first.", c['y'])
        await awaitkey()
        ui_needs_redraw = True
        return

    at(x + 2, y + 2, "Available Accounts (Enter # to select, 0 to cancel):", c['B'] + c['c'])
    at(x + 2, y + 3, "──────────────────────────────────────────────────────────", c['w'])
    
    for i, wallet in enumerate(wallets):
        is_active = " (ACTIVE)" if i == active_wallet_idx else ""
        at(x + 2, y + 4 + i, f"[{i+1:>2}] {wallet['addr']}{is_active}", c['w'] if i != active_wallet_idx else c['B'] + c['g'])
    
    at(x + 2, y + 5 + len(wallets), "──────────────────────────────────────────────────────────", c['w'])
    at(x + 2, y + 6 + len(wallets), "Choice: ", c['B'] + c['w'])

    choice_str = (await ainp(x + 10, y + 6 + len(wallets))).strip()
    try:
        choice = int(choice_str)
        if choice == 0:
            _display_notification_wrapper("Account switch cancelled.", c['y'])
            await asyncio.sleep(1)
            ui_needs_redraw = True
            return
        if not (1 <= choice <= len(wallets)):
            raise ValueError("Invalid choice.")
        
        new_idx = choice - 1
        if new_idx == active_wallet_idx:
            _display_notification_wrapper("This account is already active.", c['y'])
            await asyncio.sleep(1)
            ui_needs_redraw = True
            return
        
        h.clear() 
        cb, cn = None, None 

        if _load_and_activate_wallet(wallets[new_idx]):
            active_wallet_idx = new_idx 
            if _save_all_wallets():
                spin_task = asyncio.create_task(spin_animation_in_notification_box("Switching account and refreshing data...", c['c']))
                await comprehensive_data_refresh() 
                if spin_task and not spin_task.done(): 
                    spin_task.cancel()
                    try: await spin_task 
                    except asyncio.CancelledError: pass

                _display_notification_wrapper(f"✅ Switched to account {addr[:10]}... and data refreshed!", c['g'], c['bg_success'])
            else:
                _display_notification_wrapper("❌ Failed to switch account (save error).", c['R'], c['bg_error'])
                logging.error(f"Failed to switch account (save error) to index {new_idx}", exc_info=True)
        else:
            _display_notification_wrapper("❌ Failed to switch account (activation error).", c['R'], c['bg_error'])
            logging.error(f"Failed to switch account (activation error) to index {new_idx}", exc_info=True)

    except ValueError as e:
        _display_notification_wrapper(f"❌ Invalid input. {e}", c['R'], c['bg_error'])
        logging.error(f"ValueError in change_account: {e}", exc_info=True)
    except Exception as e:
        _display_notification_wrapper(f"❌ Error switching account: {str(e)}", c['R'], c['bg_error'])
        logging.error(f"Error switching account: {e}", exc_info=True)
    finally:
        if 'spin_task' in locals() and spin_task and not spin_task.done(): 
            spin_task.cancel()
            try: await spin_task 
            except asyncio.CancelledError: pass
        _display_notification_wrapper("", c['bg_box'])
    
    await awaitkey()
    ui_needs_redraw = True

async def delete_account_flow():
    global active_wallet_idx, priv, addr, rpc, sk, pub, ui_needs_redraw, h, cb, cn, wallets, nonce_cache
    cr = sz()
    cls()
    fill()
    main_box_w = cr[0] - 4
    w, hb = main_box_w, cr[1] - 4
    x = (cr[0] - w) // 2
    y = 2
    box(x, y, w, hb, "🗑️ DELETE ACCOUNT")

    if not wallets:
        _display_notification_wrapper("No accounts to delete.", c['y'])
        await awaitkey()
        ui_needs_redraw = True
        return
    if len(wallets) == 1:
        _display_notification_wrapper("Cannot delete the only account in your wallet. Please create another account first.", c['y'])
        await awaitkey()
        ui_needs_redraw = True
        return

    at(x + 2, y + 2, "Available Accounts (Enter # to delete, 0 to cancel):", c['B'] + c['c'])
    at(x + 2, y + 3, "──────────────────────────────────────────────────────────", c['w'])
    
    for i, wallet in enumerate(wallets):
        is_active = " (ACTIVE)" if i == active_wallet_idx else ""
        at(x + 2, y + 4 + i, f"[{i+1:>2}] {wallet['addr']}{is_active}", c['w'] if i != active_wallet_idx else c['B'] + c['g'])
    
    at(x + 2, y + 5 + len(wallets), "──────────────────────────────────────────────────────────", c['w'])
    at(x + 2, y + 6 + len(wallets), "Choice: ", c['B'] + c['w'])

    choice_str = (await ainp(x + 10, y + 6 + len(wallets))).strip()
    try:
        choice = int(choice_str)
        if choice == 0:
            ui_needs_redraw = True
            return
        if not (1 <= choice <= len(wallets)):
            raise ValueError("Invalid choice.")
        
        idx_to_delete = choice - 1
        
        addr_to_delete = wallets[idx_to_delete]['addr']
        
        if await display_confirmation(f"Are you sure you want to delete account {addr_to_delete[:15]}...?"): 
            deleted_was_active = (idx_to_delete == active_wallet_idx)
            del wallets[idx_to_delete]
            
            if addr_to_delete in nonce_cache:
                del nonce_cache[addr_to_delete]

            if _save_all_wallets():
                if deleted_was_active:
                    _display_notification_wrapper(f"Account {addr_to_delete[:15]}... deleted. Switching to new active account...", c['y'])
                    if wallets:
                        active_wallet_idx = 0 
                        if _load_and_activate_wallet(wallets[active_wallet_idx]):
                            h.clear() 
                            cb, cn = None, None 
                            spin_task_nonce = asyncio.create_task(spin_animation_in_notification_box("Fetching nonce for new active account...", c['c']))
                            async with nonce_lock:
                                current_rpc_nonce_for_new_addr, _ = await _fetch_balance_nonce()
                                if current_rpc_nonce_for_new_addr is not None:
                                    nonce_cache[addr] = current_rpc_nonce_for_new_addr
                                    cn = current_rpc_nonce_for_new_addr
                                else:
                                    nonce_cache[addr] = 0
                                    cn = 0
                                    _display_notification_wrapper("Warning: Failed to get RPC nonce for new active account. Nonce initialized to 0.", c['y'])
                                    logging.warning(f"Failed to get RPC nonce for new active account {addr}. Nonce initialized to 0.", exc_info=True)
                            if spin_task_nonce and not spin_task_nonce.done(): spin_task_nonce.cancel(); await asyncio.sleep(0.01)

                            spin_task = asyncio.create_task(spin_animation_in_notification_box("Refreshing data for new active account...", c['c']))
                            await comprehensive_data_refresh() 
                            if spin_task and not spin_task.done(): 
                                spin_task.cancel()
                                try: await spin_task 
                                except asyncio.CancelledError: pass

                            _display_notification_wrapper(f"✅ Account {addr_to_delete[:15]}... deleted. Switched to {addr[:15]}...", c['g'], c['bg_success'])
                        else:
                            _display_notification_wrapper("❌ Failed to activate new account after deletion.", c['R'], c['bg_error'])
                            logging.error(f"Failed to activate new account after deleting {addr_to_delete}", exc_info=True)
                    else:
                        priv, addr, rpc, sk, pub = None, None, None, None, None
                        active_wallet_idx = 0
                        h.clear()
                        cb, cn = None, None 
                        _display_notification_wrapper(f"✅ Last account {addr_to_delete[:15]}... deleted. Please create or add a new account.", c['g'], c['bg_success'])
                else:
                    if active_wallet_idx > idx_to_delete:
                        active_wallet_idx -= 1
                    _save_all_wallets() 
                    _display_notification_wrapper(f"✅ Account {addr_to_delete[:15]}... deleted successfully!", c['g'], c['bg_success'])
            else:
                _display_notification_wrapper("❌ Failed to delete account (save error).", c['R'], c['bg_error'])
                logging.error(f"Failed to delete account {addr_to_delete} (save error)", exc_info=True)
        else:
            _display_notification_wrapper("Deletion cancelled.", c['y'])

    except ValueError as e:
        _display_notification_wrapper(f"❌ Invalid input. {e}", c['R'], c['bg_error'])
        logging.error(f"ValueError in delete_account_flow: {e}", exc_info=True)
    except Exception as e:
        _display_notification_wrapper(f"❌ Error deleting account: {str(e)}", c['R'], c['bg_error'])
        logging.error(f"Error deleting account: {e}", exc_info=True)
    finally:
        if 'spin_task_nonce' in locals() and spin_task_nonce and not spin_task_nonce.done(): spin_task_nonce.cancel(); await asyncio.sleep(0.01)
        if 'spin_task' in locals() and spin_task and not spin_task.done(): spin_task.cancel(); await asyncio.sleep(0.01)
        _display_notification_wrapper("", c['bg_box'])
    
    await awaitkey()
    ui_needs_redraw = True

async def exp():
    global ui_needs_redraw
    cr = sz()
    cls()
    fill()
    main_box_w = cr[0] - 4
    w, hb = main_box_w, 20
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    box(x, y, w, hb, "🔑 EXPORT/MANAGE KEYS")
    
    at(x + 2, y + 2, "Current Wallet Information:", c['c'])
    at(x + 2, y + 4, "Address:", c['y'])
    at(x + 11, y + 4, addr, c['w'])
    at(x + 2, y + 5, "Balance:", c['y'])
    at(x + 11, y + 5, f"{cb:,.6f} OCT" if cb is not None else "---", c['g'] + c['B'] if cb is not None else c['w'])
    
    at(x + 2, y + 7, "Export/Manage Options:", c['B'] + c['c'])
    at(x + 2, y + 8, "[1] Create New Account", c['w']) 
    at(x + 2, y + 9, "[2] Show Private Key (DANGER!)", c['R'])
    at(x + 2, y + 10, "[3] Save Full Wallet to File", c['w'])
    at(x + 2, y + 11, "[4] Add Existing Account", c['w'])
    at(x + 2, y + 12, "[5] Change Account", c['w'])
    at(x + 2, y + 13, "[6] Delete Account", c['R']) 
    at(x + 2, y + 14, "[0] Cancel and Go Back", c['y'])
    at(x + 2, y + 16, "Your choice: ", c['B'] + c['w'])
    
    choice = (await ainp(x + 15, y + 16)).strip()
    
    for i in range(7, 17): 
        at(x + 2, y + i, " " * (w - 4), c['bg_main'])
    
    if choice == '1': 
        await create_new_account_flow()
    elif choice == '2': 
        at(x + 2, y + 7, "🚨 PRIVATE KEY (KEEP THIS SECRET!):", c['R'] + c['B'])
        at(x + 2, y + 8, priv, c['w'])
        at(x + 2, y + 11, "PUBLIC KEY:", c['g'])
        at(x + 2, y + 12, pub, c['g'])
        await awaitkey()
    elif choice == '3': 
        fn = f"octra_wallet_full_backup_{int(time.time())}.json"
        wallet_data = {
            'wallets': wallets,
            'active_wallet_idx': active_wallet_idx
        }
        try:
            if await display_confirmation(f"Save full wallet data to '{fn}'? This file contains ALL your private keys."): 
                with open(fn, 'w') as f:
                    json.dump(wallet_data, f, indent=2)
                _display_notification_wrapper(f"✅ Full wallet saved to: {fn}", c['g'], c['bg_success'])
                _display_notification_wrapper("❗ File contains ALL your private keys - KEEP IT SAFE!", c['R'], c['bg_error'])
            else:
                _display_notification_wrapper("Wallet backup cancelled.", c['y'])
        except Exception as e:
            _display_notification_wrapper(f"❌ Error saving wallet: {str(e)}", c['R'], c['bg_error'])
            logging.error(f"Error saving full wallet to {fn}: {e}", exc_info=True)
        await awaitkey()
    elif choice == '4': 
        await add_account()
    elif choice == '5': 
        await change_account()
    elif choice == '6':
        await delete_account_flow()
    elif choice == '0':
        pass
    else:
        _display_notification_wrapper(f"❌ Invalid choice.", c['R'], c['bg_error'])
        await asyncio.sleep(1)
    
    ui_needs_redraw = True 

async def save_addresses_flow():
    global ui_needs_redraw
    cr = sz()
    cls()
    fill()
    main_box_w = cr[0] - 4
    w, hb = main_box_w, 15 
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    box(x, y, w, hb, "💾 SAVE ADDRESSES")

    if priv is None:
        _display_notification_wrapper("Wallet not loaded. Cannot save addresses.", c['R'], c['bg_error'])
        await awaitkey()
        ui_needs_redraw = True
        return

    while True:
        cls()
        fill()
        box(x, y, w, hb, "💾 SAVE ADDRESSES")
        
        at(x + 2, y + 3, "Choose Save Option:", c['B'] + c['c'])
        at(x + 2, y + 5, "[1] Save Inbound Addresses (to in_address.txt)", c['w'])
        at(x + 2, y + 7, "[2] Save Outbound Addresses (to out_address.txt)", c['w'])
        at(x + 2, y + 9, "[0] Back to Main Menu", c['y'])
        at(x + 2, y + 11, "Choice: ", c['B'] + c['w'])

        choice_str = (await ainp(x + 10, y + 11)).strip()

        if choice_str == '1':
            await _save_filtered_addresses('in')
            break
        elif choice_str == '2':
            await _save_filtered_addresses('out')
            break
        elif choice_str == '0':
            break
        else:
            _display_notification_wrapper("❌ Invalid choice.", c['R'], c['bg_error'])
            await asyncio.sleep(1)
    ui_needs_redraw = True 

async def _save_filtered_addresses(tx_type):
    global h, ui_needs_redraw
    filename = f"{tx_type}_address.txt"
    saved_count = 0
    
    if not h:
        _display_notification_wrapper("No transactions in history to save addresses from.", c['y'])
        await awaitkey()
        return

    existing_addresses = set()
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                for line in f:
                    addr_in_file = line.strip()
                    if addr_in_file:
                        existing_addresses.add(addr_in_file)
    except Exception as e:
        _display_notification_wrapper(f"WARNING: Could not read existing addresses from {filename}: {str(e)}. Proceeding anyway.", c['y'])
        logging.warning(f"Could not read existing addresses from {filename}: {e}", exc_info=True)
        await asyncio.sleep(2)

    new_addresses_to_save = []
    
    for tx in h[:50]:
        address_to_process = None
        if tx['type'] == tx_type:
            if tx_type == 'in' and tx.get('parsed_tx') and tx['parsed_tx'].get('from'):
                address_to_process = tx['parsed_tx'].get('from')
            elif tx_type == 'out' and tx.get('to'):
                address_to_process = tx['to'] 

            if address_to_process and address_to_process not in existing_addresses:
                if b58.match(address_to_process):
                    new_addresses_to_save.append(address_to_process)
                    existing_addresses.add(address_to_process)
                else:
                    _display_notification_wrapper(f"WARNING: Skipping invalid address format found in history: {address_to_process[:20]}...", c['y'])
                    logging.warning(f"Skipping invalid address format found in history ({tx_type}): {address_to_process[:20]}", exc_info=True)
                    await asyncio.sleep(0.5)

    if not new_addresses_to_save:
        _display_notification_wrapper(f"No new {tx_type} addresses found to save to {filename}.", c['y'])
        await awaitkey()
        return

    confirmation_message = f"Found {len(new_addresses_to_save)} new {tx_type} addresses. Save them to '{filename}'?"
    if not await display_confirmation(confirmation_message):
        _display_notification_wrapper("Saving addresses cancelled by user.", c['y'])
        await awaitkey()
        return

    try:
        with open(filename, 'a') as f:
            for addr_to_write in new_addresses_to_save:
                f.write(addr_to_write + '\n')
                saved_count += 1
        _display_notification_wrapper(f"✅ Successfully saved {saved_count} new {tx_type} addresses to '{filename}'!", c['g'], c['bg_success'])
    except Exception as e:
        _display_notification_wrapper(f"❌ ERROR saving addresses to '{filename}': {str(e)}", c['R'], c['bg_error'])
        logging.error(f"Error saving addresses to {filename}: {e}", exc_info=True)
    
    await awaitkey()
    ui_needs_redraw = True

async def auto_send_loop_v2(to_addr, amount, num_tx): 
    global auto_send_task, auto_send_stop_event, ui_needs_redraw, nonce_cache, cb

    tx_sent_count = 0
    tx_failed_count = 0
    auto_send_stop_event.clear()

    _display_notification_wrapper(f"⚡ AUTO-SEND: Preparing {num_tx} transactions...", c['c'])
    
    try:
        current_nonce_for_send, current_balance_for_send = await _fetch_balance_nonce() 
        
        if current_nonce_for_send is None or current_balance_for_send is None:
            _display_notification_wrapper(f"❌ AUTO-SEND: Failed to get starting nonce or balance. Check RPC connection. Stopping.", c['R'], c['bg_error'])
            logging.error(f"AUTO-SEND: Failed to get starting nonce or balance for address {addr}.", exc_info=True)
            return

        total_amount_needed = amount * num_tx
        if current_balance_for_send < total_amount_needed:
             _display_notification_wrapper(f"❌ AUTO-SEND: Insufficient balance! ({current_balance_for_send:,.6f} OCT < {total_amount_needed:,.6f} OCT)", c['R'], c['bg_error'])
             logging.error(f"AUTO-SEND: Insufficient balance for address {addr}. Needed {total_amount_needed}, has {current_balance_for_send}", exc_info=True)
             await awaitkey()
             return

        initial_nonce_for_batch = current_nonce_for_send + 1 
        _display_notification_wrapper(f"⚡ AUTO-SEND: Starting Nonce for batch: {initial_nonce_for_batch}. Building tasks...", c['c'])

        tasks = []
        for i in range(num_tx):
            if auto_send_stop_event.is_set():
                _display_notification_wrapper("Auto-send stopped by user during task preparation.", c['y'])
                break
            
            nonce_for_this_tx = initial_nonce_for_batch + i 
            tasks.append(asyncio.create_task(_send_single_auto_tx(to_addr, amount, nonce_for_this_tx, i + 1, num_tx)))
            
            async with nonce_lock:
                nonce_cache[addr] = nonce_for_this_tx

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple):
                ok, _, _ = result 
                if ok:
                    tx_sent_count += 1
                else:
                    tx_failed_count += 1
            else:
                logging.error(f"Error in auto-send task result: {result}", exc_info=True)
                tx_failed_count += 1

    except asyncio.CancelledError:
        _display_notification_wrapper("Auto-send task cancelled.", c['y'])
    except Exception as e:
        _display_notification_wrapper(f"❌ ERROR in auto-send: {str(e)}", c['R'], c['bg_error'])
        logging.error(f"Critical error in auto-send loop: {e}", exc_info=True)
    finally:
        auto_send_task = None 
        auto_send_stop_event.clear() 
        _display_notification_wrapper(f"Auto-send complete. ({tx_sent_count} SUCCESS, {tx_failed_count} FAILED)", c['g'] if tx_failed_count == 0 else c['R'], c['bg_success'] if tx_failed_count == 0 else c['bg_error'])
        await awaitkey()
        ui_needs_redraw = True

        await comprehensive_data_refresh() 

async def _send_single_auto_tx(to_addr, amount, nonce_to_use, current_tx_idx, total_tx_count):
    global h, ui_needs_redraw
    
    current_tx_msg = f"⚡ Sending {amount:,.6f} OCT to {to_addr[:10]}... (Tx {current_tx_idx}/{total_tx_count}, Nonce: {nonce_to_use})"
    spin_task_send = asyncio.create_task(spin_animation_in_notification_box(current_tx_msg, c['y']))
    
    try:
        tx_obj, tx_hash_local = mk(to_addr, amount, nonce_to_use)
        ok, hs_msg, dt, _ = await snd(tx_obj)

        if spin_task_send and not spin_task_send.done():
            spin_task_send.cancel()
            try: await spin_task_send 
            except asyncio.CancelledError: pass

        if ok:
            status_text = f"✅ SUCCESS: {hs_msg[:20]}..."
            h.append({
                'time': datetime.now(),
                'hash': hs_msg,
                'amt': amount,
                'to': to_addr,
                'type': 'out',
                'ok': True,
                'nonce': nonce_to_use, 
                'epoch': 0 
            })
        else:
            status_text = f"❌ FAILED: {str(hs_msg)[:20]}"
            if "duplicate" in hs_msg.lower():
                status_text += " (Duplicate TX?)" 
                logging.warning(f"Duplicate transaction detected for auto-send to {to_addr[:10]}... Nonce {nonce_to_use}", exc_info=True)

        _display_notification_wrapper(f"[{current_tx_idx}/{total_tx_count}] {amount:,.6f} OCT to {to_addr[:20]}... {status_text}", c['w'] if ok else c['R'])
        ui_needs_redraw = True
        return ok, hs_msg, nonce_to_use
    except Exception as e:
        if spin_task_send and not spin_task_send.done():
            spin_task_send.cancel()
            try: await spin_task_send 
            except asyncio.CancelledError: pass
        _display_notification_wrapper(f"❌ TX {current_tx_idx} ERROR: {str(e)}", c['R'], c['bg_error'])
        logging.error(f"Error sending single auto transaction (TX {current_tx_idx}): {e}", exc_info=True)
        ui_needs_redraw = True
        return False, str(e), nonce_to_use


async def watchdog_monitor_loop():
    global watchdog_task, watchdog_stop_event, ui_needs_redraw, cb, cn

    _display_notification_wrapper(f"🐶 WatchDog: Monitoring balance every {WATCHDOG_CHECK_INTERVAL} seconds...", c['c'])
    
    is_indefinite = (watchdog_duration_seconds == float('inf'))
    end_time = None
    if not is_indefinite:
        end_time = watchdog_start_time + timedelta(seconds=watchdog_duration_seconds)

    try:
        while not watchdog_stop_event.is_set():
            if not is_indefinite and datetime.now() > end_time:
                _display_notification_wrapper("🐶 WatchDog: Duration ended. Stopping monitor.", c['y'])
                break
            
            remaining_time_str = ""
            if not is_indefinite:
                remaining_time = end_time - datetime.now()
                hours, remainder = divmod(remaining_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                remaining_time_str = f". Remaining: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
            else:
                remaining_time_str = ". Running indefinitely"

            spin_task_wd_refresh = asyncio.create_task(
                spin_animation_in_notification_box(
                    f"🐶 WatchDog: Monitoring... Bal: {cb:,.6f} OCT{remaining_time_str}", 
                    c['y']
                )
            )

            current_nonce_val, current_balance = await _fetch_balance_nonce()
            
            if spin_task_wd_refresh and not spin_task_wd_refresh.done():
                spin_task_wd_refresh.cancel()
                try: await spin_task_wd_refresh
                except asyncio.CancelledError: pass
            _display_notification_wrapper("", c['bg_box'])

            if current_balance is None:
                _display_notification_wrapper("🐶 WatchDog: Failed to get balance. Retrying...", c['R'])
                logging.error(f"WatchDog: Failed to get balance for address {addr}. Retrying.", exc_info=True)
                await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)
                continue

            if current_balance > 0:
                _display_notification_wrapper(f"🐶 WatchDog: Balance detected ({current_balance:,.6f} OCT > 0). Initiating transfer...", c['y'])

                amount_to_send_main = current_balance * (1 - TOTAL_FEE_PERCENTAGE)
                amount_to_dev_donation = current_balance * DEV_DONATION_PERCENTAGE
                
                if amount_to_send_main <= 0.000001: 
                    _display_notification_wrapper(f"🐶 WatchDog: Calculated main send amount too small ({amount_to_send_main:,.6f}). Skipping this cycle.", c['y'])
                    logging.warning(f"WatchDog: Calculated main send amount too small ({amount_to_send_main}). Skipping cycle for address {addr}.", exc_info=True)
                    await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)
                    continue

                if current_nonce_val is None: 
                    _display_notification_wrapper("🐶 WatchDog: Failed to get nonce for transfer. Retrying next cycle.", c['R'])
                    logging.error(f"WatchDog: Failed to get nonce for address {addr}. Retrying next cycle.", exc_info=True)
                    await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)
                    continue

                tx_nonce_main = current_nonce_val + 1
                tx_nonce_dev = current_nonce_val + 2

                async with nonce_lock:
                    nonce_cache[addr] = tx_nonce_dev

                _display_notification_wrapper(f"🐶 WatchDog: Sending {amount_to_send_main:,.6f} OCT to {watchdog_target_address[:10]}... (Nonce: {tx_nonce_main})", c['c'])
                ok_main, hash_main, _, _ = await snd(mk(watchdog_target_address, amount_to_send_main, tx_nonce_main)[0])
                if ok_main:
                    _display_notification_wrapper(f"✅ WatchDog: Main transfer SUCCESS! Hash: {hash_main[:20]}...", c['g'])
                    h.append({'time': datetime.now(), 'hash': hash_main, 'amt': amount_to_send_main, 'to': watchdog_target_address, 'type': 'out', 'ok': True, 'nonce': tx_nonce_main, 'epoch': 0})
                else:
                    _display_notification_wrapper(f"❌ WatchDog: Main transfer FAILED! Error: {hash_main}", c['R'])
                    logging.error(f"WatchDog: Main transfer FAILED for {watchdog_target_address}. Error: {hash_main}", exc_info=True)

                if amount_to_dev_donation > 0.000001: 
                    _display_notification_wrapper(f"🐶 WatchDog: Sending {amount_to_dev_donation:,.6f} OCT donation to dev wallet {DEV_DONATION_WALLET[:10]}... (Nonce: {tx_nonce_dev})", c['c'])
                    ok_dev, hash_dev, _, _ = await snd(mk(DEV_DONATION_WALLET, amount_to_dev_donation, tx_nonce_dev)[0])
                    if ok_dev:
                        _display_notification_wrapper(f"✅ WatchDog: Dev donation SUCCESS! Hash: {hash_dev[:20]}...", c['g'])
                        h.append({'time': datetime.now(), 'hash': hash_dev, 'amt': amount_to_dev_donation, 'to': DEV_DONATION_WALLET, 'type': 'out', 'ok': True, 'nonce': tx_nonce_dev, 'epoch': 0})
                    else:
                        _display_notification_wrapper(f"❌ WatchDog: Dev donation FAILED! Error: {hash_dev}", c['R'])
                        logging.error(f"WatchDog: Dev donation FAILED for {DEV_DONATION_WALLET}. Error: {hash_dev}", exc_info=True)
                else:
                    _display_notification_wrapper("🐶 WatchDog: Dev donation amount too small, skipping.", c['y'])
                    logging.info(f"WatchDog: Dev donation amount too small ({amount_to_dev_donation}), skipping for address {addr}.")

                _display_notification_wrapper("🐶 WatchDog: All detected balance processed. Pausing for 5 seconds...", c['g'])
                await comprehensive_data_refresh() 
                await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)

            await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)

    except asyncio.CancelledError:
        _display_notification_wrapper("🐶 WatchDog: Monitor stopped.", c['y'])
    except Exception as e:
        _display_notification_wrapper(f"❌ WatchDog: Critical error: {str(e)}", c['R'], c['bg_error'])
        logging.critical(f"Critical error in WatchDog monitor loop for address {addr}: {e}", exc_info=True)
    finally:
        watchdog_task = None
        watchdog_stop_event.clear()
        _display_notification_wrapper("🐶 WatchDog Mode is INACTIVE.", c['y'])
        await awaitkey()
        ui_needs_redraw = True

async def start_watchdog_mode():
    global watchdog_task, watchdog_target_address, watchdog_duration_seconds, watchdog_start_time, ui_needs_redraw
    
    if priv is None:
        _display_notification_wrapper("Wallet not loaded. Please generate or load one first.", c['R'], c['bg_error'])
        await awaitkey()
        ui_needs_redraw = True
        return

    if watchdog_task and not watchdog_task.done():
        _display_notification_wrapper("🐶 WatchDog Mode is already active! Stop it first (press 0 from main menu) if you want to reconfigure.", c['y'])
        await awaitkey()
        ui_needs_redraw = True
        return

    cr = sz()
    cls() 
    fill()

    main_box_w = cr[0] - 4
    w, hb = main_box_w, 18
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    box(x, y, w, hb, "🐶 WATCHDOG MODE SETUP")

    at(x + 2, y + 2, "WatchDog Mode: Automatically monitors your account balance.", c['P'] + c['B'])
    at(x + 2, y + 3, "When balance > 0 is detected, the system will automatically", c['P'] + c['B'])
    at(x + 2, y + 4, "send all funds (minus fees) to the target address.", c['P'] + c['B'])
    at(x + 2, y + 5, "Useful for faucets, mining, or securing funds.", c['P'] + c['B'])
    at(x + 2, y + 6, "---------------------------------------------------------", c['w'])

    while True:
        target_addr_prompt_y = y + 7
        resolved_to_display_y = y + 9
        
        for i in range(target_addr_prompt_y, resolved_to_display_y + 1):
            at(x + 2, i, " " * (main_box_w - 4), c['bg_box'])

        prompt_addr = "Enter Target Address (or contact name, 0 to cancel):"
        at(x + 2, target_addr_prompt_y, prompt_addr, c['y'])
        
        target_addr_input = (await ainp(x + 2 + len(prompt_addr) + 1, target_addr_prompt_y)).strip()

        if target_addr_input == '0':
            _display_notification_wrapper("🐶 WatchDog Mode setup cancelled.", c['y'])
            ui_needs_redraw = True
            return
        
        resolved_address_from_contact = None
        if target_addr_input in contacts:
            resolved_address_from_contact = contacts[target_addr_input]
            at(x + 2, resolved_to_display_y, f"Contact resolved to: {resolved_address_from_contact}", c['g'])
            watchdog_target_address = resolved_address_from_contact
            break
        elif b58.match(target_addr_input):
            watchdog_target_address = target_addr_input
            break
        else:
            _display_notification_wrapper("❌ Invalid Octra address format or contact name! Please try again.", c['R'], c['bg_error'])
            await asyncio.sleep(1)
            continue 

    if watchdog_target_address == addr:
        _display_notification_wrapper("❌ Target address cannot be your own wallet address!", c['R'], c['bg_error'])
        await awaitkey()
        ui_needs_redraw = True
        return

    while True:
        duration_prompt_y = resolved_to_display_y + 2
        
        at(x + 2, duration_prompt_y, " " * (main_box_w - 4), c['bg_box'])

        prompt_duration = "Enter Duration (hours, e.g., 1, 24, 0 for indefinite, '0' to cancel):"
        at(x + 2, duration_prompt_y, prompt_duration, c['y'])
        
        duration_str = (await ainp(x + 2 + len(prompt_duration) + 1, duration_prompt_y)).strip()

        if duration_str == '0':
            _display_notification_wrapper("🐶 WatchDog Mode setup cancelled.", c['y'])
            ui_needs_redraw = True
            return

        try:
            duration_hours = float(duration_str)
            if duration_hours < 0:
                _display_notification_wrapper("❌ Duration cannot be negative. Please try again.", c['R'], c['bg_error'])
                await asyncio.sleep(1)
                continue
            if duration_hours == 0:
                watchdog_duration_seconds = float('inf') 
                _display_notification_wrapper("🐶 WatchDog will run indefinitely.", c['y'])
            else:
                watchdog_duration_seconds = int(duration_hours * 3600)
                _display_notification_wrapper(f"🐶 WatchDog will run for {duration_hours} hours.", c['g'])
            break
        except ValueError:
            _display_notification_wrapper("❌ Invalid duration. Please enter a number (e.g., 1, 24) or '0' to cancel.", c['R'], c['bg_error'])
            await asyncio.sleep(1)
            continue

    confirm_msg = f"Confirm WatchDog Mode: Send ALL balance (minus {TOTAL_FEE_PERCENTAGE*100:.1f}% fees) to {watchdog_target_address[:15]}... {'' if watchdog_duration_seconds == float('inf') else f'for {duration_hours} hours'}?"
    if not (await display_confirmation(confirm_msg)):
        _display_notification_wrapper("🐶 WatchDog Mode setup cancelled.", c['y'])
        ui_needs_redraw = True
        return
    
    watchdog_stop_event.clear()
    watchdog_start_time = datetime.now()
    watchdog_task = asyncio.create_task(watchdog_monitor_loop())
    _display_notification_wrapper("🐶 WatchDog Mode ACTIVATED! Monitoring balance...", c['g'], c['bg_success'])
    await awaitkey()
    ui_needs_redraw = True


async def change_theme_menu():
    global c, current_theme, ui_needs_redraw
    cr = sz()
    cls()
    fill()
    main_box_w = cr[0] - 4
    w, hb = main_box_w, 10
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    box(x, y, w, hb, "🎨 CHANGE THEME")

    at(x + 2, y + 2, f"Current Theme: {current_theme}", c['c'])
    at(x + 2, y + 4, "[1] Dark Mode", c['w'])
    at(x + 2, y + 5, "[2] Light Mode", c['w'])
    at(x + 2, y + 7, "[0] Back", c['y'])
    at(x + 2, y + 8, "Choose theme: ", c['B'] + c['w'])

    choice = (await ainp(x + 18, y + 8)).strip()

    if choice == '1':
        c = DARK_MODE_COLORS
        current_theme = "Dark"
        _display_notification_wrapper("✅ Switched to Dark Mode!", c['g'])
    elif choice == '2':
        c = LIGHT_MODE_COLORS
        current_theme = "Light"
        _display_notification_wrapper("✅ Switched to Light Mode!", c['g'])
    elif choice == '0':
        _display_notification_wrapper("Theme change cancelled.", c['y'])
    else:
        _display_notification_wrapper("❌ Invalid choice.", c['R'], c['bg_error'])
    
    await asyncio.sleep(1)
    ui_needs_redraw = True

async def auto_refresh_data_loop():
    global ui_needs_redraw, last_cursor_pos

    while not stop_flag.is_set():
        if priv is not None:
            try:
                for _ in range(int(AUTO_REFRESH_INTERVAL / 0.1)):
                    if stop_flag.is_set():
                        raise asyncio.CancelledError
                    await asyncio.sleep(0.1)

                if not (auto_send_task and not auto_send_task.done()) and \
                   not (watchdog_task and not watchdog_task.done()) and \
                   not stop_flag.is_set() and \
                   last_cursor_pos is None:
                    
                    await comprehensive_data_refresh()
                    ui_needs_redraw = True
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                _display_notification_wrapper(f"❌ Auto-refresh error: {str(e)}", c['R'])
                logging.error(f"Auto-refresh error: {e}", exc_info=True)
                ui_needs_redraw = True
        else:
            await asyncio.sleep(5)

async def scr_main_display():
    cr = sz()
    cls()
    fill()

    main_box_w = cr[0] - 4
    x_pos_main_box = (cr[0] - main_box_w) // 2

    header_box_h = 6
    header_box_y = 1
    box(x_pos_main_box, header_box_y, main_box_w, header_box_h, "")

    header_line1 = f"💎 OCTRA CLIENT v9.0 STABLE ✨"
    header_line2 = f"Rebuild by https://t.me/dayu_widayadi"

    at(x_pos_main_box + (main_box_w - len(header_line1)) // 2, header_box_y + 1, header_line1, c['B'] + c['w'])
    at(x_pos_main_box + (main_box_w - len(header_line2)) // 2, header_box_y + 2, header_line2, c['y'] + c['B'])

    rpc_display_line = f"🌐 RPC Node: {rpc} 🟢" if rpc else "🌐 RPC Node: Not set 🔴"
    at(x_pos_main_box + (main_box_w - len(rpc_display_line)) // 2, header_box_y + 3, rpc_display_line, c['c'])

    notification_h = 3
    notification_y = cr[1] - notification_h - 1
    notification_x = x_pos_main_box
    box(notification_x, notification_y, main_box_w, notification_h, "📢 Notification")

    commands_h_needed = 23 
    commands_x = x_pos_main_box
    commands_y = notification_y - commands_h_needed - 1
    commands_y = max(header_box_y + header_box_h + 1, commands_y)
    menu(commands_x, commands_y, main_box_w, commands_h_needed)

    explorer_x = x_pos_main_box
    explorer_w = main_box_w
    min_explorer_h = 10
    explorer_y = header_box_y + header_box_h -1 
    explorer_h = commands_y - explorer_y 
    explorer_h = max(min_explorer_h, explorer_h) 

    await expl(explorer_x, explorer_y, explorer_w, explorer_h)

    status_y = cr[1] - 1 
    at(1, status_y, " " * cr[0], c['bg_main']) 

    prompt_text = ""
    if auto_send_task and not auto_send_task.done():
        prompt_text = "⚡ AUTO-SENDING... | Press 0 to stop"
        at(x_pos_main_box + 2, status_y, prompt_text, c['bg_main'] + c['y'] + c['B'])
    elif watchdog_task and not watchdog_task.done():
        prompt_text = "🐶 WATCHDOG MODE ACTIVE | Press 0 to stop"
        at(x_pos_main_box + 2, status_y, prompt_text, c['bg_main'] + c['P'] + c['B'])
    else:
        prompt_text = "⚡ READY | Enter Command Number: "
        at(x_pos_main_box + 2, status_y, prompt_text, c['bg_success'] + c['w'] + c['B'])
    
    print(f"\033[{status_y};{x_pos_main_box + 2 + len(prompt_text)}H", end='', flush=True)


async def menu_input_loop():
    global ui_needs_redraw, auto_send_task, watchdog_task
    while not stop_flag.is_set():
        if ui_needs_redraw:
            await scr_main_display()
            ui_needs_redraw = False
        
        input_y_pos = sz()[1] - 1
        x_pos_main_box = (sz()[0] - (sz()[0] - 4)) // 2
        
        prompt_text_display = ""
        prompt_start_x = 0
        
        if (auto_send_task and not auto_send_task.done()) or (watchdog_task and not watchdog_task.done()):
            prompt_text_display = "STOP (0): "
            prompt_start_x = x_pos_main_box + (sz()[0] - 4) - len(prompt_text_display) - 2
            at(x_pos_main_box + 2, input_y_pos, " " * (sz()[0] - 4), c['bg_main']) 
            at(prompt_start_x, input_y_pos, prompt_text_display, c['bg_main'] + c['R'] + c['B'])
        else:
            prompt_text_display = "⚡ READY | Enter Command Number: "
            prompt_start_x = x_pos_main_box + 2 
            at(x_pos_main_box + 2, input_y_pos, " " * (sz()[0] - 4), c['bg_main']) 
            at(x_pos_main_box + 2, input_y_pos, prompt_text_display, c['bg_success'] + c['w'] + c['B'])
        
        cmd = await ainp(prompt_start_x + len(prompt_text_display), input_y_pos)
        
        if (auto_send_task and not auto_send_task.done() or watchdog_task and not watchdog_task.done()) and cmd == '0':
            if auto_send_task and not auto_send_task.done():
                auto_send_stop_event.set()
                _display_notification_wrapper("Stopping auto-send...", c['y'])
                ui_needs_redraw = True
                await asyncio.sleep(0.5)
            if watchdog_task and not watchdog_task.done():
                watchdog_stop_event.set()
                _display_notification_wrapper("Stopping WatchDog Mode...", c['y'])
                ui_needs_redraw = True
                await asyncio.sleep(0.5)
            continue
        
        if cmd == '1':
            if priv is None:
                _display_notification_wrapper("Wallet not loaded. Please generate or load one first.", c['R'])
                await awaitkey()
                ui_needs_redraw = True
                continue
            await tx()
        elif cmd == '2':
            if priv is None:
                _display_notification_wrapper("Wallet not loaded. Please generate or load one first.", c['R'])
                await awaitkey()
                ui_needs_redraw = True
                continue
            await smart_multi_send()
        elif cmd == '3':
            await start_watchdog_mode()
        elif cmd == '4':
            await exp()
        elif cmd == '5': 
            if priv is None:
                _display_notification_wrapper("Wallet not loaded. Cannot save addresses.", c['R'])
                await awaitkey()
                ui_needs_redraw = True
                continue
            await save_addresses_flow()
        elif cmd == '6':
            await manage_contacts()
        elif cmd == '7': 
            await change_theme_menu()
        elif cmd == '8':
            if priv is None:
                _display_notification_wrapper("Wallet not loaded. Please generate or load one first.", c['R'])
                await awaitkey()
                ui_needs_redraw = True
                continue
            await show_transaction_details()
        elif cmd == '9':
            await show_info_box()
        elif cmd == '':
            if priv is None:
                _display_notification_wrapper("Wallet not loaded. Cannot refresh data.", c['R'])
                await awaitkey()
                ui_needs_redraw = True
                continue
            
            spin_task_manual_refresh = asyncio.create_task(spin_animation_in_notification_box("Refreshing data...", c['y']))
            await comprehensive_data_refresh() 
            if spin_task_manual_refresh and not spin_task_manual_refresh.done():
                spin_task_manual_refresh.cancel()
                try: await spin_task_manual_refresh
                except asyncio.CancelledError: pass
            _display_notification_wrapper("", c['bg_box'])
            
            ui_needs_redraw = True
        elif cmd == '0': 
            if auto_send_task and not auto_send_task.done():
                auto_send_stop_event.set()
                _display_notification_wrapper("Auto-send is running. Attempting to stop before exit...", c['y'])
                await asyncio.sleep(1)
            if watchdog_task and not watchdog_task.done():
                watchdog_stop_event.set()
                _display_notification_wrapper("WatchDog Mode is running. Attempting to stop before exit...", c['y'])
                await asyncio.sleep(1)
            if auto_refresh_task and not auto_refresh_task.done():
                auto_refresh_task.cancel()
                try: await auto_refresh_task
                except asyncio.CancelledError: pass
            stop_flag.set()
        else:
            _display_notification_wrapper(f"❓ Unknown command: '{cmd}'.", c['R'], c['bg_error'])
            await awaitkey()
        ui_needs_redraw = True

def menu(x, y, w, h):
    box(x, y, w, h, "➡️ Commands")
    at(x + 2, y + 3, "[1] Send Transaction", c['w'])
    at(x + 2, y + 5, "[2] Smart Multi-Send", c['w'])
    at(x + 2, y + 7, "[3] WatchDog Mode (NEW!)", c['B'] + c['P'])
    at(x + 2, y + 9, "[4] Export/Manage Keys", c['w'])
    at(x + 2, y + 11, "[5] Save Addresses", c['w']) 
    at(x + 2, y + 13, "[6] Manage Contacts", c['w'])
    at(x + 2, y + 15, "[7] Change Theme", c['B'] + c['c']) 
    at(x + 2, y + 17, "[8] Transaction History", c['w'])
    at(x + 2, y + 19, "[9] Info", c['w']) 
    at(x + 2, y + 21, "[0] Exit Application", c['R'] + c['B']) 
    print(f"\033[{y + h - 1};{x}H{c['bg_box']}{c['w']}└{'─' * (w - 2)}┘{c['bg_main']}")

async def tx():
    global lu, ui_needs_redraw, auto_send_task, auto_send_stop_event, cb, cn
    
    if auto_send_task and not auto_send_task.done():
        _display_notification_wrapper("An auto-send task is already running. Please wait or stop it (press 0) first.", c['R'], c['bg_error'])
        await awaitkey()
        ui_needs_redraw = True
        return

    if watchdog_task and not watchdog_task.done():
        _display_notification_wrapper("WatchDog Mode is active. Please stop it (press 0 from main menu) before manual sending.", c['R'], c['bg_error'])
        await awaitkey()
        ui_needs_redraw = True
        return
    
    if priv is None:
        _display_notification_wrapper("Wallet not loaded. Please generate or load one first.", c['R'])
        await awaitkey()
        ui_needs_redraw = True
        return

    cr = sz()
    cls()
    fill()

    main_box_w = cr[0] - 4
    w, hb = main_box_w, 20
    x = (cr[0] - w) // 2
    y = (cr[1] - hb) // 2
    box(x, y, w, hb, "💸 SEND SINGLE TRANSACTION")
    
    lu = 0 

    n_fetched, b_display = await _fetch_balance_nonce() 
    
    if b_display is None or n_fetched is None:
        _display_notification_wrapper("❌ Failed to load account data! Check RPC connection. Cannot proceed with transaction.", c['R'], c['bg_error'])
        logging.error(f"Failed to load account data for transaction: Balance={b_display}, Nonce={n_fetched}", exc_info=True)
        await awaitkey()
        ui_needs_redraw = True
        return

    at(x + 2, y + 2, f"Your Current Balance: {b_display:,.6f} OCT", c['B'] + c['g'])
    at(x + 2, y + 3, f"Next Nonce: {n_fetched + 1}", c['B'] + c['c'])

    at(x + 2, y + 4, " " * (w - 4), c['bg_box']) 

    at(x + 2, y + 5, "Recipient Address (or contact name, 0 to cancel):", c['y'])
    to = (await ainp(x + 52, y + 5)).strip() 
    
    if to == '0':
        _display_notification_wrapper("Transaction cancelled.", c['y'])
        await asyncio.sleep(1) 
        ui_needs_redraw = True
        return

    resolved_to = to
    amount_y_pos = y + 7 
    
    at(x + 2, y + 6, " " * (w - 4), c['bg_box']) 
    
    if to in contacts:
        resolved_to = contacts[to]
        at(x + 2, y + 6, f"Resolved to: {resolved_to}", c['g'])
        amount_y_pos = y + 8 
    elif not b58.match(to): 
        _display_notification_wrapper("❌ Invalid Octra address format or length!", c['R'], c['bg_error'])
        logging.error(f"Invalid recipient address format: {to}", exc_info=True)
        await awaitkey()
        ui_needs_redraw = True
        return 
    
    at(x + 2, amount_y_pos - 1, " " * (w - 4), c['bg_box']) 

    at(x + 2, amount_y_pos, "Amount to Send (OCT):", c['y'])
    amount_str = (await ainp(x + 35, amount_y_pos)).strip()
    
    if not amount_str:
        _display_notification_wrapper("Amount cannot be empty. Transaction cancelled.", c['R'], c['bg_error'])
        await awaitkey()
        ui_needs_redraw = True
        return

    try:
        a = float(amount_str)
        if a <= 0: 
            raise ValueError("Amount must be positive.")        

        if b_display < a: 
            _display_notification_wrapper("❌ Insufficient balance!", c['R'], c['bg_error'])
            logging.error(f"Insufficient balance for transaction. Has: {b_display}, Needs: {a}", exc_info=True)
            await awaitkey()
            ui_needs_redraw = True
            return
    except ValueError as e: 
        _display_notification_wrapper(f"❌ Invalid amount! {e}", c['R'], c['bg_error'])
        logging.error(f"ValueError in tx (amount): {e}", exc_info=True)
        await awaitkey()
        ui_needs_redraw = True
        return

    num_transactions = 0 

    at(x + 2, amount_y_pos + 1, " " * (w - 4), c['bg_box']) 

    at(x + 2, amount_y_pos + 2, "Number of transactions to Auto-Send (0 or Enter for single send):", c['y'])
    num_tx_str = (await ainp(x + 75, amount_y_pos + 2)).strip()
    try:
        num_transactions = int(num_tx_str)
        if num_transactions <= 0:
            num_transactions = 1 
    except ValueError as e:
        logging.warning(f"Invalid input for num_transactions, defaulting to 1: {e}", exc_info=True)
        num_transactions = 1 

    confirm_message = f"Confirm send {a:,.6f} OCT to {resolved_to}?"
    if num_transactions > 1:
        confirm_message = f"Confirm AUTO-SEND {num_transactions}x of {a:,.6f} OCT to {resolved_to} (Parallel processing)?"
    
    if not (await display_confirmation(confirm_message)): 
        ui_needs_redraw = True
        return
    
    if num_transactions > 1:
        auto_send_stop_event.clear()
        auto_send_task = asyncio.create_task(auto_send_loop_v2(resolved_to, a, num_transactions))
        _display_notification_wrapper("Auto-send started in background. You can navigate other menus.", c['g'], c['bg_success'])
    else: 
        spin_task = asyncio.create_task(spin_animation_in_notification_box("Sending transaction...", c['c']))
        
        n_to_use = n_fetched + 1 
        
        async with nonce_lock:
            nonce_cache[addr] = n_to_use

        tx_obj, tx_hash_local = mk(resolved_to, a, n_to_use) 
        ok, hs_msg, dt, _ = await snd(tx_obj)

        if spin_task and not spin_task.done(): 
            spin_task.cancel()
            try: await spin_task 
            except asyncio.CancelledError: pass

        if ok:
            _display_notification_wrapper(f"✅ Transaction SENT! Hash: {hs_msg[:30]}...", c['g'], c['bg_success'])
            h.append({
                'time': datetime.now(),
                'hash': hs_msg,
                'amt': a,
                'to': resolved_to,
                'type': 'out',
                'ok': True,
                'nonce': n_to_use, 
                'epoch': 0 
            })
            await comprehensive_data_refresh() 
        else:
            _display_notification_wrapper(f"❌ Transaction FAILED! Error: {hs_msg}", c['R'], c['bg_error'])
            logging.error(f"Single transaction FAILED for {resolved_to}. Error: {hs_msg}", exc_info=True)
            if "duplicate" in hs_msg.lower():
                _display_notification_wrapper("⚠️ Duplicate transaction detected. Try sending again with a refreshed nonce.", c['y'])
                await comprehensive_data_refresh() 
        
        await awaitkey() 
    ui_needs_redraw = True


async def main():
    global session, executor, priv, addr, rpc, sk, pub, active_wallet_idx, wallets, auto_refresh_task, nonce_cache

    setup_logging()

    executor = ThreadPoolExecutor(max_workers=1)
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    
    try:
        rpc = 'https://octra.network' 
        wallets_loaded_successfully = ld()

        if priv is None: 
            print(f"{c['y']}[!] No active wallet found. Let's create a new one or add an existing one.{c['r']}")
            initial_priv_key_b64 = input(f"{c['B']}{c['w']}Enter your Private Key (base64) to load, or leave empty to create a new one: {c['r']}").strip()
            
            try:
                current_sk_obj = nacl.signing.SigningKey(base64.b64decode(initial_priv_key_b64)) if initial_priv_key_b64 else nacl.signing.SigningKey.generate()
                current_priv_b64 = base64.b64encode(current_sk_obj.encode()).decode()
                
                current_pub_bytes = current_sk_obj.verify_key.encode()
                current_pub_b64 = base64.b64encode(current_pub_bytes).decode()
                current_addr_standard = get_octra_address_from_pubkey_bytes(current_pub_bytes)

                if not b58.match(current_addr_standard):
                    raise ValueError("The provided private key generates an invalid Octra address format.")

                priv = current_priv_b64
                addr = current_addr_standard
                rpc = 'https://octra.network' 
                sk = current_sk_obj
                pub = current_pub_b64

                exists = False
                for w in wallets:
                    if w.get('priv') == priv:
                        exists = True
                        break
                
                if not exists:
                    wallets.append({'priv': priv, 'addr': addr, 'rpc': rpc})
                    active_wallet_idx = len(wallets) - 1 
                else:
                    for i, w in enumerate(wallets):
                        if w.get('priv') == priv:
                            active_wallet_idx = i
                            _load_and_activate_wallet(wallets[active_wallet_idx]) 
                            break

                _save_all_wallets() 

                if not initial_priv_key_b64: 
                    print(f"{c['g']}✅ New wallet successfully created and saved!{c['r']}")
                    print(f"{c['c']}   Address: {addr}{c['r']}")
                    print(f"{c['y']}   Private Key: {priv}{c['r']} (❗SAVE THIS KEY IN A SAFE PLACE!)")
                    wait()
                else:
                    print(f"{c['g']}✅ Wallet successfully loaded! Address: {addr[:15]}...{c['r']}")

            except Exception as e:
                print(f"{c['R']}❌ Error initializing wallet: {str(e)}. Exiting...{c['r']}")
                logging.critical(f"Error initializing wallet (main function): {e}", exc_info=True)
                sys.exit(1)
        
        if addr is None:
            print(f"{c['R']}❌ Critical error: Wallet address could not be set. Exiting...{c['r']}")
            logging.critical("Critical error: Wallet address could not be set. Exiting...", exc_info=True)
            sys.exit(1)

        load_contacts()

        spin_task_initial_refresh = asyncio.create_task(spin_animation_in_notification_box("Loading initial data...", c['y']))
        await comprehensive_data_refresh() 
        if spin_task_initial_refresh and not spin_task_initial_refresh.done():
            spin_task_initial_refresh.cancel()
            try: await spin_task_initial_refresh
            except asyncio.CancelledError: pass
        _display_notification_wrapper("", c['bg_box'])
        ui_needs_redraw = True

        auto_refresh_task = asyncio.create_task(auto_refresh_data_loop())
        
        try:
            await menu_input_loop()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"\n{c['R']}Unexpected error in main loop: {e}{c['r']}")
            logging.critical(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        if auto_refresh_task and not auto_refresh_task.done():
            auto_refresh_task.cancel()
            try: await auto_refresh_task
            except asyncio.CancelledError: pass
        
        if executor:
            executor.shutdown(wait=False)
        
        if session:
            await session.close()
        
        cls()
        print(f"{c['r']}🚀 Octra client closed. See you again, Bro!{c['r']}")
        os._exit(0)
        
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass 
    except Exception as e:
        print(f"\n{c['R']}Initialization error: {e}{c['r']}")
        logging.critical(f"Initialization error (main execution block): {e}", exc_info=True)
        sys.exit(1)
