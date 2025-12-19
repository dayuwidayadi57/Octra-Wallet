🌌 Octra Wallet CLI Client V9 (ALPHA)
A robust and interactive Command-Line Interface (CLI) client for the Octra Network, rebuilt with enhanced user experience and stability in mind. This client allows you to manage your Octra wallets, automate transactions, and interact with the network directly from your terminal with a professional TUI (Terminal User Interface).

📑 Table of Contents
 * Features
 * How It Works
 * Getting Started
   * Prerequisites
   * Installation
 * Usage
   * Main Dashboard
   * Commands
 * Automation Features
 * Wallet File (wallet.json)
 * License

✨ Features
This Octra CLI Client provides a comprehensive set of features for managing your assets:
 * Interactive Dashboard: Real-time display of your wallet address, balance, nonce, and recent transactions with dynamic terminal box layouts.

 * Multi-Wallet Management: Load, add, and seamlessly switch between multiple Octra accounts securely stored in wallet.json.

 * Standard Octra Address Format: Fully compatible with the official Octra standard: oct + Base58(SHA256(PublicKey)).

 * Asynchronous Performance: Built on asyncio for non-blocking UI updates and background network requests.

 * Dynamic UI Themes: Supports Dark Mode and Light Mode with high-fidelity ANSI color schemes.

 * Contact Management: Integrated address book (contacts.json) to store and manage frequently used Octra addresses.

 * Security Focused: Cryptographic operations (SigningKey & Ed25519) are handled locally using PyNaCl.

 * Error Logging: Background errors are silently captured in octra_error.log to maintain a clean workspace.

📢 How It Works
The client operates asynchronously using Python's asyncio and aiohttp libraries. It maintains a persistent session with the Octra RPC node, ensuring that balance refreshes and transaction broadcasts do not freeze the interface.
Sensitive information, specifically your Private Keys, are stored locally in your directory. The client uses a thread-safe executor for user inputs to allow background animations (like spinners) to run simultaneously.

♻️ Getting Started
Prerequisites
 * Python 3.8 or higher
 * pip (Python package installer)
 * A terminal that supports ANSI escape codes (e.g., Windows Terminal, iTerm2, or Linux Bash).
📲 Installation
 * Clone the repository:
   git clone https://github.com/dayuwidayadi57/Octra-Wallet
cd Octra-Wallet

 * Install dependencies:
   pip install aiohttp pynacl

 * Run the client:
   python cli9.py

🧐 Main Dashboard
The dashboard provides a centralized view of your active session:
 ┌────────────────────────────────────┤ ◎ WALLET DASHBOARD ├─────────────────────────────────────┐
 │                                                                                               │
 │ 🔑 Address: octD4RxTBurxxxxxxxxxxxxxxxxxxxxxxxxxxxx                                   │
 │ 💰 Balance: 100.000000 OCT                                                                    │
 │ #️⃣ Nonce:   5                                                                                 │
 │ 🌐 RPC:     https://octra.network                                                             │
 │───────────────────────────────────────────────────────────────────────────────────────────────│
 │ 📜 RECENT TRANSACTIONS:                                                                       │
 │ ... [Time] [Type] [Amount] [Status] ...                                                       │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘

🐍 Commands
Enter the corresponding number for each action:
 * [1] Send Transaction: Send OCT to a single recipient or a saved contact.
 * [2] History Transaction: Detailed view of your recent blockchain activity.
 * [3] Multi-Send: Batch transactions to multiple recipients simultaneously.
 * [4] Export/Manage Keys: Add new accounts, switch between wallets, or view your private keys.
 * [5] Clear Local History: Reset the transaction display in your current session.
 * [6] Manage Contacts: Add or remove aliases from your contacts.json.
 * [0] Exit Application: Safely terminate the session.

⚙️ Automation Features
Unique to Version 9, this client includes high-level automation:
 * Auto-Send: Set up automated recurring transfers at specific intervals.
 * Watchdog Mode: Monitor a target address for a specific duration, providing real-time status updates in the notification box.
 * Auto-Refresh: Background task that keeps your balance and nonce updated every 30 seconds without user intervention.

📁 Wallet File (wallet.json)
Your wallet data is stored in wallet.json.
Structure Example:
{
  "wallets": [
    {
      "priv": "BASE64_PRIVATE_KEY",
      "addr": "oct_ADDRESS",
      "rpc": "https://octra.network"
    }
  ],
  "active_wallet_idx": 0
}

⚠️ WARNING: Never share your wallet.json or octra_error.log. These files contain sensitive cryptographic data.
✅ License
This project is open-source and available under the MIT License.
Developed with 💜 by dayuwidayadi