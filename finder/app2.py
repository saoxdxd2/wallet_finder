import time
import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import queue
import json
import os
from datetime import datetime
import tempfile
import sys

class ProcessManager:
    def __init__(self, log_queue):
        self.log_queue = log_queue
        self.running = False


    def _read_embedded(self, name):
        """Read embedded binary from package"""
        try:
            # PyInstaller bundle path
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = os.path.abspath(".")
            
        file_path = os.path.join(base_path, name)
        with open(file_path, "rb") as f:
            return f.read()
        
    def run_cycle(self):
        while self.running:
            try:
                self._log("Starting generator...")
                go_proc = subprocess.Popen(['go', 'run', 'app.go'], 
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                self._monitor_process(go_proc, 60)
                
                self._log("Starting checker...")
                py_proc = subprocess.Popen(['python', 'app.py'], 
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT)
                self._monitor_process(py_proc)
            except Exception as e:
                self._log(f"Error: {str(e)}")

    def _monitor_process(self, proc, timeout=None):
        start = time.time()
        threading.Thread(target=self._read_output, args=(proc,), daemon=True).start()
        
        while proc.poll() is None and self.running:
            if timeout and time.time() - start > timeout:
                proc.terminate()
                break
            time.sleep(1)
        
        if proc.poll() is None:
            proc.terminate()
        proc.wait()

    def _read_output(self, proc):
        for line in iter(proc.stdout.readline, b''):
            self._log(line.decode().strip())

    def _log(self, msg):
        self.log_queue.put(msg)

    def start(self):
        self.running = True
        threading.Thread(target=self.run_cycle, daemon=True).start()

    def stop(self):
        self.running = False

class WalletGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crypto Wallet Scanner")
        self.geometry("1400x900")
        self.last_inode = None
        self.file_version = 0
        
        # UI setup
        main_frame = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Log section
        log_frame = ttk.Frame(main_frame)
        self.log_view = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15)
        self.log_view.pack(fill=tk.BOTH, expand=True)
        main_frame.add(log_frame)

        # Results section
        results_frame = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        
        # Checked wallets
        checked_pane = ttk.Frame(results_frame)
        self.checked_tree = ttk.Treeview(checked_pane, 
            columns=('Time', 'Mnemonic', 'BTC', 'ETH'), 
            show='headings'
        )
        self.checked_tree.heading('Time', text='Time')
        self.checked_tree.heading('Mnemonic', text='Mnemonic (First 15 chars)')
        self.checked_tree.heading('BTC', text='BTC Balance')
        self.checked_tree.heading('ETH', text='ETH Balance')
        
        vsb = ttk.Scrollbar(checked_pane, orient="vertical", command=self.checked_tree.yview)
        hsb = ttk.Scrollbar(checked_pane, orient="horizontal", command=self.checked_tree.xview)
        self.checked_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.checked_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        results_frame.add(checked_pane)

        # Stats panel
        stats_frame = ttk.Frame(results_frame, width=300)
        stats_frame.pack_propagate(False)
        self.total_label = ttk.Label(stats_frame, text="Total Checked: 0", font=('Arial', 14))
        self.total_label.pack(pady=10)
        self.found_label = ttk.Label(stats_frame, text="Found with Balance: 0", font=('Arial', 12))
        self.found_label.pack(pady=10)
        results_frame.add(stats_frame)

        main_frame.add(results_frame)

        # Initialize
        self.log_queue = queue.Queue()
        self.process_manager = ProcessManager(self.log_queue)
        self.last_pos = 0
        self.total_checked = 0
        self.total_found = 0
        self.seen_entries = set()
        
        # Start processes
        self.process_manager.start()
        self.after(100, self.update_log)
        self.after(500, self.update_checked)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_log(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_view.insert(tk.END, msg + "\n")
            self.log_view.see(tk.END)
        self.after(100, self.update_log)

    def update_checked(self):
        try:
            if os.path.exists('checked.txt'):
            # Get current file stats
                current_stat = os.stat('checked.txt')
                current_inode = current_stat.st_ino
                current_size = current_stat.st_size

            # Reset position if file has been rotated
                if hasattr(self, 'last_inode'):
                    if current_inode != self.last_inode or current_size < self.last_pos:
                        self.last_pos = 0
                    
                self.last_inode = current_inode
            
                with open('checked.txt', 'r') as f:
                # Reset position if file is smaller than last position
                    if current_size < self.last_pos:
                        self.last_pos = 0
                
                    f.seek(self.last_pos)
                    new_lines = f.readlines()
                    self.last_pos = f.tell()

                    for line in new_lines:
                        if line.strip() and line not in self.seen_entries:
                            self.process_entry(line)
                            self.seen_entries.add(line)
        except Exception as e:
            pass
        finally:
            self.after(500, self.update_checked)

    def process_entry(self, line):
        try:
            parts = line.strip().split('|', 2)
            if len(parts) != 3:
                return
                
            timestamp, mnemonic, balance_str = parts
            balances = json.loads(balance_str)
            
            btc = balances.get('BITCOIN', 0)
            eth = balances.get('ETHEREUM', 0)
            
            self.total_checked += 1
            if btc > 0 or eth > 0:
                self.total_found += 1
            
            dt = datetime.fromisoformat(timestamp)
            display_time = dt.strftime("%m/%d %H:%M:%S")
            display_mnemonic = mnemonic[:15] + '...' if len(mnemonic) > 15 else mnemonic
            
            self.checked_tree.insert('', 'end', values=(
                display_time,
                display_mnemonic,
                f"{btc:.8f}",
                f"{eth:.8f}"
            ))
            
            if len(self.checked_tree.get_children()) > 200:
                self.checked_tree.delete(self.checked_tree.get_children()[0])
            
            self.total_label.config(text=f"Total Checked: {self.total_checked}")
            self.found_label.config(text=f"Found with Balance: {self.total_found}")
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            pass

    def on_closing(self):
        self.process_manager.stop()
        self.destroy()

if __name__ == "__main__":
    app = WalletGUI()
    app.mainloop()
