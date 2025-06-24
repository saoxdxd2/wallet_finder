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
import tkinter.messagebox as messagebox # For disclaimer

class ProcessManager:
    def __init__(self, log_queue):


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
        # Now, app.py is the single engine that includes generation and checking.
        # We run it as one persistent process.
        target_script = 'finder/app.py' # Relative to project root if running app2.py from root
        # If app2.py is in finder/, then 'app.py'
        # Assuming app2.py is run from the project root like "python finder/app2.py"
        # Or, if packaged, determine path to app.py or the executable.

        # For development:
        cmd = [sys.executable, target_script] # sys.executable is the current python interpreter

        # If this were a packaged app, cmd would be path to the main executable.
        # Example: cmd = [self._get_main_executable_path()]

        while self.running:
            try:
                self._log("Starting main application engine (app.py)...")
                # Use subprocess.PIPE for stdout and stderr to capture logs
                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, # Merge stderr to stdout
                    text=True, # Decode output as text
                    bufsize=1,  # Line buffered
                    universal_newlines=True # Ensure text mode for newlines
                )
                self._log(f"Main application engine started with PID: {self.current_process.pid}")

                # Monitor its output
                self._read_output(self.current_process) # This will block until process ends or error

                if not self.running: # If stop was called during _read_output
                    self._log("Process manager was stopped, terminating engine.")
                    if self.current_process.poll() is None:
                        self.current_process.terminate()
                        try:
                            self.current_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.current_process.kill()
                    break # Exit while loop

                # If process ended by itself and we are still supposed to be running
                if self.running:
                    self._log("Main application engine stopped unexpectedly. Restarting in 5 seconds...")
                    time.sleep(5) # Wait before restarting

            except FileNotFoundError:
                self._log(f"Error: Could not find the application script/executable: {' '.join(cmd)}")
                self._log("Please ensure app.py is in the correct location or the application is built correctly.")
                self.running = False # Stop trying if script not found
                break
            except Exception as e:
                self._log(f"Error in run_cycle: {str(e)}")
                if self.running:
                    self._log("Restarting after error in 5 seconds...")
                    time.sleep(5) # Wait before restarting

        self._log("Run cycle ended.")


    def _read_output(self, proc):
        # Reads output line by line and puts it into the log_queue
        # This runs in the ProcessManager's main thread (run_cycle's thread)
        if proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                if not self.running: # Check if stop was called
                    break
                if line:
                    self._log(line.strip())
            proc.stdout.close()

        # Wait for the process to complete, store return code
        return_code = proc.wait()
        self._log(f"Main application engine exited with code {return_code}.")


    def _log(self, msg):
        self.log_queue.put(msg)

    def start(self):
        if self.running:
            self._log("Process manager already running.")
            return
        self.running = True
        self.current_process = None
        # The run_cycle will be managed by a thread
        self.thread = threading.Thread(target=self.run_cycle, daemon=True)
        self.thread.start()
        self._log("Process manager started.")


    def stop(self):
        if not self.running:
            self._log("Process manager not running.")
            return

        self._log("Stopping process manager...")
        self.running = False # Signal run_cycle and _read_output to stop

        if hasattr(self, 'current_process') and self.current_process and self.current_process.poll() is None:
            self._log(f"Terminating main application engine (PID: {self.current_process.pid})...")
            self.current_process.terminate() # Send SIGTERM
            try:
                self.current_process.wait(timeout=10) # Wait for graceful shutdown
                self._log("Main application engine terminated.")
            except subprocess.TimeoutExpired:
                self._log("Main application engine did not terminate gracefully, killing...")
                self.current_process.kill() # Send SIGKILL
                try:
                    self.current_process.wait(timeout=5)
                    self._log("Main application engine killed.")
                except Exception as e:
                    self._log(f"Error during kill: {e}")
            except Exception as e: # Other errors during termination
                 self._log(f"Error during termination: {e}")

        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=5) # Wait for the run_cycle thread to exit
            if self.thread.is_alive():
                self._log("Process manager thread did not exit cleanly.")

        self._log("Process manager stopped.")

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

        # Show disclaimer before fully starting
        if not self.show_disclaimer():
            self.destroy_app() # Exit if disclaimer not accepted
            return # Stop further initialization

        # If disclaimer accepted, proceed to start processes
        self.process_manager.start()


    def show_disclaimer(self):
        title = "Ethical Use Agreement"
        message = (
            "IMPORTANT: ETHICAL AND LEGAL USE ACKNOWLEDGEMENT\n\n"
            "This software, Crypto Wallet Scanner, is intended strictly for educational and research purposes.\n\n"
            "By proceeding, you acknowledge and agree to the following:\n"
            "1. Lawful Use: You will use this software in compliance with all applicable local, state, national, and international laws and regulations.\n"
            "2. No Unauthorized Access: You will NOT use this software to attempt to access, control, or interfere with any cryptocurrency wallets, funds, or assets for which you do not have explicit, legitimate authorization.\n"
            "3. Educational Purpose Only: You understand that generating and checking wallets should be done on test networks or with your own legitimately obtained mnemonics for learning about blockchain technology and security.\n"
            "4. No Financial Harm: You will not use this software in any way that could cause financial loss or harm to others.\n"
            "5. Responsibility: You are solely responsible for your actions while using this software. The developers disclaim any liability for misuse.\n\n"
            "Do you understand these terms and agree to use this software ethically, legally, and responsibly?"
        )
        # Returns True for Yes, False for No
        return messagebox.askyesno(title, message, parent=self)

    def destroy_app(self):
        """Safely destroy the application window."""
        # Attempt to stop processes if they were somehow started
        if hasattr(self, 'process_manager') and self.process_manager.running:
            self.process_manager.stop()
        self.destroy()


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
