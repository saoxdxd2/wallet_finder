import sys
import os
import time
import json
from datetime import datetime
import subprocess
import threading

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QVBoxLayout, QHBoxLayout, QWidget,
    QMessageBox, QTreeWidget, QTreeWidgetItem, QHeaderView, QLabel, QSplitter
)
from PySide6.QtCore import Qt, QTimer, Slot, Signal, QObject
from PySide6.QtGui import QFont, QFontMetrics

# Modular imports - ensure these are accessible
import finder.config as config
import finder.logger_setup as logger_setup
import logging

module_logger = logging.getLogger(__name__)

class ProcessManager(QObject):
    log_message = Signal(str)
    process_finished = Signal(int)
    process_error = Signal(str)

    def __init__(self, target_script_name="app.py"):
        super().__init__()
        self.target_script_name = target_script_name
        self.running = False
        self.process = None
        self._thread = None
        module_logger.debug("ProcessManager initialized.")

    def _get_script_path(self):
        # Assumes this GUI script (app_pyside.py or app2.py) is in the 'finder' directory,
        # and the backend script (app.py) is also in the 'finder' directory.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, self.target_script_name)
        if os.path.exists(script_path):
            return script_path

        module_logger.error(f"Backend script {self.target_script_name} not found at {script_path}")
        return None

    def _run_target(self):
        script_to_run = self._get_script_path()
        if not script_to_run:
            err_msg = f"Backend script '{self.target_script_name}' could not be located."
            self.log_message.emit(err_msg); self.process_error.emit(err_msg); self.running = False; return

        try:
            self.log_message.emit(f"Starting backend: {sys.executable} {script_to_run}")
            self.process = subprocess.Popen(
                [sys.executable, script_to_run],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True,
                cwd=os.path.dirname(script_to_run)
            )
            self.log_message.emit(f"Backend process (PID: {self.process.pid}) started. Monitoring output...")
            if self.process.stdout:
                for line in iter(self.process.stdout.readline, ''):
                    if not self.running: break
                    if line: self.log_message.emit(line.strip())
                self.process.stdout.close()
            return_code = self.process.wait()
            self.process_finished.emit(return_code)
        except Exception as e:
            err_msg = f"Error in ProcessManager _run_target: {str(e)}"
            self.log_message.emit(err_msg); self.process_error.emit(err_msg)
            module_logger.error(err_msg, exc_info=True)
        finally:
            self.running = False
            module_logger.info(f"ProcessManager thread finished for {script_to_run if script_to_run else 'unknown script'}.")

    def start(self):
        if self.running: self.log_message.emit("Process manager: Backend already running."); return
        self.running = True
        self._thread = threading.Thread(target=self._run_target, daemon=True); self._thread.start()
        self.log_message.emit("Process manager: Backend start initiated.")

    def stop(self):
        if not self.running and not (self.process and self.process.poll() is None) :
            self.log_message.emit("Process manager: Backend not running or already stopped."); self.running = False; return

        pid_info = self.process.pid if self.process else 'N/A'
        self.log_message.emit(f"Process manager: Stopping backend (PID: {pid_info})...")
        self.running = False
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate(); self.log_message.emit("Process manager: Sent SIGTERM.")
                try: self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.log_message.emit("Process manager: Backend kill timeout, sending SIGKILL."); self.process.kill(); self.process.wait(timeout=1)
            except Exception as e: module_logger.error(f"Error during termination: {e}", exc_info=True)
        elif self.process: self.log_message.emit(f"Process manager: Backend already exited (code {self.process.returncode}).")
        else: self.log_message.emit("Process manager: No backend process to stop.")
        if self._thread and self._thread.is_alive(): self._thread.join(timeout=1)
        self.process = None; module_logger.info("ProcessManager stop sequence complete.")

class WalletGUI(QMainWindow):
    new_checked_line_signal = Signal(str)
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.GUI_WINDOW_TITLE + " (PySide6)"); self.setGeometry(100, 100, 1200, 800)
        module_logger.info("WalletGUI initializing..."); self._init_ui(); self.disclaimer_accepted = False
        if not self.show_disclaimer():
            module_logger.warning("Disclaimer not accepted. Exiting."); QTimer.singleShot(0, self.close); return
        self.disclaimer_accepted = True; module_logger.info("Disclaimer accepted.")
        self.process_manager = ProcessManager(); self.process_manager.log_message.connect(self.append_to_log_view)
        self.process_manager.process_finished.connect(self.on_backend_finished)
        self.process_manager.process_error.connect(self.on_backend_error); self.process_manager.start()
        self.checked_file_monitor_thread = None; self.stop_file_monitoring = threading.Event()
        self.start_checked_file_monitor(); self.new_checked_line_signal.connect(self.process_checked_entry_from_signal)
        self.total_checked_count = 0; self.total_found_count = 0; module_logger.info("WalletGUI init complete.")

    def _init_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget); v_layout = QVBoxLayout(main_widget)
        top_splitter = QSplitter(Qt.Horizontal); self.checked_tree = QTreeWidget(); self.checked_tree.setColumnCount(4)
        self.checked_tree.setHeaderLabels(['Time', 'Mnemonic (Elided)', 'BTC', 'ETH'])
        header = self.checked_tree.header(); header.setSectionResizeMode(QHeaderView.Stretch); header.setStretchLastSection(False)
        header.resizeSection(0, 100); header.resizeSection(1, 350); header.resizeSection(2, 100); header.resizeSection(3, 100)
        self.checked_tree.setAlternatingRowColors(True); top_splitter.addWidget(self.checked_tree)
        stats_widget = QWidget(); stats_layout = QVBoxLayout(stats_widget); stats_layout.setAlignment(Qt.AlignTop)
        self.total_label = QLabel("Total Checked: 0"); self.found_label = QLabel("Found with Balance: 0")
        stats_layout.addWidget(self.total_label); stats_layout.addWidget(self.found_label)
        stats_widget.setFixedWidth(200); top_splitter.addWidget(stats_widget); top_splitter.setSizes([800, 200])
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True); self.log_view.setFont(QFont("Courier", 9))
        main_splitter = QSplitter(Qt.Vertical); main_splitter.addWidget(top_splitter); main_splitter.addWidget(self.log_view)
        main_splitter.setSizes([500, 300]); v_layout.addWidget(main_splitter); module_logger.debug("UI initialized.")

    def show_disclaimer(self):
        title = "Ethical Use Agreement"
        full_message = ("IMPORTANT: ETHICAL AND LEGAL USE ACKNOWLEDGEMENT\n\n"
            "This software, Crypto Wallet Scanner, is intended strictly for educational and research purposes.\n\n"
            "By proceeding, you acknowledge and agree to the following:\n"
            "1. Lawful Use: You will use this software in compliance with all applicable laws.\n"
            "2. No Unauthorized Access: You will NOT use this software to attempt to access assets for which you do not have explicit authorization.\n"
            "3. Educational Purpose Only: Understand that this is for learning about blockchain technology and security.\n"
            "4. No Financial Harm: You will not use this software in any way that could cause financial harm.\n"
            "5. Responsibility: You are solely responsible for your actions. Developers disclaim liability for misuse.\n\n"
            "Do you understand these terms and agree to use this software ethically, legally, and responsibly?")
        reply = QMessageBox.question(self, title, full_message, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        return reply == QMessageBox.StandardButton.Yes

    @Slot(str)
    def append_to_log_view(self, text): self.log_view.append(text); self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())
    @Slot(int)
    def on_backend_finished(self, exit_code): self.append_to_log_view(f"--- Backend process finished (code: {exit_code}) ---")
    @Slot(str)
    def on_backend_error(self, error_message):
        self.append_to_log_view(f"--- Backend Error: {error_message} ---");
        if self.isVisible() and QApplication.instance() and QApplication.instance().applicationState() == Qt.ApplicationActive:
            QMessageBox.critical(self, "Backend Error", error_message)

    def _monitor_checked_file(self):
        module_logger.info("Checked file monitor thread started."); last_pos = 0; seen_entries = set()
        checked_file_path = config.CHECKED_WALLETS_FILE_PATH
        while not self.stop_file_monitoring.is_set():
            try:
                if not os.path.exists(checked_file_path): time.sleep(1); continue
                with open(checked_file_path, 'r', encoding='utf-8') as f:
                    current_file_size = os.path.getsize(checked_file_path)
                    if current_file_size < last_pos:
                        module_logger.info("Checked file appears smaller; resetting read position.")
                        last_pos = 0
                    f.seek(last_pos); new_lines = f.readlines(); last_pos = f.tell()
                for line in new_lines:
                    line_stripped = line.strip()
                    if line_stripped and line_stripped not in seen_entries:
                        self.new_checked_line_signal.emit(line_stripped); seen_entries.add(line_stripped)
                        if len(seen_entries) > 10000: seen_entries = set(list(seen_entries)[-5000:])
                time.sleep(0.5)
            except Exception as e: module_logger.error(f"Error in checked file monitor: {e}", exc_info=False); time.sleep(2)
        module_logger.info("Checked file monitor thread stopped.")

    def start_checked_file_monitor(self):
        if self.checked_file_monitor_thread and self.checked_file_monitor_thread.is_alive(): return
        self.stop_file_monitoring.clear(); self.checked_file_monitor_thread = threading.Thread(target=self._monitor_checked_file, daemon=True); self.checked_file_monitor_thread.start()

    @Slot(str)
    def process_checked_entry_from_signal(self, line_str):
        try:
            parts = line_str.strip().split('|', 3);
            if len(parts) != 4: return
            _, ts_str, mne, bal_json = parts; bals = json.loads(bal_json)
            
            btc_bal_str = f"{bals.get(config.Bip44Coins.BITCOIN.name, 0.0):.8f}"
            eth_bal_str = f"{bals.get(config.Bip44Coins.ETHEREUM.name, 0.0):.8f}"
            
            fm = QFontMetrics(self.checked_tree.font()); el_mne = fm.elidedText(mne, Qt.ElideRight, self.checked_tree.columnWidth(1)-20)
            try: disp_time = datetime.fromisoformat(ts_str).strftime("%H:%M:%S")
            except ValueError: disp_time = ts_str
            item = QTreeWidgetItem([disp_time, el_mne, btc_bal_str, eth_bal_str]); item.setData(1, Qt.UserRole, mne)
            self.checked_tree.addTopLevelItem(item)
            if self.checked_tree.topLevelItemCount() > (config.GUI_MAX_TREE_ITEMS if hasattr(config, 'GUI_MAX_TREE_ITEMS') else 300):
                self.checked_tree.takeTopLevelItem(0)
            self.checked_tree.scrollToBottom()
            self.total_checked_count += 1
            if bals.get(config.Bip44Coins.BITCOIN.name,0.0)>0 or bals.get(config.Bip44Coins.ETHEREUM.name,0.0)>0 or bals.get("USDT",0.0)>0: self.total_found_count +=1
            self.total_label.setText(f"Total Checked: {self.total_checked_count}"); self.found_label.setText(f"Found with Balance: {self.total_found_count}")
        except Exception as e: module_logger.error(f"Error processing GUI entry '{line_str[:60]}...': {e}", exc_info=True)

    def closeEvent(self, event):
        module_logger.info("Close event. Shutting down application and backend...");
        self.stop_file_monitoring.set()
        if self.process_manager: self.process_manager.stop()
        if self.checked_file_monitor_thread and self.checked_file_monitor_thread.is_alive():
            self.checked_file_monitor_thread.join(timeout=1)
        event.accept()

if __name__ == '__main__':
    logger_setup.setup_logging();
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    if hasattr(config, 'LOG_DIR') and config.LOG_DIR and not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR, exist_ok=True)

    app = QApplication(sys.argv)
    main_window = WalletGUI()

    if not main_window.disclaimer_accepted:
        module_logger.info("Disclaimer not accepted during __init__, exiting application via __main__ check.")
        sys.exit(0)

    main_window.show()
    sys.exit(app.exec())

```
