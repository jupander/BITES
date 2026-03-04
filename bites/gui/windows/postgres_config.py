from PySide6.QtWidgets import (
     QPushButton, QDialog, QFormLayout, QLineEdit,
    QVBoxLayout, QHBoxLayout, QSpinBox, QCheckBox, QMessageBox, QApplication, QToolButton
)
from pathlib import Path

from bites.gui.mixins import DirtyTrackingMixin, GeometryPersistenceMixin
import bites.gui.gui_utils as gui_utils


class ConfigDialog(GeometryPersistenceMixin, DirtyTrackingMixin, QDialog):
    GEOMETRY_KEY = "postgres_config"
    DEFAULT_GEOMETRY = None  # Use Qt's auto-sizing

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PostgreSQL configuration")
        self.setModal(True) # Make it modal to block interaction with the main window

        self.config = config
        pg_config = config["postgresql_config"]
        self.previous_password = pg_config.get("password", "")

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.user_edit = QLineEdit(pg_config.get("user", ""))
        self.user_edit.setToolTip(
            'PostgreSQL username for connecting to the database.\n\n'
            'This is the database user that has permissions to read/write\n'
            'to the optimization study tables.'
        )
        self.use_pgpass_checkbox = QCheckBox("Password file in use (.pgpass)")
        self.use_pgpass_checkbox.setToolTip(
            'Use a PostgreSQL password file (.pgpass) instead of storing\n'
            'the password in the config file.\n\n'
            'The .pgpass file is located at:\n'
            '• Windows: %APPDATA%\\postgresql\\pgpass.conf\n'
            '  (type "%APPDATA%" in File Explorer address bar and press Enter)\n'
            '• Linux/macOS: ~/.pgpass\n\n'
            'This is more secure for shared or version-controlled configs.'
        )
        self.password_edit = QLineEdit(pg_config.get("password", ""))
        self.password_edit.setToolTip(
            'PostgreSQL password for the database user.\n\n'
            'Note: This password is stored in the configuration file.\n'
            'For better security, consider using a .pgpass file instead.'
        )
        #self.password_edit.setEchoMode(QLineEdit.Password)

        self.host_edit = QLineEdit(pg_config.get("host", ""))
        self.host_edit.setToolTip(
            'Hostname or IP address of the PostgreSQL server.\n\n'
            'Examples:\n'
            '• localhost (for local database)\n'
            '• 192.168.1.100 (IP address)\n'
            '• db.example.com (hostname)'
        )
        self.port_edit = QSpinBox()
        self.port_edit.setRange(1, 65535)
        self.port_edit.setValue(pg_config.get("port", 5432))
        self.port_edit.setStyleSheet(
            "QSpinBox::up-button, QSpinBox::down-button { width: 16px; }"
        )
        self.port_edit.setToolTip(
            'PostgreSQL server port number.\n\n'
            'Default is 5432 (standard PostgreSQL port).\n'
            'Only change if your database server uses a different port.'
        )
        self.db_name_edit = QLineEdit(pg_config.get("db_name", ""))
        self.db_name_edit.setToolTip(
            'Name of the PostgreSQL database to connect to for administrative operations.\n\n'
            'The actual study data will be stored in a separate database named after your study_name.'
        )

        form_layout.addRow(self.use_pgpass_checkbox)
        form_layout.addRow("User:", self.user_edit)
        form_layout.addRow("Password:", self.password_edit)
        form_layout.addRow("Host:", self.host_edit)
        form_layout.addRow("Port:", self.port_edit)
        form_layout.addRow("Database name:", self.db_name_edit)

        layout.addLayout(form_layout)

        # Button row
        button_layout = QHBoxLayout()

        # Guide button
        guide_button = QToolButton()
        guide_button.setText("Guide")
        guide_button.setToolTip(
            "Open User's Guide\n\n"
            "Opens the PDF documentation in your\n"
            "system's default PDF viewer."
        )
        guide_button.clicked.connect(lambda: gui_utils.open_users_guide(self, page=37))
        button_layout.addWidget(guide_button)

        button_layout.addStretch()

        self.test_button = QPushButton("Test Connection")
        self.test_button.setToolTip(
            'Test the database connection with the current settings.\n\n'
            'This will attempt to connect to the PostgreSQL server\n'
            'and verify that the credentials are correct.'
        )
        self.test_button.clicked.connect(self.test_connection)
        button_layout.addWidget(self.test_button)

        self.save_button = QPushButton("Save")
        self.save_button.setToolTip(
            'Validate and save the PostgreSQL configuration.\n\n'
            'This will:\n'
            '1. Check that required fields are filled in\n'
            '2. Build a connection URL from your settings\n'
            '3. Save the configuration\n\n'
            'Use "Test connection" to verify your settings work.'
        )
        self.save_button.clicked.connect(self.validate_and_accept)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.use_pgpass_checkbox.toggled.connect(self.toggle_pgpass_mode)

        if self.password_edit.text() == "PGPASS_ENABLED":
            self.use_pgpass_checkbox.setChecked(True)
            self.password_edit.setDisabled(True)

        # Apply screen-aware sizing
        self._apply_screen_aware_sizing()

        # Initialize dirty tracking after UI is built
        self._init_dirty_tracking()
        self._connect_dirty_signals([
            self.user_edit, self.password_edit, self.host_edit,
            self.port_edit, self.db_name_edit, self.use_pgpass_checkbox
        ])

        # Initialize geometry persistence (restores saved position/size)
        self._init_geometry_persistence()

    def toggle_pgpass_mode(self, checked) -> None:
        if checked:
            self.previous_password = self.password_edit.text()
            self.password_edit.setText("PGPASS_ENABLED")
            self.password_edit.setDisabled(True)
        else:
            self.password_edit.setDisabled(False)
            self.password_edit.setText(self.previous_password if self.previous_password != "PGPASS_ENABLED" else "")

    def _apply_screen_aware_sizing(self) -> None:
        """Set window size constraints based on available screen space."""
        screen = QApplication.primaryScreen().availableGeometry()

        # Set minimum size so window remains usable
        self.setMinimumSize(200, 220)

        # Set maximum height to 50% of screen height (small config dialog)
        self.setMaximumHeight(int(screen.height() * 0.50))

        # Auto-size to content, then apply constraints
        self.adjustSize()

    def validate_and_accept(self) -> None:
        user = self.user_edit.text().strip()
        password = self.password_edit.text().strip()
        host = self.host_edit.text().strip()
        port = self.port_edit.value()
        db_name = self.db_name_edit.text().strip()
        use_pgpass = self.use_pgpass_checkbox.isChecked()

        # Validation: All fields must be non-empty
        if not all([user, host, db_name]):
            QMessageBox.critical(self, "Validation Error", "User, Host, and Database Name cannot be empty.")
            return

        if not use_pgpass and not password:
            QMessageBox.critical(self, "Validation Error", "Password cannot be empty if .pgpass is not used.")
            return

        # Try building a postgres URL
        try:
            url = f"postgresql://{user}:{'' if use_pgpass else password}@{host}:{port}/{db_name}"
            # If needed, parse the URL here using e.g. `sqlalchemy.engine.make_url(url)`
        except Exception as e:
            QMessageBox.critical(self, "URL Error", f"Failed to build connection URL: {e}")
            return

        # Save config and close
        self.config["postgresql_config"] = {
            "user": user,
            "password": "PGPASS_ENABLED" if use_pgpass else password,
            "host": host,
            "port": port,
            "db_name": db_name
        }
        #save_config(self.config)
        self._reset_dirty()
        super().accept()

    def test_connection(self) -> None:
        """Test the database connection with current settings."""
        host = self.host_edit.text().strip()
        port = self.port_edit.value()
        user = self.user_edit.text().strip()
        password = self.password_edit.text().strip()
        db_name = self.db_name_edit.text().strip()
        use_pgpass = self.use_pgpass_checkbox.isChecked()

        if not all([host, user, db_name]):
            QMessageBox.warning(self, "Missing Fields", "Please fill in Host, User, and Database Name.")
            return

        # Disable button and show testing state
        self.test_button.setEnabled(False)
        self.test_button.setText("Testing...")
        QApplication.processEvents()

        try:
            import psycopg2
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=None if use_pgpass else password,
                dbname=db_name,
                connect_timeout=5
            )
            conn.close()
            QMessageBox.information(self, "Success", "Connection successful!")
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency", "psycopg2 is not installed.\n\nInstall it with: pip install psycopg2-binary")
        except Exception as e:
            QMessageBox.warning(self, "Connection Failed", str(e))
        finally:
            self.test_button.setEnabled(True)
            self.test_button.setText("Test connection")

    def closeEvent(self, event) -> None:
        """Handle window close - prompt if unsaved changes."""
        if self._prompt_unsaved_changes(event):
            # GeometryPersistenceMixin.closeEvent saves geometry automatically
            super().closeEvent(event)

