"""Application stylesheet. Kept in one place so the look stays consistent."""

APP_FONT = "Segoe UI"

STYLESHEET = """
* {{
    font-family: '{font}', 'Ubuntu', 'Noto Sans', sans-serif;
    font-size: 13px;
}}

QMainWindow, QDialog {{
    background-color: #f4f6fb;
}}

/* ---------- sidebar ---------- */
QWidget#Sidebar {{
    background-color: #111827;
}}
QLabel#AppTitle {{
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
}}
QLabel#AppSubtitle {{
    color: #9ca3af;
    font-size: 11px;
}}
QPushButton#NavButton {{
    color: #cbd5e1;
    background: transparent;
    text-align: left;
    padding: 10px 14px;
    border: none;
    border-radius: 8px;
    font-size: 13px;
}}
QPushButton#NavButton:hover {{
    background-color: #1f2937;
    color: #ffffff;
}}
QPushButton#NavButton:checked {{
    background-color: #4f46e5;
    color: #ffffff;
    font-weight: 600;
}}
QLabel#SidebarFooter {{
    color: #6b7280;
    font-size: 10px;
}}

/* ---------- headers ---------- */
QLabel#PageTitle {{
    font-size: 20px;
    font-weight: 700;
    color: #111827;
}}
QLabel#PageSubtitle {{
    font-size: 12px;
    color: #6b7280;
}}
QGroupBox {{
    font-weight: 600;
    color: #374151;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 8px;
    background-color: #ffffff;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}}

/* ---------- cards ---------- */
QFrame#StatCard {{
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
}}
QLabel#CardTitle {{
    color: #6b7280;
    font-size: 11px;
    font-weight: 600;
}}
QLabel#CardValue {{
    font-size: 21px;
    font-weight: 700;
}}
QLabel#CardSub {{
    color: #9ca3af;
    font-size: 10px;
}}

/* ---------- inputs & buttons ---------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    padding: 6px 8px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    background: #ffffff;
    selection-background-color: #c7d2fe;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid #4f46e5;
}}

QPushButton {{
    padding: 7px 16px;
    border-radius: 7px;
    border: 1px solid #d1d5db;
    background-color: #ffffff;
    color: #374151;
}}
QPushButton:hover {{ background-color: #f3f4f6; }}
QPushButton:pressed {{ background-color: #e5e7eb; }}
QPushButton:disabled {{ color: #9ca3af; background-color: #f3f4f6; }}

QPushButton#PrimaryButton {{
    background-color: #4f46e5;
    border: none;
    color: #ffffff;
    font-weight: 600;
    padding: 8px 20px;
}}
QPushButton#PrimaryButton:hover {{ background-color: #4338ca; }}
QPushButton#PrimaryButton:pressed {{ background-color: #3730a3; }}
QPushButton#PrimaryButton:disabled {{ background-color: #a5b4fc; }}

/* ---------- progress ---------- */
QProgressBar {{
    border: none;
    border-radius: 7px;
    background-color: #e5e7eb;
    text-align: center;
    font-weight: 600;
    color: #111827;
    height: 18px;
}}
QProgressBar::chunk {{
    border-radius: 7px;
    background-color: #4f46e5;
}}

/* ---------- tables ---------- */
QTableView, QTableWidget {{
    background-color: #ffffff;
    alternate-background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    gridline-color: #eef0f4;
}}
QHeaderView::section {{
    background-color: #f3f4f6;
    color: #374151;
    font-weight: 600;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #e5e7eb;
}}
QTableCornerButton::section {{ background-color: #f3f4f6; border: none; }}

/* ---------- logs ---------- */
QPlainTextEdit#LogView {{
    background-color: #0f172a;
    color: #e2e8f0;
    border: none;
    border-radius: 8px;
    font-family: 'Consolas', 'DejaVu Sans Mono', monospace;
    font-size: 12px;
}}

/* ---------- misc ---------- */
QStatusBar {{ background-color: #eef1f7; color: #4b5563; }}
QToolTip {{ background-color: #111827; color: #f9fafb; border: none; padding: 4px; }}
QCheckBox {{ spacing: 6px; }}
QSplitter::handle {{ background-color: #e5e7eb; }}
""".format(font=APP_FONT)
