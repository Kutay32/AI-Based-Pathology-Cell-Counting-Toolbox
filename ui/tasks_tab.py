"""
Tasks Tab UI component for the Pathology Cell Counting Toolbox.

This module provides a UI component for displaying and managing analysis tasks
in a clear and organized manner.
"""

import os
import sys
import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QListWidget, QListWidgetItem, QSplitter, QGroupBox, 
    QGridLayout, QProgressBar, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QColor

class TaskItem(QListWidgetItem):
    """
    Custom list widget item for representing a task.
    """
    def __init__(self, task_id, task_name, status, timestamp, details=None):
        super().__init__()
        self.task_id = task_id
        self.task_name = task_name
        self.status = status
        self.timestamp = timestamp
        self.details = details or {}
        
        # Set display text
        self.setText(f"{task_name} - {status}")
        
        # Set tooltip with more details
        self.setToolTip(f"Task: {task_name}\nStatus: {status}\nStarted: {timestamp}")
        
        # Set icon based on status
        self.update_appearance()
        
    def update_status(self, status, details=None):
        """Update the status of the task."""
        self.status = status
        if details:
            self.details.update(details)
        self.setText(f"{self.task_name} - {status}")
        self.update_appearance()
        
    def update_appearance(self):
        """Update the item's appearance based on its status."""
        # Set text color based on status
        if self.status == "Completed":
            self.setForeground(QColor(0, 128, 0))  # Green
        elif self.status == "Failed":
            self.setForeground(QColor(255, 0, 0))  # Red
        elif self.status == "In Progress":
            self.setForeground(QColor(0, 0, 255))  # Blue
        elif self.status == "Queued":
            self.setForeground(QColor(128, 128, 128))  # Gray
        
        # Make the item a bit larger for better visibility
        self.setSizeHint(QSize(self.sizeHint().width(), 40))

class TasksTab(QWidget):
    """
    Tab for displaying and managing analysis tasks.
    """
    
    # Signal emitted when a task is selected
    task_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize task storage
        self.tasks = {}  # Dictionary to store tasks by ID
        self.current_task_id = None
        
        # Create UI components
        self.create_ui()
        
    def create_ui(self):
        """Create the UI components for the Tasks tab."""
        layout = QVBoxLayout(self)
        
        # Create a splitter for task list and task details
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Task list
        task_list_widget = QWidget()
        task_list_layout = QVBoxLayout(task_list_widget)
        
        # Task list header
        task_list_header = QLabel("Analysis Tasks")
        task_list_header.setStyleSheet("font-size: 14pt; font-weight: bold;")
        task_list_layout.addWidget(task_list_header)
        
        # Task list
        self.task_list = QListWidget()
        self.task_list.setAlternatingRowColors(True)
        self.task_list.currentItemChanged.connect(self.on_task_selected)
        task_list_layout.addWidget(self.task_list)
        
        # Task actions
        task_actions = QWidget()
        task_actions_layout = QHBoxLayout(task_actions)
        
        self.clear_completed_button = QPushButton("Clear Completed")
        self.clear_completed_button.clicked.connect(self.clear_completed_tasks)
        task_actions_layout.addWidget(self.clear_completed_button)
        
        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.clicked.connect(self.clear_all_tasks)
        task_actions_layout.addWidget(self.clear_all_button)
        
        task_list_layout.addWidget(task_actions)
        
        # Right side - Task details
        task_details_widget = QWidget()
        task_details_layout = QVBoxLayout(task_details_widget)
        
        # Task details header
        self.task_details_header = QLabel("Task Details")
        self.task_details_header.setStyleSheet("font-size: 14pt; font-weight: bold;")
        task_details_layout.addWidget(self.task_details_header)
        
        # Create a scroll area for task details
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Task details content
        self.task_details_content = QWidget()
        self.task_details_content_layout = QVBoxLayout(self.task_details_content)
        
        # Task info group
        self.task_info_group = QGroupBox("Task Information")
        task_info_layout = QGridLayout(self.task_info_group)
        
        self.task_name_label = QLabel("Name:")
        self.task_name_value = QLabel("No task selected")
        task_info_layout.addWidget(self.task_name_label, 0, 0)
        task_info_layout.addWidget(self.task_name_value, 0, 1)
        
        self.task_status_label = QLabel("Status:")
        self.task_status_value = QLabel("")
        task_info_layout.addWidget(self.task_status_label, 1, 0)
        task_info_layout.addWidget(self.task_status_value, 1, 1)
        
        self.task_time_label = QLabel("Time:")
        self.task_time_value = QLabel("")
        task_info_layout.addWidget(self.task_time_label, 2, 0)
        task_info_layout.addWidget(self.task_time_value, 2, 1)
        
        self.task_details_content_layout.addWidget(self.task_info_group)
        
        # Task progress group
        self.task_progress_group = QGroupBox("Progress")
        task_progress_layout = QVBoxLayout(self.task_progress_group)
        
        self.task_progress_bar = QProgressBar()
        self.task_progress_bar.setRange(0, 100)
        self.task_progress_bar.setValue(0)
        task_progress_layout.addWidget(self.task_progress_bar)
        
        self.task_progress_status = QLabel("Not started")
        task_progress_layout.addWidget(self.task_progress_status)
        
        self.task_details_content_layout.addWidget(self.task_progress_group)
        
        # Task results group
        self.task_results_group = QGroupBox("Results")
        task_results_layout = QVBoxLayout(self.task_results_group)
        
        self.task_results_label = QLabel("No results available")
        task_results_layout.addWidget(self.task_results_label)
        
        self.task_details_content_layout.addWidget(self.task_results_group)
        
        # Add spacer at the bottom
        self.task_details_content_layout.addStretch()
        
        # Set the content widget for the scroll area
        scroll_area.setWidget(self.task_details_content)
        task_details_layout.addWidget(scroll_area)
        
        # Add widgets to splitter
        splitter.addWidget(task_list_widget)
        splitter.addWidget(task_details_widget)
        
        # Set initial sizes (40% for list, 60% for details)
        splitter.setSizes([400, 600])
        
        # Add splitter to main layout
        layout.addWidget(splitter)
        
        # Initialize with empty state
        self.update_task_details(None)
        
    def add_task(self, task_name, details=None):
        """
        Add a new task to the list.
        
        Args:
            task_name: Name of the task
            details: Optional dictionary with task details
            
        Returns:
            task_id: ID of the created task
        """
        # Generate a unique task ID
        task_id = f"task_{len(self.tasks) + 1}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create task item
        task_item = TaskItem(task_id, task_name, "Queued", timestamp, details)
        
        # Add to list widget
        self.task_list.addItem(task_item)
        
        # Store in tasks dictionary
        self.tasks[task_id] = task_item
        
        # Select the new task
        self.task_list.setCurrentItem(task_item)
        
        return task_id
        
    def update_task(self, task_id, status, progress=None, status_message=None, results=None):
        """
        Update an existing task.
        
        Args:
            task_id: ID of the task to update
            status: New status of the task
            progress: Optional progress value (0-100)
            status_message: Optional status message
            results: Optional dictionary with task results
        """
        if task_id not in self.tasks:
            return
            
        task_item = self.tasks[task_id]
        
        # Update task details
        details = {}
        if progress is not None:
            details['progress'] = progress
        if status_message is not None:
            details['status_message'] = status_message
        if results is not None:
            details['results'] = results
            
        task_item.update_status(status, details)
        
        # If this is the currently selected task, update the details view
        if self.current_task_id == task_id:
            self.update_task_details(task_item)
            
    def on_task_selected(self, current, previous):
        """Handle task selection in the list."""
        if current is None:
            self.current_task_id = None
            self.update_task_details(None)
            return
            
        self.current_task_id = current.task_id
        self.update_task_details(current)
        self.task_selected.emit(current.task_id)
        
    def update_task_details(self, task_item):
        """Update the task details view with the selected task."""
        if task_item is None:
            # No task selected
            self.task_details_header.setText("Task Details")
            self.task_name_value.setText("No task selected")
            self.task_status_value.setText("")
            self.task_time_value.setText("")
            self.task_progress_bar.setValue(0)
            self.task_progress_status.setText("Not started")
            self.task_results_label.setText("No results available")
            return
            
        # Update task details
        self.task_details_header.setText(f"Task Details: {task_item.task_name}")
        self.task_name_value.setText(task_item.task_name)
        self.task_status_value.setText(task_item.status)
        self.task_time_value.setText(task_item.timestamp)
        
        # Update progress
        progress = task_item.details.get('progress', 0)
        self.task_progress_bar.setValue(progress)
        
        status_message = task_item.details.get('status_message', "")
        self.task_progress_status.setText(status_message if status_message else task_item.status)
        
        # Update results
        results = task_item.details.get('results', {})
        if results:
            results_text = "<ul>"
            for key, value in results.items():
                if isinstance(value, dict):
                    results_text += f"<li><b>{key}:</b><ul>"
                    for k, v in value.items():
                        results_text += f"<li>{k}: {v}</li>"
                    results_text += "</ul></li>"
                else:
                    results_text += f"<li><b>{key}:</b> {value}</li>"
            results_text += "</ul>"
            self.task_results_label.setText(results_text)
            self.task_results_label.setTextFormat(Qt.RichText)
        else:
            self.task_results_label.setText("No results available")
            
    def clear_completed_tasks(self):
        """Clear all completed tasks from the list."""
        for i in range(self.task_list.count() - 1, -1, -1):
            item = self.task_list.item(i)
            if item.status == "Completed":
                # Remove from list and dictionary
                del self.tasks[item.task_id]
                self.task_list.takeItem(i)
                
        # Update details view if needed
        if self.current_task_id not in self.tasks:
            self.current_task_id = None
            self.update_task_details(None)
            
    def clear_all_tasks(self):
        """Clear all tasks from the list."""
        self.task_list.clear()
        self.tasks.clear()
        self.current_task_id = None
        self.update_task_details(None)