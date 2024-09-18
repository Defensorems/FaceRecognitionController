# -*- coding: utf-8 -*-
"""
Admin panel for face recognition application using Flask.
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Images'
app.config['LOG_FILE'] = 'admin_logs.log'
app.secret_key = 'supersecretkey'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

logging.basicConfig(filename=app.config['LOG_FILE'], level=logging.INFO, format='%(asctime)s - %(message)s')

program_status = {
    "files_loaded": False,
    "files": [],
    "search_in_progress": False,
    "results": []
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def admin_panel():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('admin.html', status=program_status, images=images)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        flash('No files selected for uploading')
        return redirect(url_for('admin_panel'))

    uploaded_files = request.files.getlist('files')
    file_paths = []
    
    if not uploaded_files:
        flash('No valid files found.')
        return redirect(url_for('admin_panel'))

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
            logging.info(f"Image {filename} uploaded successfully.")
        else:
            flash(f'File "{file.filename}" is not allowed. Only .jpg, .jpeg, .png files are accepted.')
            logging.warning(f"Attempt to upload unsupported file: {file.filename}")

    if file_paths:
        program_status["files_loaded"] = True
        program_status["files"] = file_paths
        flash('Images successfully uploaded.')

    return redirect(url_for('admin_panel'))

@app.route('/delete_image/<filename>', methods=['POST'])
def delete_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f'Image {filename} deleted successfully.')
        logging.info(f"Image {filename} deleted.")
    else:
        flash(f'File {filename} not found.')
        logging.warning(f"Attempt to delete non-existent file: {filename}")
    return redirect(url_for('admin_panel'))

@app.route('/rename_image', methods=['POST'])
def rename_image():
    old_name = request.form.get('old_name')
    new_name = request.form.get('new_name')

    if not allowed_file(new_name):
        flash(f'New name must end with .jpg, .jpeg, or .png')
        logging.warning(f"Invalid file extension in renaming attempt: {new_name}")
        return redirect(url_for('admin_panel'))

    old_path = os.path.join(app.config['UPLOAD_FOLDER'], old_name)
    new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        flash(f'Renamed {old_name} to {new_name}.')
        logging.info(f"Image {old_name} renamed to {new_name}.")
    else:
        flash(f'File {old_name} not found.')
        logging.warning(f"Attempt to rename non-existent file: {old_name}")
    
    return redirect(url_for('admin_panel'))

@app.route('/logs')
def view_logs():
    log_file_path = 'admin_logs.log'
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            log_content = log_file.read()
    else:
        log_content = "Log file not found."
    return render_template('logs.html', logs=log_content)

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/start_search', methods=['POST'])
def start_search():
    search_query = request.form.get('search_query')

    if not search_query:
        flash('No search query provided.')
        return redirect(url_for('admin_panel'))

    if not program_status["files_loaded"]:
        flash('No images loaded. Please upload files first.')
        return redirect(url_for('admin_panel'))

    program_status["search_in_progress"] = True
    program_status["results"] = [
        {"file": "example.jpg", "relevancy": 95, "top_matches": "Sample match content"}
    ]
    program_status["search_in_progress"] = False

    flash('Search completed successfully.')
    return redirect(url_for('admin_panel'))

@app.route('/results', methods=['GET'])
def get_results():
    return jsonify(program_status["results"])

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    if not os.path.exists(app.config['LOG_FILE']):
        open(app.config['LOG_FILE'], 'w').close()

    app.run(host='0.0.0.0', port=5000)