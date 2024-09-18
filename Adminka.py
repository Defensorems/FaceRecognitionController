from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_from_directory
from werkzeug.utils import secure_filename
from flask_wtf import CSRFProtect
import os
import logging
from Main1 import GroupManager  # Импортируем класс из скрипта Main1

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Images'
app.config['LOG_FILE'] = 'admin_logs.log'
app.secret_key = 'supersecretkey'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

logging.basicConfig(filename=app.config['LOG_FILE'], level=logging.INFO, format='%(asctime)s - %(message)s')

# Инициализируем CSRF защиту
csrf = CSRFProtect(app)

# Глобальный объект для управления группами
group_manager = GroupManager()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def admin_panel():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    groups = group_manager.groups  # Получаем словарь групп из GroupManager

    # Находим лица без группы
    all_faces = set(images)
    grouped_faces = set()
    for members in groups.values():
        grouped_faces.update(members)
    
    ungrouped_faces = all_faces - grouped_faces  # Лица, не входящие в группы

    return render_template('admin.html', images=images, groups=groups, ungrouped_faces=ungrouped_faces)


@app.route('/create_group', methods=['POST'])
def create_group():
    group_name = request.form.get('group_name')
    if group_name:
        if group_name in group_manager.groups:
            flash(f'Group "{group_name}" already exists.')
        else:
            group_manager.create_group(group_name)
            flash(f'Group "{group_name}" created successfully.')
            logging.info(f'Group "{group_name}" created.')
    else:
        flash("Group name cannot be empty.")
    return redirect(url_for('admin_panel'))

@app.route('/delete_group/<group_name>', methods=['POST'])
def delete_group(group_name):
    if group_name in group_manager.groups:
        group_manager.delete_group(group_name)
        flash(f'Group "{group_name}" deleted successfully.')
        logging.info(f'Group "{group_name}" deleted.')
    else:
        flash(f'Group "{group_name}" not found.')
    return redirect(url_for('admin_panel'))

@app.route('/rename_group', methods=['POST'])
def rename_group():
    old_name = request.form.get('old_name')
    new_name = request.form.get('new_name')
    
    if old_name in group_manager.groups:
        group_manager.rename_group(old_name, new_name)
        flash(f'Group "{old_name}" renamed to "{new_name}".')
        logging.info(f'Group "{old_name}" renamed to "{new_name}".')
    else:
        flash(f'Group "{old_name}" not found.')
    return redirect(url_for('admin_panel'))

@app.route('/add_face_to_group', methods=['POST'])
def add_face_to_group():
    group_name = request.form.get('group_name')
    face_name = request.form.get('face_name')
    
    if group_name in group_manager.groups:
        if face_name not in group_manager.groups[group_name]:
            group_manager.add_face_to_group(face_name, group_name)
            flash(f'Face "{face_name}" added to group "{group_name}".')
            logging.info(f'Face "{face_name}" added to group "{group_name}".')
        else:
            flash(f'Face "{face_name}" is already in group "{group_name}".')
    else:
        flash(f'Group "{group_name}" not found.')
    return redirect(url_for('admin_panel'))

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

@app.route('/move_face_to_group', methods=['POST'])
@csrf.exempt  # Исключение CSRF для этого маршрута, если нужно
def move_face_to_group():
    face_name = request.json.get('face_name')
    new_group_name = request.json.get('group_name')

    logging.info(f"Received face_name: {face_name}, group_name: {new_group_name}")

    if not face_name or not new_group_name:
        return jsonify(success=False, message="Invalid data"), 400

    current_group_name = group_manager.get_group_of_face(face_name)

    if current_group_name and face_name in group_manager.groups.get(current_group_name, []):
        group_manager.groups[current_group_name].remove(face_name)

    if new_group_name == 'ungrouped':
        logging.info(f'Face "{face_name}" moved to ungrouped.')
    else:
        if new_group_name not in group_manager.groups:
            group_manager.create_group(new_group_name)
        group_manager.add_face_to_group(face_name, new_group_name)
        logging.info(f'Face "{face_name}" moved to group "{new_group_name}".')

    return jsonify(success=True)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    if not os.path.exists(app.config['LOG_FILE']):
        open(app.config['LOG_FILE'], 'w').close()

    app.run(host='0.0.0.0', port=5000)
