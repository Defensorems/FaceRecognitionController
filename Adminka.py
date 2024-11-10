from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_from_directory
from werkzeug.utils import secure_filename
from flask_wtf import CSRFProtect
import os
import logging
from Main1 import GroupManager  # Import GroupManager class from Main1

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Images'  # Folder where uploaded images will be stored
app.config['LOG_FILE'] = 'admin_logs.log'  # Log file location
app.secret_key = 'supersecretkey'  # Secret key for CSRF protection
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}  # Allowed image formats

# Set up logging to both console and file, ensuring it works across platforms
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(app.config['LOG_FILE'])
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logging.getLogger().addHandler(file_handler)
logging.getLogger().addHandler(console_handler)

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Global object to manage groups
group_manager = GroupManager()

def allowed_file(filename):
    """
    Check if the uploaded file has a valid extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def admin_panel():
    """
    Admin panel that lists all images and groups.
    Displays faces without groups and current groups.
    """
    images = os.listdir(app.config['UPLOAD_FOLDER'])  # List files in the upload folder
    groups = group_manager.groups  # Get groups from GroupManager

    # Find faces that aren't in any group
    all_faces = set(images)
    grouped_faces = set()
    for members in groups.values():
        grouped_faces.update(members)
    
    ungrouped_faces = all_faces - grouped_faces  # Faces not part of any group

    return render_template('admin.html', images=images, groups=groups, ungrouped_faces=ungrouped_faces)


@app.route('/create_group', methods=['POST'])
def create_group():
    """
    Create a new group from the admin panel.
    """
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
    """
    Delete a group from the admin panel.
    """
    if group_name in group_manager.groups:
        group_manager.delete_group(group_name)
        flash(f'Group "{group_name}" deleted successfully.')
        logging.info(f'Group "{group_name}" deleted.')
    else:
        flash(f'Group "{group_name}" not found.')
    return redirect(url_for('admin_panel'))

@app.route('/rename_group', methods=['POST'])
def rename_group():
    """
    Rename a group.
    """
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
    """
    Add a face to a group from the admin panel.
    """
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
    """
    Handle file uploads from the admin panel.
    """
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
    """
    Delete an image from the upload folder.
    """
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
    """
    Rename an image in the upload folder.
    """
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

@app.route('/set_liveness', methods=['POST'])
def set_liveness():
    """
    Set the liveness level for a group.
    """
    group_name = request.form.get('group_name')
    liveness_level = request.form.get('liveness_level', type=int)

    if group_name in group_manager.groups:
        try:
            group_manager.set_liveness_level(liveness_level)
            flash(f'Liveness level for group "{group_name}" set to {liveness_level}.')
            logging.info(f'Liveness level for group "{group_name}" set to {liveness_level}.')
        except Exception as e:
            flash(f'Failed to set liveness level: {str(e)}')
            logging.error(f'Error setting liveness level for group "{group_name}": {str(e)}')
    else:
        flash(f'Group "{group_name}" not found.')
        logging.warning(f'Attempt to set liveness for non-existent group "{group_name}".')

    return redirect(url_for('admin_panel'))

@app.route('/logs', methods=['GET'])
def view_logs():
    log_level = request.args.get('log_level', '')
    search_query = request.args.get('search_query', '')
    selected_log_file = request.args.get('log_file', 'admin_logs.log')  # По умолчанию отображаем admin_logs.log
    page = request.args.get('page', 1, type=int)

    # Получаем все доступные лог-файлы в директории
    log_files = [f for f in os.listdir() if f.endswith('.log')]  # Путь может измениться в зависимости от места расположения логов

    # Открываем выбранный лог-файл с указанием кодировки
    logs = []
    try:
        with open(selected_log_file, 'r', encoding='utf-8', errors='ignore') as file:
            logs = file.readlines()
    except FileNotFoundError:
        logs = []
    except UnicodeDecodeError as e:
        # Логирование ошибки кодировки, если файл не может быть открыт в UTF-8
        app.logger.error(f"Ошибка кодировки при открытии файла {selected_log_file}: {str(e)}")
        logs = []

    # Фильтрация логов по уровню и поисковому запросу
    if log_level:
        logs = [log for log in logs if log_level.upper() in log]

    if search_query:
        logs = [log for log in logs if search_query.lower() in log.lower()]

    per_page = 20
    total_logs = len(logs)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_logs = logs[start:end]

    total_pages = (total_logs + per_page - 1) // per_page
    prev_page = page - 1 if page > 1 else None
    next_page = page + 1 if page < total_pages else None

    # Логика отображения первой и последней страницы
    show_first = page > 2
    show_last = page < total_pages - 1

    page_numbers = list(range(max(1, page - 2), min(page + 3, total_pages + 1)))

    return render_template(
        'logs.html', 
        logs=paginated_logs, 
        current_page=page, 
        prev_page=prev_page, 
        next_page=next_page,
        log_level=log_level, 
        search_query=search_query,
        total_pages=total_pages,
        log_files=log_files,  # Список файлов логов для отображения в селекте
        selected_log_file=selected_log_file,  # Выбранный файл логов
        show_first=show_first, 
        show_last=show_last,
        page_numbers=page_numbers
    )

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/images/<filename>')
def serve_image(filename):
    """
    Serve an image from the upload folder.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/move_face_to_group', methods=['POST'])
@csrf.exempt  # CSRF exemption for this route
def move_face_to_group():
    """
    Move a face to a different group.
    """
    face_name = request.json.get('face_name')
    new_group_name = request.json.get('group_name')

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
