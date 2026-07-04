from pathlib import Path

UPLOAD_FOLDER = Path("data/uploads")


def save_uploaded_files(uploaded_files):

    
    saved_paths = []
    skipped_files = []

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:

        file_path = UPLOAD_FOLDER / uploaded_file.name

        if file_path.exists():

            skipped_files.append(uploaded_file.name)

            continue

        with open(file_path, "wb") as f:

            f.write(uploaded_file.getbuffer())

        saved_paths.append(str(file_path))

    return saved_paths, skipped_files