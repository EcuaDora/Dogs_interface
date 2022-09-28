import os

#FIXME: Затычка
def validate_csv(file_path):

    if not os.path.isfile(file_path):
        return False

    file_expansion = os.path.splitext(file_path)[-1]
    if file_expansion == '.csv':
        return True