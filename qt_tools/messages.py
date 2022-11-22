## Строки с ошибками валидации

# Плохая ячейка
def get_bad_cell_error_string(cell_number, row_number, cell_type):
    return f'Cell {cell_number} in row {row_number} could not be converted to {str(cell_type)}'

# Плохое расширение
bad_extension_error_string = 'File has illegal extension'


# Добовляемое лекарство уже существует
drug_already_exist_error_string = 'This drug is already added!'

# Файла нету ваще 
file_doesnt_exist_error_string = 'Such file doesnt exist'


# Пустой файл
empty_file_error_string = 'File is empty'
# Привет и ку всем!

# Неправильное количество ячеек в строке
def get_wrong_cells_amount_error_string(row_number):
    return f'Wrong amount of cells in string {row_number}'


def file_name_wrong_format(file_path, drug_name):
    return f"File {file_path} doesnt match template: {drug_name}_<group_name>_<number>.csv"

def drug_doesnt_have_control_group(drug):
    return f"Drug {drug} doesnt have control group"

# Неправильно количество колонок во всем файле (допустмое количество - 2 или 3)
wrong_columns_amount_error_string = 'Wrong amount of columns in file'


#Не выбран тип анализа
analysis_type_is_not_chosen = 'Choose analysis type!'


## Добовление файла

# Файл уже добавлен
file_already_added_error_string = 'File is already added'

# Файл добавлен (все норм)
file_added_success_string  = 'File is added'



## Всякое другое

# Приветствие
greeting_message_string = """

Woof - Woof! 🐶
"""


# Строка с выбраными файлами
def  get_chosen_files_label(files_amount):
    return f'SELECTED FILES: ({files_amount})'
