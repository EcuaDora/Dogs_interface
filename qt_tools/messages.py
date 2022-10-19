## Строки с ошибками валидации

# Плохая ячейка
def get_bad_cell_error_string(cell_number, row_number, cell_type):
    return f'🐶: Cell {cell_number} in row {row_number} could not be converted to {str(cell_type)}'

# Плохое расширение
bad_extension_error_string = '🐶: File has illegal extension'


# Файл отсутствуетттттт
file_doesnt_exist_error_string = '🐶: Such file doesnt exist'


# Пустой файл
empty_file_error_string = '🐶: File is empty'
# Привет и ку всем!

# Неправильное количество ячеек в строке
def get_wrong_cells_amount_error_string(row_number):
    return f'🐶: Wrong amount of cells in string {row_number}'

# Неправильно количество колонок во всем файле (допустмое количество - 2 или 3)
wrong_columns_amount_error_string = '🐶: Wrong amount of columns in file'




## Добовление файла

# Файл уже добавлен
file_already_added_error_string = '🐶: File is already added'

# Файл добавлен (все норм)
file_added_success_string  = '🐶: File is added'



## Всякое другое

# Приветствие
greeting_message_string = """

🐶: Woof - Woof!
"""


# Строка с выбраными файлами
def  get_chosen_files_label(files_amount):
    return f'Chosen Files:   ({files_amount})'
