import os
import csv
from itertools import count

def does_file_exists(file_path):
    if not os.path.isfile(file_path):
       return 'Файла не существет'

    file_expansion = os.path.splitext(file_path)[-1]

    if file_expansion == '.csv':
       return False



def validate_csv(file_path):

    file_check = does_file_exists(file_path)
    if file_check:
        return file_check

    with open(file_path, newline='') as csv_file:
        reader = iter(csv.reader(csv_file, delimiter=',', quotechar='"'))

        try:
            first_row = next(reader)
        except StopIteration:
            return 'Файл - пустой'

        col_amount = len(first_row)

        if col_amount == 3:
            col_types = (int, float, float)

        elif col_amount == 2:
            col_types = (float, float)

        else:
            return 'Неправильное количество колонок в файле'


        for current_row_number in count(1):

            try:
                row = next(reader)
                if len(row) != col_amount:
                    return f'Неправильное количество столбцов в строке {current_row_number}'

                for cell_number, cell_type, cell in zip(count(1), col_types, row):
                    try:
                        cell_type(cell)
                    except ValueError:
                        return f'Ячейка {cell_number} в строке {current_row_number} не приводится к типу {str(cell_type)}'

            except StopIteration:
                return 0
