import os
import csv
from itertools import count
from .messages import *


def does_file_exists(file_path):
    if not os.path.isfile(file_path):
       return file_doesnt_exist_error_string

    file_expansion = os.path.splitext(file_path)[-1]

    if file_expansion != '.csv':
        return bad_extension_error_string

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
            return empty_file_error_string

        col_amount = len(first_row)

        if col_amount == 3:
            col_types = (int, float, float)

        elif col_amount == 2:
            col_types = (float, float)

        else:
            return wrong_columns_amount_error_string


        for current_row_number in count(1):

            try:
                row = next(reader)
                if len(row) != col_amount:
                    return get_wrong_cells_amount_error_string(current_row_number)

                for cell_number, cell_type, cell in zip(count(1), col_types, row):
                    try:
                        cell_type(cell)
                    except ValueError:
                        error_message = get_bad_cell_error_string(cell_number, current_row_number, cell_type)
                        return error_message

            except StopIteration:
                return 0
