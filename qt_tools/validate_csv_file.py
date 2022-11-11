import os
import csv
from itertools import count
from .messages import *


def check_files_names(file_paths, drug_name):
    drug_groups = {}
    for file_path in file_paths:
        try:
            file_name = os.path.basename(file_path)
            file_drug_name, file_group_name, file_number = file_name.split('_')
            if file_drug_name != drug_name:
                raise Exception
            if file_group_name in drug_groups.keys():
                drug_groups[file_group_name].append(file_path)
            else:
                drug_groups[file_group_name] = [file_path]
        except Exception:
            return file_name_wrong_format(file_name, drug_name)
    return drug_groups



def validate_csv(file_path):

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
