import os
import csv
from itertools import count
from .messages import *


def check_files_names(file_paths, drug_name):
    drug_groups = {}
    for file_path in file_paths:
        try:
            file_name = os.path.basename(file_path)
            file_drug_name, file_group_name, file_number = file_name.split("_")
            if file_drug_name != drug_name:
                raise Exception
            if file_group_name in drug_groups.keys():
                drug_groups[file_group_name].add(file_path)
            else:
                drug_groups[file_group_name] = {
                    file_path,
                }
        except Exception:
            return file_name_wrong_format(file_name, drug_name)

    return drug_groups


def validate_csv(file_path):

    with open(file_path, newline="") as csv_file:
        reader = iter(csv.reader(csv_file, delimiter=",", quotechar='"'))

        try:
            first_row = next(reader)
        except StopIteration:
            return empty_file_error_string

        col_amount = len(first_row)

        scv_formats = [["x", "y"], ["midbody_x", "midbody_y"]]

        for scv_format in scv_formats:
            if scv_format[0] in first_row and scv_format[1] in first_row:
                coord_indexes = {
                    "x": first_row.index(scv_format[0]),
                    "y": first_row.index(scv_format[1]),
                }
                break

        else:
            return wrong_columns_amount_error_string

        for current_row_number in count(1):

            try:
                row = next(reader)

                try:
                    float(row[coord_indexes["x"]])
                except ValueError:
                    return 1, get_bad_cell_error_string(
                        mask["x"], current_row_number, float
                    )

                try:
                    float(row[coord_indexes["y"]])
                except ValueError:
                    return 1, get_bad_cell_error_string(
                        mask["y"], current_row_number, float
                    )

            except StopIteration:
                return 0, scv_format
