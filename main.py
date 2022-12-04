from copyreg import pickle
from mimetypes import init
import os
from shutil import rmtree
import platform
import sys
from utilites import conventional_analysis, model_analysis
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from qt_tools.messages import *
from qt_tools.validate_csv_file import validate_csv, check_files_names
from pprint import pprint
from collections import OrderedDict
from functools import partial
import time

if platform.system() == "Linux":
    os.environ["QT_QPA_PLATFORM"] = "wayland"


# To Do:
# 1. Если не выбран анализ и нажата кнопка готов, то ничего не делать, а вывести в окне для сообщений "Выбери тип анализа"
# 2. Подвинуть текст сообщения немного вниз
# 3. Поменять шрифт сообщению
# 4. в белом прямоугольнике выводить только название
# 5. в выбранных файлах поменять шрифт, цвет и сократить до названий


analysis_types = {
    "type_1": "Analysis (conventional)",
    "type_2": "Analysis (model based)",
    "type_3": "Analysis (complicated)",
}

FIRST_WINDOW_PATH = os.path.join("interfaces", "first_window.ui")
SECOND_WINDOW_PATH = os.path.join("interfaces", "second_window.ui")
THIRD_WINDOW_PATH = os.path.join("interfaces", "third_window.ui")

DEFAULT_DRUGS = []

DRUGS_DATA = {}
FILES_MASKS = {}
ANALYSIS_TYPE = None
TARGET_PATH = os.path.join(os.getcwd(), "data", "original", "parameters")
VISUAL_TARGET_PATH = os.path.join(os.getcwd(), "data", "original", "visualization")


class ModelAnalysisThread(QtCore.QThread):
    def run(self):
        model_analysis(DRUGS_DATA, TARGET_PATH, VISUAL_TARGET_PATH, FILES_MASKS)


class ConvAnalysisThread(QtCore.QThread):
    def run(self):
        conventional_analysis(DRUGS_DATA, TARGET_PATH, VISUAL_TARGET_PATH, FILES_MASKS)


def sort_by_groups_names(group_name):
    if group_name == "control":
        return -1

    return int(group_name.replace("mg", ""))


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi(FIRST_WINDOW_PATH, self)

        self.setWindowTitle("Analysis")

        self.confirm_button = self.findChild(QtWidgets.QPushButton, "PushButtonConfirm")
        self.confirm_button.clicked.connect(self.confirm_button_pressed)

        self.ready_for_2_window = self.findChild(
            QtWidgets.QPushButton, "CrunchButton"
        )  # Костыльное решение для отправки сигнала о том что все гуд и можно открывать 2ое окно

        self.add_file_button = self.findChild(QtWidgets.QPushButton, "AddFile")
        self.add_file_button.clicked.connect(self.add_file_to_list)

        self.drug_chooser = self.findChild(QtWidgets.QComboBox, "DrugsChooser")
        self.drug_chooser.addItems(DEFAULT_DRUGS)
        self.drug_chooser.currentTextChanged.connect(self.update_files_list)
        self.chosen_drug = self.drug_chooser.currentText()
        self.drugs_files = {drug: {} for drug in DEFAULT_DRUGS}
        self.files_by_visualized_names = {}
        self.csv_masks_by_files = {}

        self.add_drug_button = self.findChild(QtWidgets.QPushButton, "AddDrug")
        self.add_drug_button.clicked.connect(self.add_drug)
        self.new_drug_input_line = self.findChild(QtWidgets.QLineEdit, "NewDrugInput")

        self.chosen_flies_label = self.findChild(QtWidgets.QLabel, "ChosenFilesLabel")

        self.files_list = self.findChild(QtWidgets.QListWidget, "FilesList")
        self.files_list.itemDoubleClicked.connect(self.remove_file_from_list)

        self.analysis_type_button_1 = self.findChild(
            QtWidgets.QRadioButton, "AnalysisType1"
        )
        self.analysis_type_button_2 = self.findChild(
            QtWidgets.QRadioButton, "AnalysisType2"
        )
        self.analysis_type_button_3 = self.findChild(
            QtWidgets.QRadioButton, "AnalysisType3"
        )

        self.analysis_type = None
        self.analysis_type_buttons = QtWidgets.QButtonGroup()
        self.analysis_type_buttons.addButton(self.analysis_type_button_1)
        self.analysis_type_buttons.addButton(self.analysis_type_button_2)
        self.analysis_type_buttons.addButton(self.analysis_type_button_3)
        self.analysis_type_buttons.buttonClicked.connect(self.change_analysis_type)

        self.message_text_field = self.findChild(QtWidgets.QTextEdit, "MessageTextArea")

        self.update_files_amount_label()
        self.send_user_message(greeting_message_string, 12)

        self.show()

    @staticmethod
    def sort_by_groups_names(group_name):
        if group_name == "control":
            return -1

        return int(group_name.replace("mg", ""))

    def send_user_message(self, message, font_size=10):
        self.message_text_field.clear()
        self.message_text_field.setAlignment(QtCore.Qt.AlignCenter)
        self.message_text_field.setFont(QtGui.QFont("MS Shell Dlg 2", font_size))
        self.message_text_field.insertPlainText(message)

    def add_drug(self):
        new_drug_name = self.new_drug_input_line.text()
        if not new_drug_name:
            return
        if new_drug_name in self.drugs_files.keys():
            self.send_user_message(drug_already_exist_error_string)
            return

        self.drugs_files[new_drug_name] = {}
        self.drug_chooser.addItem(new_drug_name)
        self.new_drug_input_line.clear()
        self.drug_chooser.setCurrentIndex(self.drug_chooser.count() - 1)
        self.send_user_message(drug_added_success_string)

    def update_files_amount_label(self):
        if not self.drug_chooser.currentText():
            return
        chosen_files_label = get_chosen_files_label(
            len(self.drugs_files[self.chosen_drug])
        )
        self.chosen_flies_label.setText(chosen_files_label)

    def update_files_list(self):
        self.files_list.clear()
        current_drug = self.drug_chooser.currentText()
        drug_files = self.drugs_files[current_drug]
        for group_name, group_files in drug_files.items():
            QtWidgets.QListWidgetItem(group_name, self.files_list)
            for file_name in group_files:
                if len(file_name) > 40:
                    truncated_file_name = file_name[-40:]
                    list_item_label = f"   📄 ...{truncated_file_name}"
                else:
                    list_item_label = f"   📄 {file_name}"

                QtWidgets.QListWidgetItem(list_item_label, self.files_list)
                self.files_by_visualized_names[list_item_label] = file_name

    def change_analysis_type(self, button):
        global ANALYSIS_TYPE
        self.analysis_type = button.toolTip()
        self.send_user_message(
            get_analysis_type_is_chosen_success_string(
                analysis_types.get(self.analysis_type)
            )
        )
        ANALYSIS_TYPE = self.analysis_type

    def confirm_button_pressed(self):
        global DRUGS_DATA
        global FILES_MASKS

        analysis_type = analysis_types.get(self.analysis_type)
        if not analysis_type:
            self.send_user_message(analysis_type_is_not_chosen)
            return

        drugs_data = {}

        for drug_name, drug_groups in self.drugs_files.items():
            if not drug_name:
                del self.drugs_files[drug_name]
                continue

            if "control" not in drug_groups.keys():
                self.send_user_message(drug_doesnt_have_control_group(drug_name))
                continue

            drug_data = OrderedDict(drug_groups)
            group_names = drug_data.keys()
            group_names = sorted(group_names, key=self.sort_by_groups_names)

            for group_name in group_names:
                drug_data.move_to_end(group_name)

            drugs_data[drug_name] = drug_data

        if not drugs_data:
            return

        try:
            os.makedirs(TARGET_PATH)
        except FileExistsError:
            rmtree(TARGET_PATH)
            os.makedirs(TARGET_PATH)

        try:
            os.makedirs(VISUAL_TARGET_PATH)
        except FileExistsError:
            rmtree(VISUAL_TARGET_PATH)
            os.makedirs(VISUAL_TARGET_PATH)

        self.send_user_message(
            get_analysis_started_string(analysis_types.get(self.analysis_type))
        )

        DRUGS_DATA = drugs_data
        FILES_MASKS = self.csv_masks_by_files

        self.confirm_button.setEnabled(False)
        self.add_file_button.setEnabled(False)
        self.drug_chooser.setEnabled(False)
        self.drug_chooser.setEnabled(False)
        self.new_drug_input_line.setEnabled(False)
        self.files_list.setEnabled(False)
        self.add_drug_button.setEnabled(False)
        self.analysis_type_button_1.setEnabled(False)
        self.analysis_type_button_2.setEnabled(False)
        self.analysis_type_button_3.setEnabled(False)

        self.ready_for_2_window.clicked.emit()

    def add_file_to_list(self):
        current_drug = self.drug_chooser.currentText()
        if not current_drug:
            self.send_user_message(no_chosen_drug)
            return

        file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Открыть файл", ".", "SCV File (*.csv)"
        )
        files_by_groups = check_files_names(file_paths, current_drug)
        if type(files_by_groups) == str:
            self.send_user_message(files_by_groups)
            return

        if current_drug not in self.drugs_files:
            self.drugs_files[current_drug] = {}

        for group_name, file_paths in files_by_groups.items():

            if group_name not in self.drugs_files[current_drug]:
                self.drugs_files[current_drug][group_name] = set()

            for file_path in file_paths:
                validation_result = validate_csv(file_path)

                if validation_result[0]:
                    self.send_user_message(validation_result[1])
                    return

                file_mask = validation_result[1]

                self.drugs_files[current_drug][group_name].add(file_path)
                self.csv_masks_by_files[file_path] = file_mask

        self.update_files_list()
        self.send_user_message(files_added_success_string)

    def remove_file_from_list(self, item):
        current_drug = self.drug_chooser.currentText()
        if item.text() in self.drugs_files[current_drug].keys():
            group_name = item.text()
            del self.drugs_files[current_drug][group_name]
            self.update_files_list()
            self.send_user_message(get_group_deletion_success_string(group_name))
            return

        file_to_remove = self.files_by_visualized_names[item.text()]

        drug_group = os.path.basename(file_to_remove).split("_")[1]
        self.drugs_files[current_drug][drug_group].remove(file_to_remove)
        self.update_files_list()
        self.send_user_message(get_file_deletion_success_string(file_to_remove))


class Ui_Dialog(QtWidgets.QDialog):
    def __init__(self):
        super(Ui_Dialog, self).__init__()
        uic.loadUi(SECOND_WINDOW_PATH, self)
        self.graphicsView = self.findChild(QtWidgets.QGraphicsView, "graphicsView")
        self.textBrowser = self.findChild(QtWidgets.QTextBrowser, "textBrowser")
        self.label = self.findChild(QtWidgets.QLabel, "label")
        self.setFixedSize(1200, 800)

        self.initUI()
        self.setWindowTitle("Analysis results")

    def initUI(self):

        scr = QtWidgets.QScrollArea(self)
        scr.setFixedSize(1200, 730)
        scr.move(0, 70)
        pnl = QtWidgets.QDialog(self)

        vbox = QtWidgets.QGridLayout(self)
        images_dir = os.path.join("data", "original", "visualization")
        images = os.listdir(images_dir)
        amount = len(images)

        j = 1
        k = 1
        n = 1  # количество картинок в строке

        for i in range(amount):

            pxm_path = os.path.join("data/original/visualization", images[i])
            name_lbl = QtWidgets.QLabel()
            # but = QtWidgets.QPushButton()

            self.pxm = QtGui.QPixmap(pxm_path)
            name_lbl.setScaledContents(1)

            pxm_width = self.pxm.width()

            if pxm_width > 1000:
                self.pxm = self.pxm.scaledToWidth(1160 / 2)

            elif pxm_width > 366:
                self.pxm = self.pxm.scaledToWidth(285)

            elif pxm_width < 100:
                self.pxm = self.pxm.scaledToWidth(285)

            # but.setSizePolicy(self.pxm.width(), self.pxm.height())
            name_lbl.resize(self.pxm.width(), self.pxm.height())

            #  but.setFlat(True)
            #   but.setStyleSheet("background-color: white")
            #   but.setAutoFillBackground(1)

            if i == 0:
                n = int(1140 // self.pxm.width())

            name_lbl.setBackgroundRole(QtGui.QPalette.Dark)
            name_lbl.setPixmap(self.pxm)

            # тк i = [0 1 2 ... amount - 1], введем доп переменную p
            p = i + 1

            if p % n == 0:
                vbox.addWidget(name_lbl, j, k)
                #    vbox.addWidget(but, j, k)
                j = j + 1
                k = 0
            else:
                vbox.addWidget(name_lbl, j, k)
            #   vbox.addWidget(but, j, k)

            k = k + 1
            name_lbl.mousePressEvent = partial(self.emit_zoom, pic=pxm_path)
        #  but.mouseDoubleClickEvent = partial(self.emit_zoom, pic = pxm_path)

        pnl.setLayout(vbox)
        scr.setWidget(pnl)
        self.show()

    def emit_zoom(self, event, pic):
        global PIC_TO_SHOW
        PIC_TO_SHOW = pic

        self.zoomed_window = Ui_Dialog_graphics(PIC_TO_SHOW)
        self.zoomed_window.show()


class Ui_Dialog_graphics(QtWidgets.QDialog):
    def __init__(self, pic):
        super(Ui_Dialog_graphics, self).__init__()
        uic.loadUi(THIRD_WINDOW_PATH, self)
        self.setWindowTitle("Zoomed graphic")

        pic_path = pic

        label = self.findChild(QtWidgets.QLabel, "label")

        pxm = QtGui.QPixmap(pic_path)

        label.setPixmap(pxm)
        label.resize(pxm.width(), pxm.height())
        label.setScaledContents(1)

        self.setFixedSize(pxm.width() + 30, pxm.height() + 30)


class MyWin(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWin, self).__init__()

    def start_analysis(self):
        # print(ANALYSIS_TYPE)

        if ANALYSIS_TYPE == "type_1":
            self.thread = ConvAnalysisThread()
            self.thread.finished.connect(self.show_window_2)
            self.thread.start()

        if ANALYSIS_TYPE == "type_2":
            self.thread = ModelAnalysisThread()
            self.thread.finished.connect(self.show_window_2)
            self.thread.start()

    def show_window_1(self):
        self.w1 = Ui()

        self.w1.ready_for_2_window.clicked.connect(self.start_analysis)
        self.w1.show()

    def show_window_2(self):
        self.w1.close()
        self.w2 = Ui_Dialog()
        self.w2.show()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MyWin()
    w.show_window_1()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
