from PyQt5 import QtWidgets, uic, QtCore
import sys
from validate_csv_file import validate_csv
from itertools import count

analysis_types = {
    'type_1': 'Analysis (conventional)',
    'type_2': 'Analysis (model based)',
    'type_3': 'Analysis (complicated)'
}


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('first_window.ui', self)

        self.confirm_button = self.findChild(QtWidgets.QPushButton, 'PushButtonConfirm')
        self.confirm_button.clicked.connect(self.ConfirmButtonPressed)

        self.add_file_button = self.findChild(QtWidgets.QPushButton, 'AddFile')
        self.add_file_button.clicked.connect(self.AddFileToList)


        self.accpept_file = self.findChild(QtWidgets.QDialogButtonBox, 'AccpeptFile')
        self.accpept_file.accepted.connect(self.AcceptFileButtonPresse)
        self.accpept_file.rejected.connect(self.RejectFileButtonPresse)


        self.chosen_flies_label = self.findChild(QtWidgets.QLabel, 'ChosenFilesLabel')

        self.chosen_flies = []

        self.file_search_line = self.findChild(QtWidgets.QLineEdit, 'FileSearchLineEdit')

        self.files_list = self.findChild(QtWidgets.QListWidget, 'FilesList')
        self.files_list.itemDoubleClicked.connect(self.RemoveFileFromList)



        self.analysis_type_button_1 = self.findChild(QtWidgets.QRadioButton, 'AnalysisType1')
        self.analysis_type_button_2 = self.findChild(QtWidgets.QRadioButton, 'AnalysisType2')
        self.analysis_type_button_3 = self.findChild(QtWidgets.QRadioButton, 'AnalysisType3')


        self.analysis_type = None
        self.analysis_type_buttons = QtWidgets.QButtonGroup()
        self.analysis_type_buttons.addButton(self.analysis_type_button_1)
        self.analysis_type_buttons.addButton(self.analysis_type_button_2)
        self.analysis_type_buttons.addButton(self.analysis_type_button_3)

        self.analysis_type_buttons.buttonClicked.connect(self.change_analysis_type)

        self.show()


    def update_files_amount_label(self):
        self.chosen_flies_label.setText(f'Выбранные Файлы:   ({len(self.chosen_flies)})')

    def change_analysis_type(self, button):
        self.analysis_type = button.toolTip()


    def ConfirmButtonPressed(self):
        print(f'Идет анализ типа {analysis_types.get(self.analysis_type)} ')


    def AcceptFileButtonPresse(self):
        file_path = self.file_search_line.text()
        validation_result = validate_csv(file_path)

        if validation_result:
            print(validation_result)
            return

        if file_path in self.chosen_flies:
            #TODO: Вывести ошибку
            return


        self.chosen_flies.append(file_path)
        self.f = QtWidgets.QListWidgetItem(file_path, self.files_list)
        self.update_files_amount_label()


    def RejectFileButtonPresse(self):
        self.file_search_line.clear()


    def AddFileToList(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '.', 'SCV File (*.csv)')
        self.file_search_line.clear()
        self.file_search_line.insert(file_path)


    def RemoveFileFromList(self, item):
        for row in count():
            list_item = self.files_list.item(row)

            if list_item == None:
                break

            if item.text() == list_item.text():
                self.files_list.takeItem(row)
                self.chosen_flies.remove(item.text())
                self.update_files_amount_label()




def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()


if __name__ == '__main__':
    main()