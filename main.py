import os
import sys
from itertools import count
import math

import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets, uic

from qt_tools.messages import *
from qt_tools.validate_csv_file import validate_csv

# To Do:
# 1. Если не выбран анализ и нажата кнопка готов, то ничего не делать, а вывести в окне для сообщений "Выбери тип анализа"
# 2. Подвинуть текст сообщения немного вниз
# 3. Поменять шрифт сообщению
# 4. в белом прямоугольнике выводить только название
# 5. в выбранных файлах поменять шрифт, цвет и сократить до названий



analysis_types = {
    'type_1': 'Analysis (conventional)',
    'type_2': 'Analysis (model based)',
    'type_3': 'Analysis (complicated)'
}

FIRST_WINDOW_PATH = os.path.join('interfaces', 'first_window.ui')
SECOND_WINDOW_PATH = os.path.join('interfaces', 'second_window.ui')

flag = 0

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi(FIRST_WINDOW_PATH, self)
        

        self.confirm_button = self.findChild(QtWidgets.QPushButton, 'PushButtonConfirm')
        self.confirm_button.clicked.connect(self.confirm_button_pressed)

        self.add_file_button = self.findChild(QtWidgets.QPushButton, 'AddFile')
        self.add_file_button.clicked.connect(self.add_file_to_list)


        self.accpept_file = self.findChild(QtWidgets.QDialogButtonBox, 'AccpeptFile')
        self.accpept_file.accepted.connect(self.accept_file_button_pressed)
        self.accpept_file.rejected.connect(self.reject_file_button_pressed)


        self.chosen_flies_label = self.findChild(QtWidgets.QLabel, 'ChosenFilesLabel')

        self.chosen_flies = []

        self.file_search_line = self.findChild(QtWidgets.QLineEdit, 'FileSearchLineEdit')

        self.files_list = self.findChild(QtWidgets.QListWidget, 'FilesList')
        self.files_list.itemDoubleClicked.connect(self.remove_file_from_list)


        self.analysis_type_button_1 = self.findChild(QtWidgets.QRadioButton, 'AnalysisType1')
        self.analysis_type_button_2 = self.findChild(QtWidgets.QRadioButton, 'AnalysisType2')
        self.analysis_type_button_3 = self.findChild(QtWidgets.QRadioButton, 'AnalysisType3')


        self.analysis_type = None
        self.analysis_type_buttons = QtWidgets.QButtonGroup()
        self.analysis_type_buttons.addButton(self.analysis_type_button_1)
        self.analysis_type_buttons.addButton(self.analysis_type_button_2)
        self.analysis_type_buttons.addButton(self.analysis_type_button_3)
        self.analysis_type_buttons.buttonClicked.connect(self.change_analysis_type)


        self.message_text_field = self.findChild(QtWidgets.QTextEdit, 'MessageTextArea')

        self.update_files_amount_label()
        self.send_user_message(greeting_message_string, 12)

        self.show()


    def send_user_message(self, message, font_size = 10):
        self.message_text_field.clear()
        self.message_text_field.setAlignment(QtCore.Qt.AlignCenter)
        self.message_text_field.setFont(QtGui.QFont('MS Shell Dlg 2', font_size))
        self.message_text_field.insertPlainText(message)


    def update_files_amount_label(self):
        chosen_files_label = get_chosen_files_label(len(self.chosen_flies))
        self.chosen_flies_label.setText(chosen_files_label)

    def change_analysis_type(self, button):
        self.analysis_type = button.toolTip()


    def confirm_button_pressed(self):
        self.send_user_message(f'Идет анализ типа {analysis_types.get(self.analysis_type)}')


    def accept_file_button_pressed(self):
        file_path = self.file_search_line.text()
        validation_result = validate_csv(file_path)

        if validation_result:
            self.send_user_message(validation_result)
            return

        if file_path in self.chosen_flies:
            self.send_user_message(file_already_added_error_string)
            return


        self.chosen_flies.append(file_path)
        self.f = QtWidgets.QListWidgetItem(file_path, self.files_list)
        self.update_files_amount_label()
        self.send_user_message(file_added_success_string)

    def reject_file_button_pressed(self):
        self.file_search_line.clear()

    def add_file_to_list(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '.', 'SCV File (*.csv)')
        self.file_search_line.clear()
        self.file_search_line.insert(file_path)


    def remove_file_from_list(self, item):
        for row in count():
            list_item = self.files_list.item(row)

            if list_item == None:
                break

            if item.text() == list_item.text():
                self.files_list.takeItem(row)
                self.chosen_flies.remove(item.text())
                self.update_files_amount_label()

class Ui_Dialog(QtWidgets.QDialog ):
    def __init__(self):
        super(Ui_Dialog, self).__init__()
        uic.loadUi(SECOND_WINDOW_PATH, self)
       
        # нужно по клику на кнопку ready из главного окна сделать:
           #1. Открыть второе окно
           #2. Отрисовать картинки 
           #3. По окончании поменять текст в сообщении 
        self.graphicsView = self.findChild(QtWidgets.QGraphicsView, 'graphicsView')
        self.textBrowser = self.findChild(QtWidgets.QTextBrowser, 'textBrowser') 
        self.label = self.findChild(QtWidgets.QLabel, 'label')
        self.setFixedSize(1200, 800)
        self.initUI()
        self.setWindowTitle('Analysis results')
    
    
        
    def initUI(self):
        
        
        scr = QtWidgets.QScrollArea(self)
        scr.setFixedSize(1200, 730)
        scr.move(0, 70)
        pnl = QtWidgets.QDialog(self)
        
        vbox = QtWidgets.QGridLayout(self)
        images = os.listdir('photos') 
        amount = len(images)
        
        j = 1
        k = 1
        n = 4 #количество картинок в строке
        
        
  
        for i in range(amount):
            
            pxm_path = os.path.join('photos', images[i])
            
            lbl = QtWidgets.QLabel()
            self.pxm = QtGui.QPixmap(pxm_path)
            
   
            lbl.setBackgroundRole(QtGui.QPalette.Dark)
            lbl.setScaledContents(1)
            lbl.setPixmap(self.pxm)
            
            lbl.mousePressEvent = self.zoom
            
            pixWidth = float(self.pxm.width())
            
            grid_Width  = float(scr.geometry().width())
            factor = float(pixWidth * 2.05/grid_Width)
            lbl.setFixedWidth(int(factor * self.pxm.width()))
            lbl.setFixedHeight(int(factor * self.pxm.height()))
            

            #тк i = [0 1 2 ... amount - 1], введем доп переменную p
            p = i + 1
            
            if p % n == 0 :
                vbox.addWidget(lbl, j, k)
                j = j + 1
                k = 0
            else: 
                vbox.addWidget(lbl, j, k)
                
            k = k + 1
            
        pnl.setLayout(vbox)
        scr.setWidget(pnl)
        self.show()
    
       
     
    def zoom(self, event):
        if flag == 0:
            print ('Label click')
            flag == 1
       
        
    
                  
        
     

      

class MyWin(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWin, self).__init__()

    def show_window_1(self):
        self.w1 = Ui()
        
        self.w1.confirm_button.clicked.connect(self.show_window_2)
        self.w1.confirm_button.clicked.connect(self.w1.close)
        self.w1.show()
    
    def show_window_2(self):
        self.w2 = Ui_Dialog()
        self.w2.show()
 
 
   
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Ui()
   
    sys.exit(app.exec_())
        

if __name__ == '__main__':
    main()
    