from lxml import objectify
import os
import argparse
import sys
import math

def check_fpath(file_path):
    """
    Проверка существование файла
    :param file_path: путь к файлу
    :return:  True - существует, False нет
    """
    if os.path.isfile(file_path):
        return True
    else:
        print('file ',file_path, ' does not exist\n')
        return False

#read List_txt
def read_list(input_path):
    """
    Возвращает список строк и текстового файла
    :param input_path: имя текстового файла
    :return: список строк
    """
    list_str=[]
    if check_fpath(input_path):
        file_txt = open(input_path, 'rt')
        for line in file_txt.readlines():
            list_str.append(line.rstrip('\r\n').lower())        
    return list_str

def parseXML(xmlFile, type_code='ascii'):
    """Parse the XML file
    :param: xmlFile - исходный xml-файл
    :return: Вытаскивает из xml-файла ширину и высоту исходного изображения и объекты
    """
    width_image = 1
    height_image = 1
    with open(xmlFile) as f:
        xml = f.read()
        xml = xml.encode(type_code)
    root = objectify.fromstring(xml)

    objects = []
    for appt in root.getchildren():
        if appt.tag == "size":
            width_image = appt.width
            height_image = appt.height
        if appt.tag == "object":
            object_dict = {}
            for e in appt.getchildren():
                if e.tag == "bndbox":
                    object_dict[e.tag] = {"xmin": e.xmin,
                                          "ymin": e.ymin,
                                          "xmax": e.xmax,
                                          "ymax": e.ymax}
                else:
                    object_dict[e.tag] = e.text
            objects.append(object_dict)
    return width_image, height_image, objects


def rect_to_box(width_image, height_image, rect_list):
    """
    Функция конверта прямоугольника (левый верх и правый нижн угол)
    в box (центр прямоугольника, ширина и высота) в относительных координатах
    :param: width_image - ширина исходного изображения (данные и3 xml)
    :param: height_image - высота исходного изображения (данные и3 xml)
    :param: rect_list - список координат прямоугольника [x_min,y_min,x_max,y_max]
    :return: координаты центра прямогольника, его ширину и высоту
    """
    width = math.fabs((rect_list[2]-rect_list[0]))
    height = math.fabs((rect_list[3]-rect_list[1]))
    x = round((rect_list[0]+width/2)/width_image,6)
    y = round((rect_list[1]+height/2)/height_image,6)
    width_new = round(width/width_image,6)
    height_new = round(height/height_image,6)

#for check
#    x = round((rect_list[0]-width/2),6)
#    y = round((rect_list[1]-height/2),6)
#    width_new = round(width,6)
#    height_new = round(height,6)
    return [x, y, width_new, height_new]


def xmlbox_to_txtbox(list_classes, xmlFile):
    """
    Функция для конвертирования xml файла в строчный блок
    cтруктура строки: имя_класса_объекта координата_х_центра координата_y_центра ширина_прямоугольника высота_прямоугольника
    :param list_classes: список классов
    :param: xmlFile исходный xml файл
    :return: Строчный блок
    """
    string_file=''
    width_image, height_image, objects = parseXML(xmlFile)
    for i in objects:
        temp_list = []
        try:
            temp_list.append(list_classes.index(str(i['name']).lower()))
        except ValueError:
            temp_list.append(i['name'])
        temp_list+=rect_to_box(width_image, height_image, list(i['bndbox'].values())) 
        string_file +=  ' '.join(map(str,temp_list))+'\n'
    return string_file


def check_add_dir(dir_name, arg_dir, name_dir):
    """
    Функция проверки аргумента из argparse
    :param: dir_name - корневая директория
    :param: arg_dir - аргумент из argparse
    :param: name_dir - имя директории
    :return: вывод пути директории, если изначальнго не задано, то директория находится в корневой директории
    """
    if arg_dir == None:
        return dir_name + '\\'+name_dir;
    else:
        return arg_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script for convert xml (labelxml) to txt for YOLO')
    parser.add_argument('-g', type=str, action='store', dest='gts_dir',
                        help="input directory xml")
    parser.add_argument('-t', type=str, action='store', dest='txt_dir',
                        help="output directory txt")
    parser.add_argument('-fClasses', type=str, action='store', dest='file_classes',
                        help="file list names of classes")
    args = parser.parse_args()

    dir_name = os.path.dirname(os.path.abspath(__file__))
    gts_directory = check_add_dir(dir_name, args.gts_dir, 'gts')
    txt_directory = check_add_dir(dir_name, args.txt_dir, 'txt')
    file_Classes = args.file_classes
    print(file_Classes)


    if not os.path.isabs(gts_directory):
        gts_directory = os.path.join(dir_name, gts_directory)

    if not os.path.isabs(txt_directory):
        txt_directory = os.path.join(dir_name, txt_directory)

    if not os.path.isdir(gts_directory):
        print('directory ', gts_directory, 'does not exist')
        sys.exit()

    if not os.path.isdir(txt_directory):
        print('directory ', txt_directory, 'does not exist')
        sys.exit()

    print('gts_directory:', gts_directory)
    print('txt_directory:', txt_directory)



    list_classes = read_list(file_Classes)
    print(list_classes)

    for file in os.listdir(gts_directory):
       if file.endswith(".xml"):
          if os.path.isfile(os.path.join(gts_directory, file)):
              file_name = os.path.basename(os.path.join(gts_directory, file)) #? file_name = file
              file_name = os.path.splitext(file_name)[0]
              
              file_txt = open(os.path.join(txt_directory, file_name+'.txt'),'w')
              try:
                  file_txt.write(xmlbox_to_txtbox(list_classes, os.path.join(gts_directory, file)))
              finally:
                  file_txt.close()
                        
              print(file, "convert: ", file_txt.closed)


