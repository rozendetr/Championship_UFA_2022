# python3
# coding=utf-8
#================================================================
#
#
#   Editor      : PyCharm
#   File name   : post_detect.py
#   Author      : Khomyakov_VV
#   Created date: 2019-10-10
#   Description : Библиотека для обработки постдетектирования объектов
#                 1. Save detect data to xml (for program labelxml)
#
#================================================================

import os
import argparse
from lxml import etree, objectify


def dict2xml(dic):
    """
    Конвертирует словарь (и вложения словаря в словарь) в структуру xml <key>value</key> or <key>dict</key>
    :param dic:
    :return:
    """
    def __subnodes(inner_dic):
        subnodes = []
        for key, value in inner_dic.items():
            elem = etree.Element(str(key))
            if isinstance(value, dict):
                for subnode in __subnodes(value):
                    elem.append(subnode)
            else:
                elem.text = str(value)
            subnodes.append(elem)
        return subnodes
    try:
        root = etree.Element(str(list(dic.keys())[0]))
        if isinstance(list(dic.values())[0], dict):
            for subnode in __subnodes(list(dic.values())[0]):
                root.append(subnode)
        else:
            root.text = str(list(dic.values())[0])
    except Exception as e:
        raise
    return root

def create_list_xml_objects(classIds,
                    confidences,
                    boxes):
    """
    Создание xml структуру из распознаного списка объектов
    Эта xml структура используется для labelImg
    :param classIds:     классы распознанных объектов (list)
    :param confidences:  точность задектированных объектов (list)
    :param boxes:        координаты задетектированных объектов [[],[],...,[]]
    :return: xml структура
    """
    if len(classIds) != len(confidences) != len(boxes):
        print("Error: amount of elements in classes")
        return etree.Element('object')
    else:
        #objects_xml = etree.Element('object')
        #objects_xml = etree.ElementTree(element=None, file=None)
        list_xml = []
        for class_Id, conf_Id, box_Id in zip(classIds, confidences, boxes):
            #Формируем данные для object
            obj = {}
            obj["name"] = str(class_Id)
            obj["pose"] = 'Unspecified'
            obj["truncated"] = '0'
            obj["difficult"] = '0'
            obj["confidience"] = conf_Id
            bndbox={}
            try:
                bndbox["xmin"] = box_Id[0]
                bndbox["ymin"] = box_Id[1]
                bndbox["xmax"] = box_Id[2]
                bndbox["ymax"] = box_Id[3]
#                 bndbox["xmax"] = box_Id[0] + box_Id[2]
#                 bndbox["ymax"] = box_Id[1] + box_Id[3]
            except Exception as e:
                bndbox = {}
            obj["bndbox"] = bndbox
            ###########################
            list_xml.append(dict2xml({'object': obj}))
        return list_xml

def create_annotation_xmlfile(input_file,size_image):
    """
    Создание структуры аннотации xml файла для labelImg
    Эта xml структура используется для labelImg
    :param input_file: путь к входному файлу
    :return: xml структура
    """
    #file_xml = etree.Element()
    try:
        dir_input_file = os.path.dirname(input_file)
        #### Аннотоция файла xml для labelImg ####
        file_dict = {}
        file_dict["folder"]    = dir_input_file.split('/')[-1]
        file_dict["filename"]  = os.path.basename(input_file)
        file_dict["path"]      = input_file
        file_dict["source"]    = {"database" : 'Unknown'}
        file_dict["size"]      = {"height": size_image[0],
                                  "width":  size_image[1],
                                  "depth":  size_image[2]}
        file_dict["segmented"] = 0
    except Exception as e:
        file_dict = {}
   # file_xml.append(dict2xml({'annotation': file_dict}))
  #  file_xml = etree.Element(dict2xml({'': file_dict}))
    file_xml = dict2xml({'annotation': file_dict})
    return file_xml

def save_xml(xml_struct, ouput_file):
    """
    Save xml structure to file
    :param xml_struct: структура xml
    :param ouput_file: путь к сохранению файла
    :return:
    """
    objectify.deannotate(xml_struct)
    etree.cleanup_namespaces(xml_struct)
    # конвертируем все в привычную нам xml структуру.
    obj_xml = etree.tostring(xml_struct,
                             pretty_print=True,
                             xml_declaration=True)
    try:
        with open(ouput_file, "wb") as xml_writer:
            xml_writer.write(obj_xml)
            print('save xml', ouput_file)
    except IOError:
        print('dont save xml')
        pass

def create_xml(output_file,
               input_file,
               image_shape,
               classIds,
               confidences,
               boxes):
    """
    создание xml файла из распознаного изображения

    :param output_file:  путь вывода файла
    :param input_file:   путь исходного файла
    :param image_shape:  размер исходного изображения [weight, height, depth]
    :param classIds:     классы распознанных объектов (list)
    :param confidences:  точность задектированных объектов (list)
    :param boxes:        координаты задетектированных объектов [[],[],...,[]]
    :return:
    """
    xml_file = create_annotation_xmlfile(input_file, list(image_shape))
    list_xml_objects = create_list_xml_objects(classIds, confidences, boxes)
    for xml_object in list_xml_objects:
        xml_file.append(xml_object)

    save_xml(xml_file, output_file)