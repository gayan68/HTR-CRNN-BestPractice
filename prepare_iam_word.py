import os 
import sys
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import shutil


# main function - read the xml files and prepare the data
def prepare_iam_data(form_path, xmls_path, splits_path, output_path, pad_size=16, scale=1.0):

    # check if output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

        # check for train / val / test subdirectories
        os.makedirs(os.path.join(output_path, "train"))
        os.makedirs(os.path.join(output_path, "val"))
        os.makedirs(os.path.join(output_path, "test"))

    # check if form directory exists
    if not os.path.exists(form_path):
        print("Form directory does not exist")
        return None
    
    # check if xml directory exists
    if not os.path.exists(xmls_path):
        print("XML directory does not exist")
        return None

    # check if splits directory exists
    if not os.path.exists(splits_path):
        print("Splits directory does not exist")
        return None
    
    # get the list of xml files
    xml_files = os.listdir(xmls_path)

    train_set = np.loadtxt(os.path.join(splits_path, 'train.uttlist'), dtype=str)
    val_set = np.loadtxt(os.path.join(splits_path, 'validation.uttlist'), dtype=str)
    test_set = np.loadtxt(os.path.join(splits_path, 'test.uttlist'), dtype=str)

    gt_words = {'train': [], 'val': [], 'test': []}
    # iterate over the xml files
    for xml_file in tqdm(xml_files):

        # get the file name
        file_name = xml_file.split(".")[0]

        if file_name in train_set:
            subset = "train"
        elif file_name in val_set:
            subset = "val"
        elif file_name in test_set:
            subset = "test"
        else:
            continue

        # get the form file
        #form_file = os.path.join(form_path, file_name + ".png")

        # read the form image with PIL
        #form_img = Image.open(form_file)
            
        # resize to further compress it
        #form_img = form_img.resize((int(form_img.width * scale), int(form_img.height * scale))) #, Image.LANCZOS)
    
    
        # get the xml file
        xml_file = os.path.join(xmls_path, xml_file)

        # use xml parser to read the xml file
        xml_tree = ET.parse(xml_file)
        #h, w = form_img.shape

        #w, h = form_img.size

        # find the <handwritten-part> tag
        handwritten_part = xml_tree.find("handwritten-part")

        # find tags starting with <line ...>
        lines = handwritten_part.findall("line")

        # for each line tag find id, text and bounding box
        for line in lines:

            if line.get("segmentation") == "ok": 

                words = line.findall("word")

                for word in words:
                    # get the word id
                    word_id = word.get("id")

                    # get the word text
                    word_text = word.get("text")
                    word_text = word_text.replace("&amp;", "&")
                    word_text = word_text.replace("&quot;", "\"")
                    word_text = word_text.replace("&apos;", "\'")

                    gt_words[subset].append(word_id + " " + word_text + "\n")

                    pth_split = word_id.split("-")
                    word_file_src = os.path.join(form_path, pth_split[0], pth_split[0] +"-"+ pth_split[1], word_id + ".png")
                    print(word_file_src)
                    word_file_dst = os.path.join(output_path, subset, word_id + ".png")
                    print(word_file_dst)

                    # Copy image from sorce directory to destination directory
                    shutil.copy(word_file_src, word_file_dst)

    # write the gt file
    for subset in gt_words.keys():
        with open(os.path.join(output_path, subset, "gt.txt"), "w") as f:
            f.writelines(gt_words[subset])

    return None



# main call - arguments are the paths to the form and xml directories
if __name__ == '__main__':

    # 1st argument is the path to the form directory
    form_path = sys.argv[1]

    # 2nd argument is the path to the xml directory
    xmls_path = sys.argv[2]

    # 3rd argument is the path to the splits directory
    splits_path = sys.argv[3]

    # 4rth argument is the path to the output directory
    output_path = sys.argv[4]

    # prepare the data
    prepare_iam_data(form_path, xmls_path, splits_path, output_path, pad_size=2, scale=0.5)





